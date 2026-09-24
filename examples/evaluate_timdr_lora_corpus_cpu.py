"""Run the frozen post-training audit for a completed Qwen LoRA source run."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="completed learning_runs/timdr_lora_corpus_cpu/<timestamp>")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "timdr_source_lora_v0.1.json")
    parser.add_argument("--perturbations", type=Path, default=ROOT / "data" / "timdr_lora_perturbation_v0.1.json")
    args = parser.parse_args()
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    plan = json.loads((args.run / "plan.json").read_text(encoding="utf-8"))
    training_report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    if not training_report.get("status", "").startswith("TRAINED_"):
        raise ValueError("Run is not a completed training artifact")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    perturbations = json.loads(args.perturbations.read_text(encoding="utf-8"))
    audit = {"status": "STARTED", "run": str(args.run), "plan_sha256": sha(args.run / "plan.json"),
             "corpus_sha256_current": sha(args.corpus), "corpus_sha256_planned": plan["corpus_sha256"],
             "evaluation_ids_current": [row["id"] for row in corpus["evaluation"]],
             "evaluation_ids_planned": plan["evaluation_chunk_ids"], "checks": {}}
    if audit["corpus_sha256_current"] != audit["corpus_sha256_planned"] or audit["evaluation_ids_current"] != audit["evaluation_ids_planned"]:
        audit["status"] = "INCONCLUSIVE_INPUT_DRIFT"
        save(args.run / "post_training_audit.json", audit)
        raise SystemExit("Frozen corpus or evaluation split drifted.")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep one uniform CPU placement. A device map here makes PEFT attempt an
    # unnecessary offload dispatch while attaching the local adapter.
    base = AutoModelForCausalLM.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, str(args.run / "adapter"), is_trainable=False)
    model.eval()

    representation_rows, base_losses, adapter_losses, exact_reference = [], [], [], True
    for row in corpus["evaluation"]:
        inputs = tokenizer(row["text"], return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=corpus["max_tokens"])
        labels = inputs["input_ids"].clone()
        with torch.no_grad():
            with model.disable_adapter():
                base_first = model(**inputs, labels=labels, output_hidden_states=True, use_cache=False)
                base_second = model(**inputs, labels=labels, output_hidden_states=True, use_cache=False)
            adapted = model(**inputs, labels=labels, output_hidden_states=True, use_cache=False)
        exact_reference = exact_reference and torch.equal(base_first.hidden_states[-1], base_second.hidden_states[-1])
        h_base, h_adapt = base_first.hidden_states[-1][0], adapted.hidden_states[-1][0]
        energy_base, energy_adapt = h_base.square().sum(-1), h_adapt.square().sum(-1)
        delta_base, delta_adapt = h_base[1:] - h_base[:-1], h_adapt[1:] - h_adapt[:-1]
        rho = (energy_adapt - energy_base).abs().mean()
        j = torch.nn.functional.cosine_similarity(delta_adapt, delta_base, dim=-1).mean()
        representation_rows.append({"id": row["id"], "relative_l2": float((h_adapt - h_base).norm() / h_base.norm().clamp_min(1e-12)),
                                    "cosine": float(torch.nn.functional.cosine_similarity(h_adapt.flatten(), h_base.flatten(), dim=0)),
                                    "rho_adapter_energy_only": float(rho), "J_adapter_direction_only": float(j)})
        base_losses.append(float(base_first.loss))
        adapter_losses.append(float(adapted.loss))
    audit["checks"]["reference_path_bitwise_repeat"] = exact_reference
    audit["evaluation"] = {"fragments": len(representation_rows), "base_mean_loss": float(np.mean(base_losses)),
                            "adapter_mean_loss": float(np.mean(adapter_losses)), "loss_delta": float(np.mean(adapter_losses) - np.mean(base_losses)),
                            "representation": representation_rows}

    probe_rows = []
    for probe in perturbations["prompts"]:
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": probe["text"]}], tokenize=True,
                                                add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            logits = model(**inputs, use_cache=False).logits[0, -1]
            probabilities = torch.softmax(logits, -1)
            entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
            output = model.generate(**inputs, max_new_tokens=perturbations["generation_tokens"], do_sample=False,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id)
        generated = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        probe_rows.append({"id": probe["id"], "kind": probe["kind"], "next_token_entropy": entropy, "generated": generated})
    audit["perturbations"] = {"manifest_sha256": sha(args.perturbations), "rows": probe_rows}
    audit["status"] = "AUDIT_COMPLETE" if exact_reference else "INCONCLUSIVE_REFERENCE_NONDETERMINISM"
    audit["note"] = "rho_adapter and J_adapter are source-separated representation-drift diagnostics, not a new empirical TIMDR result."
    save(args.run / "post_training_audit.json", audit)
    print(json.dumps({"status": audit["status"], "loss_delta": audit["evaluation"]["loss_delta"],
                      "reference_repeat": exact_reference, "perturbation_rows": len(probe_rows)}, indent=2))


if __name__ == "__main__":
    main()
