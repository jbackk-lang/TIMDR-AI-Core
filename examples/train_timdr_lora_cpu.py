"""Bounded CPU LoRA feasibility run. No empirical TIMDR verdict is produced."""
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"
DATA = ROOT / "data" / "timdr_text_pilot.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    import numpy as np
    import psutil
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    process = psutil.Process()
    if os.name == "nt":
        try:
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except psutil.AccessDenied:
            pass
    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    if dataset["schema"] != "timdr-text-pilot/1":
        raise ValueError("Unknown dataset schema")
    train_ids = {row["id"] for row in dataset["train"]}
    evaluation_ids = {row["id"] for row in dataset["evaluation"]}
    if train_ids & evaluation_ids:
        raise ValueError("Training and evaluation IDs overlap")
    sources = {row["source"] for split in ("train", "evaluation") for row in dataset[split]}
    source_hashes = {name: sha(ROOT.parent / "GIA-TIMDR" / name) for name in sources}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = ROOT / "learning_runs" / "timdr_lora_cpu" / stamp
    run.mkdir(parents=True, exist_ok=False)
    plan = {"status": "CPU_FEASIBILITY_PILOT", "seed": 42, "epochs": 2,
            "batch_size": 1, "max_sequence_tokens": 192, "learning_rate": 0.0002,
            "lora_r": 4, "lora_alpha": 8, "layers": [26, 27],
            "target_modules": ["q_proj", "v_proj"], "threads": 2, "dtype": "float32",
            "max_training_seconds": 1200, "min_available_GiB": 0.5,
            "max_process_RSS_GiB": 8, "generation_tokens": 32,
            "dataset_sha256": sha(DATA), "script_sha256": sha(Path(__file__)),
            "sources_sha256": source_hashes, "train_ids": sorted(train_ids),
            "evaluation_ids": sorted(evaluation_ids), "evaluation_role": "held-out paraphrases for descriptive comparison only",
            "base_manifest": json.loads((BASE / "DOWNLOAD_MANIFEST.json").read_text())}
    save(run / "plan.json", plan)
    save(run / "dataset.json", dataset)
    report = {"run": str(run), "steps": [], "status": "STARTED"}
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prepare(row):
        messages = [{"role": "user", "content": row["question"]}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}], tokenize=True)
        if full[:len(prefix)] != prefix or len(full) > plan["max_sequence_tokens"]:
            raise ValueError(f"Prompt prefix mismatch or sequence too long: {row['id']} ({len(full)})")
        ids = torch.tensor([full])
        labels = ids.clone()
        labels[:, :len(prefix)] = -100
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

    training = [prepare(row) for row in dataset["train"]]
    evaluation = [prepare(row) for row in dataset["evaluation"]]
    available = psutil.virtual_memory().available / 2**30
    print(f"Run: {run}\nAvailable RAM: {available:.2f} GiB", flush=True)
    if available < 6.5:
        report["status"] = "STOPPED_INSUFFICIENT_FREE_RAM_BEFORE_LOAD"
        save(run / "report.json", report)
        raise SystemExit("Need at least 6.5 GiB available RAM for this float32 pilot.")
    print("Loading Qwen 1.5B on CPU, float32...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={"": "cpu"},
        attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0,
        target_modules=plan["target_modules"], layers_to_transform=plan["layers"], bias="none"))
    trainable = [p for p in model.parameters() if p.requires_grad]
    report["trainable_parameters"] = sum(p.numel() for p in trainable)
    initial = {name: tensor.detach().clone() for name, tensor in get_peft_model_state_dict(model).items()}
    print(f"Trainable adapter parameters: {report['trainable_parameters']}", flush=True)

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in evaluation:
                value = float(model(**batch).loss)
                if not np.isfinite(value):
                    raise RuntimeError("Non-finite evaluation loss")
                losses.append(value)
        return losses

    def generate():
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": dataset["evaluation"][0]["question"]}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        model.eval()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=plan["generation_tokens"],
                                    do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("Baseline evaluation and short answer...", flush=True)
    report["evaluation_loss_before"] = evaluate()
    report["answer_before"] = generate()
    save(run / "report.json", report)
    optimizer = torch.optim.AdamW(trainable, lr=plan["learning_rate"], weight_decay=0)
    training_start = time.monotonic()
    stop_reason = None
    for epoch in range(plan["epochs"]):
        for index in np.random.default_rng(42 + epoch).permutation(len(training)):
            memory = psutil.virtual_memory()
            rss = process.memory_info().rss
            if (memory.available < plan["min_available_GiB"] * 2**30
                    or rss > plan["max_process_RSS_GiB"] * 2**30
                    or time.monotonic() - training_start > plan["max_training_seconds"]):
                stop_reason = "RESOURCE_LIMIT_BETWEEN_STEPS"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            loss = model(**training[int(index)]).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            row = {"step": len(report["steps"]) + 1, "epoch": epoch + 1,
                   "id": dataset["train"][int(index)]["id"], "loss": float(loss.detach()),
                   "grad_norm": float(grad_norm), "seconds": time.monotonic() - tick,
                   "rss_GiB": process.memory_info().rss / 2**30}
            report["steps"].append(row)
            save(run / "report.json", report)
            print(f"Step {row['step']}/12: loss={row['loss']:.4f}, {row['seconds']:.1f}s, RAM={row['rss_GiB']:.2f} GiB", flush=True)
        if stop_reason:
            break
    if not report["steps"]:
        report["status"] = stop_reason or "NO_TRAINING_STEPS"
        save(run / "report.json", report)
        raise SystemExit("No training step completed; no adapter saved.")
    print("Saving adapter...", flush=True)
    adapter = run / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    del optimizer
    changes = {name: float(torch.linalg.vector_norm(value.detach() - initial[name]))
               for name, value in get_peft_model_state_dict(model).items()}
    if not any(value > 0 for value in changes.values()):
        raise RuntimeError("Adapter weights did not change")
    report["adapter_weight_change_L2"] = changes
    # Check serialization by loading the saved adapter back into the same frozen base.
    model.load_adapter(str(adapter), adapter_name="saved_check", is_trainable=False)
    model.set_adapter("saved_check")
    reloaded = get_peft_model_state_dict(model, adapter_name="saved_check")
    for name, value in get_peft_model_state_dict(model, adapter_name="default").items():
        torch.testing.assert_close(value, reloaded[name], rtol=0, atol=0)
    print("Saved adapter reloaded; final evaluation...", flush=True)
    report["evaluation_loss_after"] = evaluate()
    report["answer_after"] = generate()
    report["adapter"] = str(adapter)
    report["adapter_sha256"] = sha(adapter / "adapter_model.safetensors")
    report["elapsed_seconds"] = time.monotonic() - start
    report["status"] = "TRAINED_CPU_PILOT" if stop_reason is None else "TRAINED_PARTIAL_CPU_PILOT"
    report["stop_reason"] = stop_reason
    report["note"] = "Pilot feasibility only. Two evaluation paraphrases cannot establish general TIMDR competence. No independent final holdout was used."
    save(run / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
