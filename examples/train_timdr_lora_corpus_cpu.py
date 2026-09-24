"""Bounded CPU LoRA continued pretraining on a frozen local TIMDR corpus."""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
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
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "timdr_source_lora_v0.1.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means one full frozen training split")
    parser.add_argument("--max-training-seconds", type=int, default=14400)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 2 or args.max_steps < 0 or not 60 <= args.max_training_seconds <= 14400:
        raise ValueError("epochs=1..2, max-steps>=0, max-training-seconds=60..14400")
    import numpy as np
    import psutil
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(51)
    np.random.seed(51)
    process = psutil.Process()
    if os.name == "nt":
        try:
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except psutil.AccessDenied:
            pass
    corpus = json.loads(args.data.read_text(encoding="utf-8"))
    if corpus.get("schema") != "timdr-source-lora/0.1":
        raise ValueError("Unsupported corpus schema")
    run = ROOT / "learning_runs" / "timdr_lora_corpus_cpu" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=False)
    plan = {"status": "FROZEN_SOURCE_LORA_RUN", "seed": 51, "epochs": args.epochs,
            "max_steps": args.max_steps, "max_training_seconds": args.max_training_seconds,
            "batch_size": 1, "max_sequence_tokens": corpus["max_tokens"], "learning_rate": 0.0001,
            "lora_r": 8, "lora_alpha": 16, "layers": [24, 25, 26, 27],
            "target_modules": ["q_proj", "v_proj"], "threads": 2, "dtype": "float32",
            "min_available_GiB": 0.75, "max_process_RSS_GiB": 8.0,
            "corpus_sha256": sha(args.data), "corpus_source_sha256": corpus["source_sha256"],
            "train_chunk_ids": [row["id"] for row in corpus["train"]],
            "evaluation_chunk_ids": [row["id"] for row in corpus["evaluation"]],
            "base_manifest": json.loads((BASE / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8"))}
    save(run / "plan.json", plan)
    report = {"status": "STARTED", "run": str(run), "steps": []}
    save(run / "report.json", report)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prepare(row):
        ids = tokenizer(row["text"], return_tensors="pt", add_special_tokens=True, truncation=True,
                        max_length=plan["max_sequence_tokens"])["input_ids"]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    training = [prepare(row) for row in corpus["train"]]
    evaluation = [prepare(row) for row in corpus["evaluation"]]
    available = psutil.virtual_memory().available / 2**30
    print(f"Run: {run}\nTrain chunks: {len(training)} | eval chunks: {len(evaluation)} | RAM: {available:.2f} GiB", flush=True)
    if available < 6.5:
        report["status"] = "STOPPED_INSUFFICIENT_FREE_RAM_BEFORE_LOAD"
        save(run / "report.json", report)
        raise SystemExit("Need at least 6.5 GiB free RAM before loading Qwen.")
    model = AutoModelForCausalLM.from_pretrained(
        str(BASE), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, device_map={"": "cpu"}, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                           lora_dropout=0, target_modules=plan["target_modules"],
                           layers_to_transform=plan["layers"], bias="none"))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    report["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
    initial = {name: value.detach().clone() for name, value in get_peft_model_state_dict(model).items()}
    optimizer = torch.optim.AdamW(trainable, lr=plan["learning_rate"], weight_decay=0)

    @torch.no_grad()
    def evaluate():
        model.eval()
        values = [float(model(**batch).loss) for batch in evaluation]
        return {"mean_loss": float(np.mean(values)), "chunks": len(values)}

    report["evaluation_before"] = evaluate()
    save(run / "report.json", report)
    started, stop_reason = time.monotonic(), None
    step = 0
    for epoch in range(args.epochs):
        for index in np.random.default_rng(51 + epoch).permutation(len(training)):
            if args.max_steps and step >= args.max_steps:
                stop_reason = "MAX_STEPS"
                break
            memory = psutil.virtual_memory()
            if memory.available < plan["min_available_GiB"] * 2**30 or process.memory_info().rss > plan["max_process_RSS_GiB"] * 2**30:
                stop_reason = "RESOURCE_LIMIT_BETWEEN_STEPS"
                break
            if time.monotonic() - started > args.max_training_seconds:
                stop_reason = "TIME_LIMIT_BETWEEN_STEPS"
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            loss = model(**training[int(index)]).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {"step": step, "epoch": epoch + 1, "id": corpus["train"][int(index)]["id"],
                   "loss": float(loss.detach()), "grad_norm": float(gradient),
                   "seconds": time.monotonic() - tick, "rss_GiB": process.memory_info().rss / 2**30}
            report["steps"].append(row)
            if step == 1 or step % 10 == 0:
                save(run / "report.json", report)
                print(f"Step {step}: loss={row['loss']:.4f}, {row['seconds']:.1f}s, RAM={row['rss_GiB']:.2f} GiB", flush=True)
        if stop_reason:
            break
    if not report["steps"]:
        report["status"] = stop_reason or "NO_TRAINING_STEPS"
        save(run / "report.json", report)
        raise SystemExit("No adapter step completed.")
    adapter = run / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    changes = {name: float(torch.linalg.vector_norm(value.detach() - initial[name]))
               for name, value in get_peft_model_state_dict(model).items()}
    if not any(value > 0 for value in changes.values()):
        raise RuntimeError("Adapter weights did not change")
    report.update({"status": "TRAINED_SOURCE_LORA" if stop_reason is None else "TRAINED_PARTIAL_SOURCE_LORA",
                   "stop_reason": stop_reason, "adapter": str(adapter), "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
                   "adapter_weight_change_L2": changes, "evaluation_after": evaluate(),
                   "elapsed_seconds": time.monotonic() - started,
                   "note": "Source-document adaptation only; this does not establish factual reliability or empirical TIMDR validity."})
    save(run / "report.json", report)
    print(json.dumps({key: report[key] for key in ("status", "stop_reason", "trainable_parameters", "evaluation_before", "evaluation_after", "elapsed_seconds", "adapter")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
