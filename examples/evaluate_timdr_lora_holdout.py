"""Compare frozen Qwen 1.5B baseline and LoRA adapter on held-out TIMDR prompts."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"
HOLDOUT = ROOT / "data" / "timdr_text_holdout_v0.1.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def latest_adapter() -> Path:
    runs = ROOT / "learning_runs" / "timdr_lora_cpu"
    candidates = sorted(runs.glob("*/adapter/adapter_config.json"))
    if not candidates:
        raise FileNotFoundError("No LoRA adapter found in learning_runs/timdr_lora_cpu")
    return candidates[-1].parent


def loss_for(model, tokenizer, item):
    import torch

    messages = [{"role": "user", "content": item["question"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": item["answer"]}], tokenize=True
    )
    ids = torch.tensor([full])
    labels = ids.clone()
    labels[:, : len(prefix)] = -100
    with torch.no_grad():
        return float(model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss)


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--adapter", type=Path, default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    if data["schema"] != "timdr-text-holdout/1" or not data["rules"]["optimization_forbidden"]:
        raise ValueError("Holdout is not frozen for evaluation")
    items = data["items"][: args.limit]
    adapter = args.adapter or latest_adapter()
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE), local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    base.config.use_cache = False
    before = [{"id": item["id"], "loss": loss_for(base, tokenizer, item)} for item in items]
    adapted = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    after = [{"id": item["id"], "loss": loss_for(adapted, tokenizer, item)} for item in items]
    by_id = {row["id"]: row["loss"] for row in before}
    report = {
        "status": "DESCRIPTIVE_FROZEN_HOLDOUT_COMPARISON",
        "holdout_sha256": sha256(HOLDOUT),
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "items": len(items),
        "baseline": before,
        "adapter_result": after,
        "mean_loss_before": sum(by_id[row["id"]] for row in after) / len(after),
        "mean_loss_after": sum(row["loss"] for row in after) / len(after),
        "note": "Comparison measures fit to held-out reference answers only; it is not evidence of general TIMDR competence. The frozen holdout must not enter training.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "learning_runs" / "timdr_lora_cpu" / f"holdout_eval_{stamp}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(destination), "items": len(items), "mean_loss_before": report["mean_loss_before"], "mean_loss_after": report["mean_loss_after"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
