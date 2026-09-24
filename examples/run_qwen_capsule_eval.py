"""Frozen CPU evaluation: unchanged Qwen versus the same Qwen with Phase 1C evidence."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"
PLAN, CAPSULE = ROOT / "data" / "qwen_capsule_eval_v0.1.json", ROOT / "data" / "timdr_knowledge_injection_v0.2.json"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt(question, evidence):
    instruction = ("Odpowiadasz w zamrożonym teście wierności dowodowej. Odpowiedz krótko po polsku. "
                   "Nie dopowiadaj faktów niewynikających z dostarczonych rekordów. "
                   "Jeśli nie masz rekordu, odpowiedz dokładnie: INCONCLUSIVE_NO_MATCH. "
                   "Jeśli masz rekord, zakończ odpowiedź linią Evidence: [id1, id2].")
    if not evidence:
        return f"{instruction}\n\nPytanie: {question}"
    records = "\n\n".join(f"ID: {r['id']}\nClaim: {r['claim']}\nStatus: {r['status']}\nScope: {r['scope']}\nLimitation: {r['limitation']}" for r in evidence)
    return f"{instruction}\n\nDostarczone rekordy:\n{records}\n\nPytanie: {question}"


def generate(model, tokenizer, text, max_new_tokens):
    import torch
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--verify-only", action="store_true"); parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from examples.phase1c_knowledge_injection import trajectory
    plan, capsule = json.loads(PLAN.read_text(encoding="utf-8")), json.loads(CAPSULE.read_text(encoding="utf-8"))
    model_manifest = BASE / "DOWNLOAD_MANIFEST.json"
    preflight = {"plan_sha256": sha(PLAN), "capsule_sha256": sha(CAPSULE), "expected_capsule_sha256": plan["capsule"]["sha256"], "model_manifest_sha256": sha(model_manifest) if model_manifest.is_file() else None, "model_present": BASE.is_dir()}
    if not preflight["model_present"] or preflight["capsule_sha256"] != preflight["expected_capsule_sha256"]:
        raise RuntimeError("Frozen model or capsule preflight failed")
    if args.verify_only:
        print(json.dumps({"status": "PREFLIGHT_OK", **preflight}, ensure_ascii=False, indent=2)); return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(plan["generation"]["cpu_threads"]); torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.eval(); rows, review_queue = [], []
    for item in plan["items"]:
        route = trajectory(item["question"], capsule)
        if route["status"] not in {"EVIDENCE_READY", "INCONCLUSIVE_NO_MATCH"} or route["selected_record_ids"] != item["expected_record_ids"]:
            raise RuntimeError(f"Frozen evidence route mismatch for {item['id']}: {route}")
        base_answer = generate(model, tokenizer, prompt(item["question"], []), plan["generation"]["max_new_tokens"])
        capsule_answer = generate(model, tokenizer, prompt(item["question"], route["evidence"]), plan["generation"]["max_new_tokens"])
        expected = item["expected_record_ids"]
        format_ok = all(record_id in capsule_answer for record_id in expected) if expected else "INCONCLUSIVE_NO_MATCH" in capsule_answer
        rows.append({"id": item["id"], "question": item["question"], "route_status": route["status"], "expected_record_ids": expected, "base_answer": base_answer, "capsule_answer": capsule_answer, "capsule_format_check": format_ok})
        for variant, answer in (("A", base_answer), ("B", capsule_answer)):
            review_queue.append({"id": item["id"], "variant_blind": variant, "answer": answer, "criteria": item["review_criteria"]})
    technical = {"capsule_format_passes": sum(bool(row["capsule_format_check"]) for row in rows), "total": len(rows), "note": "Automatic checks assess trace/format only. Semantic correctness requires the blinded review queue."}
    report = {"status": "RUN_COMPLETE_PENDING_HUMAN_REVIEW", "finished_at_utc": datetime.now(timezone.utc).isoformat(), "preflight": preflight, "rows": rows, "technical": technical, "review_queue": review_queue, "verdict": "INCONCLUSIVE_PENDING_HUMAN_REVIEW"}
    output = args.output or ROOT / "learning_runs" / f"qwen_capsule_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "format": technical}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
