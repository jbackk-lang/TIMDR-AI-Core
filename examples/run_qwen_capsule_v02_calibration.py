"""Run only frozen synthetic prompt controls before any new Qwen+Capsule holdout."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2"})
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
PLAN = ROOT / "data" / "qwen_capsule_v0.2_calibration.json"
BASE = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    torch.set_num_threads(plan["generation"]["cpu_threads"]); torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model.eval(); rows=[]
    for control in plan["controls"]:
        inputs=tokenizer.apply_chat_template([{"role":"user","content":control["prompt"]}],tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True)
        with torch.no_grad(): output=model.generate(**inputs,do_sample=False,max_new_tokens=plan["generation"]["max_new_tokens"],use_cache=True,pad_token_id=tokenizer.pad_token_id)
        answer=tokenizer.decode(output[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        rows.append({"id":control["id"],"kind":control["kind"],"answer":answer,"passed":all(term.lower() in answer.lower() for term in control["required_terms"])})
    passed=all(row["passed"] for row in rows)
    report={"status":"CALIBRATION_PASS" if passed else "CALIBRATION_FAIL_HOLDOUT_BLOCKED","finished_at_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":sha(PLAN),"rows":rows,"next":"A new v0.2 holdout may be preregistered only after PASS." if passed else "Do not open a new holdout; document this failure or create a later calibration version."}
    out=ROOT / "learning_runs" / f"qwen_capsule_v02_calibration_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}" / "report.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"output":str(out),"rows":rows},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
