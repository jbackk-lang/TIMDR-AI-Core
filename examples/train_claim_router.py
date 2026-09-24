"""Train and calibrate Claim Router without loading its frozen holdout."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from claim_router_learning import DATA, ROOT, guarded_route, save_model, train


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


dataset = json.loads(DATA.read_text(encoding="utf-8"))
model = train(dataset["train"])
rows = []
for question, expected in dataset["calibration"]:
    result = guarded_route(model, question)
    rows.append({"question": question, "expected": expected, **result, "learned_correct": result["learned_label"] == expected})
report = {"status": "TRAIN_CALIBRATION_COMPLETE_HOLDOUT_UNREAD", "dataset_sha256": sha(DATA), "calibration": rows,
          "learned_accuracy": sum(row["learned_correct"] for row in rows) / len(rows),
          "logic_agreement": sum(row["agreement"] for row in rows) / len(rows),
          "holdout_accessed": False}
out_dir = ROOT / "learning_runs" / f"claim_router_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"; out_dir.mkdir(parents=True, exist_ok=True)
save_model(model, out_dir / "model.json"); (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": report["status"], "learned_accuracy": report["learned_accuracy"], "logic_agreement": report["logic_agreement"], "holdout_accessed": False, "report": str(out_dir / "report.json")}, ensure_ascii=False, indent=2))
