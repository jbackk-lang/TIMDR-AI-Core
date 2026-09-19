from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from b4_kitchen_learning_adapter import assess_imported_b4_result  # noqa: E402

report = json.loads((ROOT / "evidence" / "B4_KITCHEN_v0.3_import.json").read_text(encoding="utf-8"))
eligibility = assess_imported_b4_result(report)
print("Eligible for learning:", eligibility.eligible)
print("Reason:", eligibility.reason)
print("Required next artifact:", eligibility.required_artifact)
