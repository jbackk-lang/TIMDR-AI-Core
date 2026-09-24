"""Run one automatic, bounded online-learning cycle from configured catalogs."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_learning import OnlineLearningError, run_cycle
from environment_policy import EnvironmentPolicyError

if len(sys.argv) != 2:
    raise SystemExit("Usage: python examples/online_learn.py online_catalogs.json")
config_path = Path(sys.argv[1])
if not config_path.is_file():
    raise SystemExit(
        f"Missing configuration: {config_path}. Create it with:\n"
        "  Copy-Item .\\online_catalogs.template.json .\\online_catalogs.json\n"
        "Then replace the example catalog URL with a real HTTPS JSON catalog."
    )
try:
    report = run_cycle(config_path, ROOT / "external_cache" / "online_learning_state.json")
except (OnlineLearningError, EnvironmentPolicyError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Online cycle rejected: {exc}") from exc
print(f"Online learning cycle complete: {report['new_documents']} new documents.")
print(f"Refreshed source descriptions: {report['refreshed_documents']}")
print(f"Unavailable items skipped: {report['rejected_documents']}")
print(f"Known documents: {report['total_documents']} | holdout accessed: False")
