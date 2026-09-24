"""Run bounded online retrieval followed by skill-guided candidate routing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_learning import OnlineLearningError
from environment_policy import EnvironmentPolicyError
from skill_guided_learning import run_skill_guided_cycle

if len(sys.argv) != 2:
    raise SystemExit("Usage: python examples/skill_guided_learn.py online_catalogs.json")

config_path = Path(sys.argv[1])
if not config_path.is_file():
    raise SystemExit("Missing configuration. Copy online_catalogs.template.json to online_catalogs.json and declare HTTPS catalogs.")

try:
    report = run_skill_guided_cycle(config_path, ROOT / "external_cache")
except (OnlineLearningError, EnvironmentPolicyError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Skill-guided cycle rejected: {exc}") from exc

print(f"Skill-guided online cycle complete: {report['new_documents']} new documents.")
print(f"Unavailable items skipped: {report['rejected_documents']}")
print(f"Candidate reading cards: {report['candidate_cards']} | holdout accessed: False")
print(f"Local report: {report['report_path']}")
