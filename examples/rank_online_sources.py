"""Create a local ranking report from the online-learning state."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from online_ranker import rank_state

state = ROOT / "external_cache" / "online_learning_state.json"
if not state.is_file():
    raise SystemExit("No online-learning state yet. Run --online-learn first.")
report = rank_state(state)
target = ROOT / "external_cache" / "online_source_ranking.json"
target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Ranked {len(report['documents'])} online documents.")
for item in report["documents"][:5]:
    print(f"{item['candidate_score']:>2} | {item['title'] or item['url']}")
print(f"Local report: {target}")
