"""Create a local candidate queue from the online ranking."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research_queue import build_queue

cache = ROOT / "external_cache"
ranking, state = cache / "online_source_ranking.json", cache / "online_learning_state.json"
if not ranking.is_file() or not state.is_file():
    raise SystemExit("Missing online ranking/state. Run --online-learn and --online-rank first.")
queue = build_queue(ranking, state)
target = cache / "research_queue.json"
target.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Research queue: {len(queue['entries'])} candidate sources.")
for entry in queue["entries"]:
    print(f"{entry['branch']:<15} score={entry['branch_score']} | {entry['title'] or entry['url']}")
print(f"Local queue: {target}")
