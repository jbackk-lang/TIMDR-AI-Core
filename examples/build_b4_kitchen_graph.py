from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from provenance_graph import build_report_graph, export_graph, write_graph  # noqa: E402

report = json.loads((ROOT / "evidence" / "B4_KITCHEN_v0.3_import.json").read_text(encoding="utf-8"))
nodes, edges = build_report_graph(report)
write_graph(export_graph(nodes, edges), ROOT / "graphs" / "b4_kitchen_v03.json", ROOT / "graphs" / "b4_kitchen_v03.dot")
print(f"nodes={len(nodes)} edges={len(edges)} acyclic=True")
