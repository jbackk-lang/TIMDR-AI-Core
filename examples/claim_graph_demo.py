import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from claim_graph_gate import decide, render

parser = argparse.ArgumentParser()
parser.add_argument("question")
args = parser.parse_args()
print(render(decide(args.question)))
