"""Create the immutable PROTECT-90 multiclass preregistration before first fit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protect90_learning_adapter import Protect90Error, write_preregistration


TARGET = ROOT / "prereg" / "PROTECT90_MULTICLASS_v0.1.json"

if len(sys.argv) != 2:
    raise SystemExit('Usage: python examples/create_protect90_prereg.py "C:\\...\\TIMDR-Grid-Monitor"')

try:
    plan = write_preregistration(sys.argv[1], TARGET)
except Protect90Error as exc:
    raise SystemExit(f"Cannot freeze preregistration: {exc}") from exc

print(f"Frozen preregistration: {TARGET}")
print(f"Episodes: {plan['dataset']['n_selected_episodes']} | splits: {plan['split']['counts']}")
print("No fitting and no holdout access occurred.")
