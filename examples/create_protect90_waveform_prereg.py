"""Freeze the waveform-specific hypothesis before feature extraction or fitting."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from protect90_learning_adapter import Protect90Error, write_waveform_preregistration
TARGET = ROOT / "prereg" / "PROTECT90_WAVEFORM_MULTICLASS_v0.3.json"
if len(sys.argv) != 2:
    raise SystemExit('Usage: python examples/create_protect90_waveform_prereg.py "C:\\...\\TIMDR-Grid-Monitor"')
try:
    plan = write_waveform_preregistration(sys.argv[1], TARGET)
except Protect90Error as exc:
    raise SystemExit(f"Cannot freeze waveform preregistration: {exc}") from exc
print(f"Frozen waveform preregistration: {TARGET}")
print(f"Features: {plan['task']['features']['expected_feature_count']} | no additional waveform was opened by this command.")
print("A prior train-only schema inspection is declared in the preregistration.")
