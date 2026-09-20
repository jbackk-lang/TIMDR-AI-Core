"""Run the same train/calibration stage as train_protect90_waveform.py,
but with the small numpy-only MLP instead of the nearest-centroid baseline.

Same frozen hypothesis, same split, same holdout wall."""
from __future__ import annotations
import sys
from dataclasses import asdict
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from protect90_waveform_nn_learning import run_train_calibration
from protect90_waveform_learning import WaveformDependencyError
from protect90_learning_adapter import Protect90Error
PREREG = ROOT / "prereg" / "PROTECT90_WAVEFORM_MULTICLASS_v0.3.json"
if len(sys.argv) != 2:
    raise SystemExit('Usage: python examples/train_protect90_waveform_nn.py "C:\\...\\TIMDR-Grid-Monitor"')
try:
    run = run_train_calibration(sys.argv[1], PREREG)
except (Protect90Error, WaveformDependencyError) as exc:
    raise SystemExit(str(exc)) from exc
report = {"kind": "protect90_waveform_nn_train_calibration", "holdout_accessed": False, "result": asdict(run)}
target = ROOT / "learning_runs" / "protect90_waveform_nn_latest.json"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Waveform NN train/calibration complete. Accuracy: {run.calibration_accuracy:.4f}")
print("Holdout accessed: False")
