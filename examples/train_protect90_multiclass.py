"""Run only the authorized train/calibration stage of the frozen PROTECT-90 plan."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_sandbox import LearningSandbox
from protect90_learning_adapter import Protect90Error, samples_from_preregistration


PREREG = ROOT / "prereg" / "PROTECT90_MULTICLASS_v0.1.json"

if len(sys.argv) != 2:
    raise SystemExit('Usage: python examples/train_protect90_multiclass.py "C:\\...\\TIMDR-Grid-Monitor"')
if not PREREG.is_file():
    raise SystemExit("Missing frozen preregistration. Run create_protect90_prereg.py first.")

try:
    samples = samples_from_preregistration(sys.argv[1], PREREG)
    sandbox = LearningSandbox()
    run = sandbox.fit(samples)
    proposal = sandbox.propose(run)
except Protect90Error as exc:
    raise SystemExit(f"Cannot run frozen plan: {exc}") from exc

report = {
    "kind": "protect90_multiclass_training_stage",
    "scope": "Research baseline only; not a grid protection device or TIMDR B4 result.",
    "holdout_accessed": False,
    "learning_run": asdict(run),
    "candidate_proposal": asdict(proposal),
}
output = ROOT / "learning_runs" / "protect90_multiclass_latest.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("Training/calibration stage complete.")
print(f"Classes: {run.labels} | calibration accuracy: {run.calibration_accuracy:.4f}")
print("Holdout accessed: False")
print(f"Report (ignored by Git): {output}")
