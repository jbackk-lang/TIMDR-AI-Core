from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learning_sandbox import LearningSandbox, Sample  # noqa: E402


samples = [
    Sample((0.0, 0.1), 0, "train"), Sample((0.2, 0.0), 0, "train"),
    Sample((1.0, 0.9), 1, "train"), Sample((0.9, 1.1), 1, "train"),
    Sample((0.1, 0.2), 0, "calibration"), Sample((1.1, 1.0), 1, "calibration"),
    Sample((0.15, 0.05), 0, "holdout"), Sample((1.05, 0.95), 1, "holdout"),
]
run = LearningSandbox().fit(samples)
candidate = LearningSandbox().propose(run)
print("Calibration accuracy:", run.calibration_accuracy)
print("Holdout accessed:", candidate.proposed_parameters["holdout_accessed"])
print("Requires human preregistration:", candidate.requires_human_preregistration)
