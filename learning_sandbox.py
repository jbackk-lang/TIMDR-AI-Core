"""Offline candidate learning, isolated from TIMDR verdicts and preregistration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import sqrt
from typing import Any, Literal, Sequence


Split = Literal["train", "calibration", "holdout"]


class LearningError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    features: tuple[float, ...]
    label: int
    split: Split


@dataclass(frozen=True)
class LearningRun:
    dataset_fingerprint: str
    train_count: int
    calibration_count: int
    holdout_count: int
    feature_count: int
    labels: tuple[int, ...]
    centroids: dict[int, tuple[float, ...]]
    calibration_accuracy: float


@dataclass(frozen=True)
class CandidateProposal:
    kind: str
    learning_run_fingerprint: str
    proposed_parameters: dict[str, Any]
    requires_human_preregistration: bool = True
    may_not_change_existing_preregistration: bool = True


def _fingerprint(samples: Sequence[Sample]) -> str:
    payload = [asdict(sample) for sample in samples]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _centroid(samples: Sequence[Sample], label: int, n_features: int) -> tuple[float, ...]:
    chosen = [sample.features for sample in samples if sample.label == label]
    if not chosen:
        raise LearningError(f"Training split has no samples for label {label}.")
    return tuple(sum(row[i] for row in chosen) / len(chosen) for i in range(n_features))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


class LearningSandbox:
    """A transparent nearest-centroid baseline for proposing, never deciding, a hypothesis."""

    def fit(self, samples: Sequence[Sample]) -> LearningRun:
        if not samples:
            raise LearningError("No samples supplied.")
        n_features = len(samples[0].features)
        if n_features == 0 or any(len(sample.features) != n_features for sample in samples):
            raise LearningError("All samples need the same non-empty feature vector.")
        if any(not isinstance(sample.label, int) for sample in samples):
            raise LearningError("Labels must be integer class identifiers.")
        train = [sample for sample in samples if sample.split == "train"]
        calibration = [sample for sample in samples if sample.split == "calibration"]
        holdout = [sample for sample in samples if sample.split == "holdout"]
        if not train or not calibration or not holdout:
            raise LearningError("train, calibration, and untouched holdout splits are all required.")
        labels = tuple(sorted({sample.label for sample in train}))
        if len(labels) < 2:
            raise LearningError("Training split needs samples from at least two classes.")
        centroids = {label: _centroid(train, label, n_features) for label in labels}
        correct = sum(
            int(self.predict_features(sample.features, centroids) == sample.label)
            for sample in calibration
        )
        return LearningRun(
            _fingerprint(samples), len(train), len(calibration), len(holdout), n_features,
            labels, centroids, correct / len(calibration),
        )

    @staticmethod
    def predict_features(features: Sequence[float], centroids: dict[int, tuple[float, ...]]) -> int:
        """Deterministic tie-breaking: the lower numeric class identifier wins."""
        return min(centroids, key=lambda label: (_distance(features, centroids[label]), label))

    def propose(self, run: LearningRun) -> CandidateProposal:
        """Return a candidate configuration, deliberately not a TIMDR verdict."""
        run_hash = sha256(json.dumps(asdict(run), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return CandidateProposal(
            kind="nearest_centroid_multiclass_candidate",
            learning_run_fingerprint=run_hash,
            proposed_parameters={
                "classes": run.labels,
                "centroids": run.centroids,
                "calibration_accuracy": run.calibration_accuracy,
                "holdout_accessed": False,
            },
        )
