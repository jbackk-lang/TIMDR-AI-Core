"""Boundary adapter between B4-Kitchen artifacts and the learning sandbox.

The completed B4-Kitchen v0.3 result is an evaluation artifact.  It cannot be
silently reused as training data.  A future training bundle needs a distinct
preregistration, explicit authorization, and block-level feature rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from learning_sandbox import Sample


class B4KitchenLearningError(ValueError):
    pass


@dataclass(frozen=True)
class LearningEligibility:
    eligible: bool
    reason: str
    required_artifact: str


def assess_imported_b4_result(report: Mapping[str, Any]) -> LearningEligibility:
    """Reject an imported B4 result as a training set, even when its test passed."""
    if report.get("dataset_id") != "CMU_KITCHEN_S13_BROWNIE_v0.1":
        return LearningEligibility(False, "This adapter only recognizes the B4-Kitchen dataset.", "")
    return LearningEligibility(
        False,
        "B4-Kitchen v0.3 is a frozen evaluation artifact, not an authorized training split.",
        "timdr-b4-kitchen-learning/1 bundle with a new preregistration and disjoint train/calibration/holdout blocks",
    )


def load_authorized_training_bundle(path: Path) -> list[Sample]:
    """Load a separately preregistered feature bundle for future candidate learning."""
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise B4KitchenLearningError(f"Cannot read training bundle: {path}") from exc
    if bundle.get("schema") != "timdr-b4-kitchen-learning/1":
        raise B4KitchenLearningError("Unexpected training bundle schema.")
    if bundle.get("learning_authorized") is not True:
        raise B4KitchenLearningError("Training requires explicit authorization in a new preregistration.")
    if bundle.get("source_evaluation_prereg") == "B4_KITCHEN_v0.3":
        raise B4KitchenLearningError("The frozen B4-Kitchen v0.3 evaluation preregistration cannot authorize training.")
    samples: list[Sample] = []
    seen_blocks: set[int] = set()
    for row in bundle.get("samples", []):
        block_id = int(row["block_id"])
        if block_id in seen_blocks:
            raise B4KitchenLearningError(f"Repeated block_id {block_id} across learning splits.")
        seen_blocks.add(block_id)
        samples.append(Sample(tuple(float(v) for v in row["features"]), int(row["label"]), row["split"]))
    if not samples:
        raise B4KitchenLearningError("Training bundle contains no block-level features.")
    return samples
