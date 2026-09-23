"""Read only authorized Paderborn train/calibration vibration windows.

The holdout is deliberately rejected before a RAR member is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
from typing import Literal


Split = Literal["train", "calibration", "holdout"]
SAMPLE_RATE_HZ = 64000
WINDOW_SAMPLES = SAMPLE_RATE_HZ


class PaderbornExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class VibrationWindow:
    archive: str
    member: str
    split: Split
    window_index: int
    samples: object


def _dependencies():
    try:
        import numpy as np
        from scipy.io import loadmat
    except ImportError as exc:
        raise PaderbornExtractionError("Install Paderborn analysis dependencies: pip install -e '.[paderborn-analysis]'") from exc
    if not shutil.which("tar"):
        raise PaderbornExtractionError("Windows bsdtar is required to read RAR members.")
    return np, loadmat


def _members(root: Path, split: Split) -> list[dict[str, str]]:
    prereg = json.loads((root / "prereg" / "PADERBORN_MS_REPLICATION_v0.1.json").read_text(encoding="utf-8"))
    if prereg.get("state") != "FROZEN_BEFORE_WAVEFORM_EXTRACTION":
        raise PaderbornExtractionError("Paderborn dataset selection is not frozen.")
    if split == "holdout":
        raise PaderbornExtractionError("Holdout payload access is not authorized by this extractor.")
    return prereg["split"]["members"][split]


def _vibration_from_member(root: Path, archive: str, member: str):
    np, loadmat = _dependencies()
    source = root / "data" / "paderborn_candidate" / "raw" / archive
    try:
        raw = subprocess.check_output(["tar", "-xOf", str(source), member], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise PaderbornExtractionError(f"Cannot read {archive}:{member}") from exc
    payload = loadmat(BytesIO(raw), squeeze_me=True, struct_as_record=False)
    variable_names = [name for name in payload if not name.startswith("__")]
    if len(variable_names) != 1:
        raise PaderbornExtractionError(f"Unexpected MATLAB top-level schema in {member}.")
    channels = payload[variable_names[0]].Y
    vibration = next((channel.Data for channel in channels if str(channel.Name) == "vibration_1"), None)
    if vibration is None:
        raise PaderbornExtractionError(f"Channel vibration_1 is absent from {member}.")
    values = np.asarray(vibration, dtype=float).reshape(-1)
    expected = 4 * WINDOW_SAMPLES
    if len(values) not in (expected, expected + 1):
        raise PaderbornExtractionError(f"Unexpected vibration length {len(values)} in {member}.")
    return values[:expected]


def authorized_windows(root: str | Path, split: Literal["train", "calibration"]) -> list[VibrationWindow]:
    """Return four one-second windows per frozen member; holdout is unrepresentable."""
    dataset_root = Path(root)
    answer: list[VibrationWindow] = []
    for item in _members(dataset_root, split):
        signal = _vibration_from_member(dataset_root, item["archive"], item["member"])
        for index in range(4):
            start = index * WINDOW_SAMPLES
            answer.append(VibrationWindow(item["archive"], item["member"], split, index, signal[start:start + WINDOW_SAMPLES]))
    return answer


def training_schema(root: str | Path) -> dict[str, int | str]:
    """Inspect exactly one authorized train member and return metadata, not statistics."""
    dataset_root = Path(root)
    first = _members(dataset_root, "train")[0]
    signal = _vibration_from_member(dataset_root, first["archive"], first["member"])
    return {"archive": first["archive"], "member": first["member"], "channel": "vibration_1", "samples_after_trim": len(signal), "windows": 4}
