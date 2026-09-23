from __future__ import annotations

import json
from pathlib import Path

import pytest

from paderborn_extractor import PaderbornExtractionError, _members


def test_holdout_is_rejected_before_any_archive_access(tmp_path: Path):
    prereg = tmp_path / "prereg"
    prereg.mkdir()
    (prereg / "PADERBORN_MS_REPLICATION_v0.1.json").write_text(json.dumps({
        "state": "FROZEN_BEFORE_WAVEFORM_EXTRACTION",
        "split": {"members": {"train": [], "calibration": [], "holdout": [{"archive": "never-open.rar", "member": "never-open.mat"}]}},
    }), encoding="utf-8")
    with pytest.raises(PaderbornExtractionError, match="not authorized"):
        _members(tmp_path, "holdout")
