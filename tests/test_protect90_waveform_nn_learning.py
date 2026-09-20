from __future__ import annotations

import json

import pytest

from protect90_learning_adapter import Protect90Error
from protect90_waveform_nn_learning import EPOCHS, HIDDEN_UNITS, LEARNING_RATE, SEED, run_train_calibration


def test_rejects_wrong_preregistration_version_before_touching_any_data(tmp_path):
    """Mirrors protect90_waveform_learning's own guard: a wrong prereg version
    must be rejected before the (nonexistent, here) source_root is ever read."""
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps({"version": "SOME_OTHER_VERSION"}), encoding="utf-8")

    with pytest.raises(Protect90Error, match="frozen waveform preregistration"):
        run_train_calibration(tmp_path / "does-not-exist", prereg_path)


def test_hyperparameters_are_frozen_module_constants():
    """The point of freezing hyperparameters as named constants (not inline
    literals) is that any change to them is a visible diff, not a silent
    edit buried in a function body."""
    assert HIDDEN_UNITS > 0
    assert EPOCHS > 0
    assert LEARNING_RATE > 0
    assert isinstance(SEED, int)
