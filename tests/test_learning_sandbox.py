import pytest

from learning_sandbox import LearningError, LearningSandbox, Sample


def test_learning_requires_an_untouched_holdout_and_never_promotes_itself():
    samples = [Sample((0.0,), 0, "train"), Sample((1.0,), 1, "train"), Sample((0.1,), 0, "calibration"), Sample((0.9,), 1, "calibration"), Sample((0.2,), 0, "holdout")]
    run = LearningSandbox().fit(samples)
    proposal = LearningSandbox().propose(run)
    assert proposal.requires_human_preregistration
    assert proposal.proposed_parameters["holdout_accessed"] is False


def test_learning_rejects_missing_holdout():
    samples = [Sample((0.0,), 0, "train"), Sample((1.0,), 1, "train"), Sample((0.1,), 0, "calibration")]
    with pytest.raises(LearningError, match="holdout"):
        LearningSandbox().fit(samples)
