from __future__ import annotations

from protect90_learning_adapter import FEATURE_COLUMNS, _split_ids


def test_stratified_split_is_disjoint_and_complete():
    ids = list(range(16))
    rows = {sample_id: {"sc_type": str(sample_id % 4)} for sample_id in ids}
    splits = _split_ids(ids, rows)
    combined = [sample_id for values in splits.values() for sample_id in values]
    assert sorted(combined) == ids
    assert len(combined) == len(set(combined))
    assert FEATURE_COLUMNS
