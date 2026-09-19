from b4_kitchen_learning_adapter import assess_imported_b4_result


def test_b4_evaluation_result_is_not_training_data():
    status = assess_imported_b4_result({"dataset_id": "CMU_KITCHEN_S13_BROWNIE_v0.1", "declared_verdict": "SUPPORTED"})
    assert status.eligible is False
    assert "frozen evaluation" in status.reason
