from timdr_ai_core import (
    ControlResult,
    FundamentalModelLTR,
    Hypothesis,
    LayerIInformation,
    LayerTTopology,
    ProtocolCriteria,
    TestEvidence,
    TIMDRProtocol,
    TIMDR_AI_System,
)


def test_preregistration_fingerprint_is_stable_and_parameters_are_frozen():
    protocol = TIMDRProtocol()
    hypothesis = Hypothesis("h", "d", "e", {"b": 2, "a": [1, 2]})
    preregistration = protocol.preregister(hypothesis)
    reordered = protocol.preregister(Hypothesis("h", "d", "e", {"a": [1, 2], "b": 2}))

    hypothesis.params["a"].append(3)

    assert preregistration.fingerprint == reordered.fingerprint
    assert preregistration.frozen_params == {"a": [1, 2], "b": 2}


def test_missing_controls_or_evidence_can_never_support_a_claim():
    system = TIMDR_AI_System()
    result = system.run("raw", {"name": "h", "params": {}})

    assert result["controls"].passed is False
    assert result["test_result"].verdict == "INCONCLUSIVE"


def test_failed_negative_control_blocks_even_statistically_positive_evidence():
    protocol = TIMDRProtocol()
    result = protocol.run_test(
        ControlResult(positive_ok=True, negative_ok=False),
        TestEvidence(0.0001, 0.9, "Mann-Whitney U"),
    )

    assert result.verdict == "INCONCLUSIVE"


def test_supported_requires_passed_controls_significance_and_effect_size():
    protocol = TIMDRProtocol(ProtocolCriteria(alpha=0.05, min_abs_effect_size=0.30))
    controls = ControlResult(positive_ok=True, negative_ok=True)

    weak = protocol.run_test(controls, TestEvidence(0.01, 0.20, "permutation"))
    strong = protocol.run_test(controls, TestEvidence(0.01, 0.31, "permutation"))

    assert weak.verdict == "NOT_SUPPORTED"
    assert strong.verdict == "SUPPORTED"


def test_ltr_pipeline_is_composable_and_does_not_set_the_verdict():
    model = FundamentalModelLTR(
        topology=LayerTTopology(lambda value: value + 2),
        information=LayerIInformation(lambda value: value * 3),
    )
    system = TIMDR_AI_System(model=model)
    result = system.run(4, {"name": "h", "params": {}})

    assert result["model_output"] == 18
    assert result["test_result"].verdict == "INCONCLUSIVE"
