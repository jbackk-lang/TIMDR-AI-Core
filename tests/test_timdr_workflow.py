import pytest
from timdr_ai_core import (TIMDRProtocol, Hypothesis, ControlResult, TestEvidence, ProtocolError, TIMDR_AI_System)
from timdr_workflow import WorkflowPlan, TIMDRWorkflow, SIGNAL_STAGES
import timdr_answer_engine as engine


def prepared():
    protocol = TIMDRProtocol()
    plan = protocol.preregister(Hypothesis("h", "d", "e", {"method": "permutation", "threshold": 2}))
    return protocol, plan


def test_evidence_requires_prior_plan_and_binding():
    p, plan = prepared()
    controls = ControlResult(True, True)
    evidence = TestEvidence(.001, .8, "permutation", plan.fingerprint)
    assert p.run_test(controls, evidence).verdict == "INCONCLUSIVE"
    assert p.run_test(controls, TestEvidence(.001, .8, "permutation"), preregistration=plan).verdict == "INCONCLUSIVE"
    assert p.run_test(controls, evidence, preregistration=plan).verdict == "SUPPORTED"


def test_changed_method_or_reference_cannot_confirm_original_plan():
    p, plan = prepared()
    assert p.run_test(ControlResult(True, True), TestEvidence(.001, .8, "different", plan.fingerprint), preregistration=plan).verdict == "INCONCLUSIVE"
    plan.frozen_params["threshold"] = 3
    with pytest.raises(ProtocolError):
        p.run_test(ControlResult(True, True), TestEvidence(.001, .8, "permutation", plan.fingerprint), preregistration=plan)


@pytest.mark.parametrize("effect", [float("inf"), float("nan")])
def test_nonfinite_effect_is_rejected(effect):
    p, plan = prepared()
    with pytest.raises(ProtocolError):
        p.run_test(ControlResult(True, True), TestEvidence(.001, effect, "permutation", plan.fingerprint), preregistration=plan)


def test_system_does_not_create_preregistration_from_supplied_results():
    s = TIMDR_AI_System()
    result = s.run([], {"params": {"method": "permutation"}}, controls=ControlResult(True, True), evidence=TestEvidence(.001, .8, "permutation"))
    assert result["preregistration"] is None
    assert result["test_result"].verdict == "INCONCLUSIVE"


def test_signal_order_and_frozen_reference_are_preserved():
    raw = [1, 2, 3]; reference = {"centre": 10}
    plan = WorkflowPlan.freeze(route="signal", clock="seconds", anchor="calibration sensor", calibration=reference)
    reference["centre"] = 999
    seen = []
    def adapter(ctx):
        seen.append((ctx["reference"]["centre"], list(ctx["raw"])))
        ctx["raw"][0] = 999
        ctx["reference"]["centre"] = 999
        return len(ctx["states"])
    result = TIMDRWorkflow(plan, {k: adapter for k in SIGNAL_STAGES}).run(raw)
    assert [r["stage"] for r in result["trace"]] == list(SIGNAL_STAGES)
    assert all(x == (10, [1, 2, 3]) for x in seen)
    assert raw == [1, 2, 3]
    assert result["empirical_verdict"] == "INCONCLUSIVE"


def test_missing_adapters_are_not_silent_identity_operations():
    plan = WorkflowPlan.freeze(route="signal", clock="seconds", anchor="sensor", calibration={})
    with pytest.raises(ProtocolError):
        TIMDRWorkflow(plan, {})


def test_optional_geometry_requires_reason_and_text_is_separate():
    plan = WorkflowPlan.freeze(route="signal", clock="seconds", anchor="sensor", calibration={}, skips={"geometry": "Only a scalar time series is available"})
    result = TIMDRWorkflow(plan, {k: lambda ctx: 1 for k in SIGNAL_STAGES if k != "geometry"}).run([1])
    assert result["trace"][-1]["status"] == "skipped"
    with pytest.raises(ProtocolError):
        WorkflowPlan.freeze(route="text", clock="conversation turn", anchor="source records", calibration={}, skips={"validation": "skip"})


@pytest.mark.parametrize("bad", ["TIMDR guarantees truth.", "AI interpretation establishes SUPPORTED"])
def test_unverified_explanation_is_not_published(monkeypatch, bad):
    monkeypatch.setattr(engine, "explain", lambda *args: bad)
    r = engine.answer("Czy AI TIMDR moze oglosic wynik SUPPORTED?")
    assert bad not in r["answer"]
    assert r["explanation_status"] == "rejected_unverified"


def test_citation_and_status_do_not_make_added_claim_safe(monkeypatch):
    monkeypatch.setattr(engine, "explain", lambda q, verified, v: verified + " This always guarantees truth.")
    r = engine.answer("Czy AI TIMDR moze oglosic wynik SUPPORTED?")
    assert "guarantees" not in r["answer"]
    assert r["explanation_status"] == "rejected_unverified"


def test_system_uses_explicit_workflow_and_keeps_empirical_gate_separate():
    plan = WorkflowPlan.freeze(route="signal", clock="seconds", anchor="sensor", calibration={})
    w = TIMDRWorkflow(plan, {k: lambda ctx: len(ctx["raw"]) for k in SIGNAL_STAGES})
    r = TIMDR_AI_System(workflow=w).run([1, 2], {"name": "candidate"})
    assert r["model_output"]["states"]["sieve"] == 2
    assert r["test_result"].verdict == "INCONCLUSIVE"


def test_real_modal_operator_workflow_separates_reference_from_other_tone():
    import numpy as np
    from timdr_operators import fft_dominant_mode, Modality, is_resonant
    plan = WorkflowPlan.freeze(
        route="signal", clock="seconds at 1000 Hz", anchor="40 Hz calibration tone",
        calibration={"frequency": 40.0, "phase": 0.0},
        skips={"self_correction": "Fixed reference; adaptive tuning is not part of this experiment",
               "geometry": "No spatial measurements"})
    def resonance(ctx):
        f, phase, amplitude = fft_dominant_mode(ctx["states"]["field"], fs=1000)
        reference = ctx["reference"]
        return is_resonant(Modality(f, phase, amplitude),
                           Modality(reference["frequency"], reference["phase"]), eps_f=.01, eps_phi=.01)
    adapters = {"signal_type": lambda ctx: "modal_test_fixture",
                "field": lambda ctx: ctx["raw"], "resonance": resonance,
                "sieve": lambda ctx: ctx["states"]["field"] if ctx["states"]["resonance"] else []}
    workflow = TIMDRWorkflow(plan, adapters)
    t = np.arange(1000)/1000
    positive = workflow.run(np.sin(2*np.pi*40*t))
    negative = workflow.run(np.sin(2*np.pi*80*t))
    assert positive["states"]["resonance"]
    assert not negative["states"]["resonance"]
    assert len(positive["states"]["sieve"]) == 1000
    assert len(negative["states"]["sieve"]) == 0
    assert positive["empirical_verdict"] == "INCONCLUSIVE"


def test_answer_engine_runs_text_route_in_order_without_model():
    r = engine.answer("Czy AI TIMDR moze oglosic wynik SUPPORTED?", use_lora=False)
    assert [s["stage"] for s in r["workflow_trace"]] == ["claims", "sources", "candidate", "validation"]
    assert r["explanation_status"] == "not_requested"
