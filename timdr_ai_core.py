"""TIMDR as an epistemic protocol, not an autonomous proof engine.

This module deliberately separates four things that must not be conflated:
formal mathematics, deterministic code, empirical evidence, and AI-assisted
interpretation.  In particular, an AI or a placeholder LTR pipeline cannot
manufacture a ``SUPPORTED`` verdict: that verdict requires an explicit,
pre-registered test result and passed positive and negative controls.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from typing import Any, Callable, Literal, Mapping, Optional


Verdict = Literal["SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"]
Transform = Callable[[Any], Any]


class ProtocolError(ValueError):
    """Raised when a purported experiment does not meet protocol requirements."""


def _canonical_json(value: Any) -> str:
    """Serialize a configuration deterministically or fail before preregistration."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Preregistration parameters must be JSON-serializable.") from exc


@dataclass(frozen=True)
class Hypothesis:
    name: str
    description: str
    effect_description: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class Preregistration:
    hypothesis: Hypothesis
    fingerprint: str
    frozen_params: Mapping[str, Any]
    protocol_version: str = "timdr-ai-core/1"


@dataclass(frozen=True)
class ControlResult:
    """Result of controls supplied by a domain-specific, reproducible runner."""

    positive_ok: bool
    negative_ok: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.positive_ok and self.negative_ok


@dataclass(frozen=True)
class TestEvidence:
    """An already computed result from the pre-registered domain test.

    ``method`` may be Mann-Whitney, a block permutation test, Spearman, or a
    branch-specific test.  This general core does not substitute one test for
    another, because that choice belongs to the preregistration.
    """

    p_value: float
    effect_size: float
    method: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestResult:
    p_value: Optional[float]
    effect_size: Optional[float]
    verdict: Verdict
    reason: str
    method: Optional[str] = None


@dataclass(frozen=True)
class ProtocolCriteria:
    alpha: float = 0.05
    min_abs_effect_size: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ProtocolError("alpha must be strictly between 0 and 1.")
        if self.min_abs_effect_size < 0.0:
            raise ProtocolError("min_abs_effect_size cannot be negative.")


class TIMDRProtocol:
    """Protocol guardrail which never infers evidence from model output."""

    def __init__(self, criteria: ProtocolCriteria = ProtocolCriteria()) -> None:
        self.criteria = criteria

    def preregister(self, hypothesis: Hypothesis) -> Preregistration:
        frozen_params = deepcopy(dict(hypothesis.params))
        payload = {
            "name": hypothesis.name,
            "description": hypothesis.description,
            "effect_description": hypothesis.effect_description,
            "params": frozen_params,
            "criteria": asdict(self.criteria),
            "protocol_version": "timdr-ai-core/1",
        }
        fingerprint = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        frozen_hypothesis = Hypothesis(
            name=hypothesis.name,
            description=hypothesis.description,
            effect_description=hypothesis.effect_description,
            params=deepcopy(frozen_params),
        )
        return Preregistration(frozen_hypothesis, fingerprint, frozen_params)

    def run_controls(self, controls: Optional[ControlResult]) -> ControlResult:
        """Accept only an explicit control outcome; missing controls never pass."""
        if controls is None:
            return ControlResult(
                positive_ok=False,
                negative_ok=False,
                details={"reason": "No reproducible positive/negative controls were supplied."},
            )
        return controls

    def run_test(self, controls: ControlResult, evidence: Optional[TestEvidence]) -> TestResult:
        if not controls.passed:
            return TestResult(
                p_value=None,
                effect_size=None,
                verdict="INCONCLUSIVE",
                reason="Controls did not both pass; the main test is not interpretable.",
            )
        if evidence is None:
            return TestResult(
                p_value=None,
                effect_size=None,
                verdict="INCONCLUSIVE",
                reason="No reproducible, pre-registered test evidence was supplied.",
            )
        if not 0.0 <= evidence.p_value <= 1.0:
            raise ProtocolError("p_value must be in [0, 1].")
        if not evidence.method.strip():
            raise ProtocolError("A named pre-registered test method is required.")

        if evidence.p_value <= self.criteria.alpha and abs(evidence.effect_size) >= self.criteria.min_abs_effect_size:
            return TestResult(
                evidence.p_value,
                evidence.effect_size,
                "SUPPORTED",
                "Controls passed and the pre-registered significance/effect criteria were met.",
                evidence.method,
            )
        return TestResult(
            evidence.p_value,
            evidence.effect_size,
            "NOT_SUPPORTED",
            "Controls passed, but the pre-registered significance/effect criteria were not met.",
            evidence.method,
        )


class _TransformLayer:
    """Explicit adapter point. Identity is a transport default, not a TIMDR operator."""

    def __init__(self, transform: Optional[Transform] = None) -> None:
        self.transform = transform or (lambda value: value)

    def forward(self, value: Any) -> Any:
        return self.transform(value)


class LayerTTopology(_TransformLayer):
    pass


class LayerIInformation(_TransformLayer):
    pass


class LayerMModal(_TransformLayer):
    def __init__(self, mode: str = "tetra_earth", transform: Optional[Transform] = None) -> None:
        super().__init__(transform)
        self.mode = mode


class LayerItTemporal(_TransformLayer):
    pass


class LayerRResonance(_TransformLayer):
    pass


class LayerEEmergence(_TransformLayer):
    pass


class FundamentalModelLTR:
    """Composable Λ–τ–ρ pipeline; it supplies representations, not truth claims."""

    def __init__(
        self,
        topology: Optional[LayerTTopology] = None,
        information: Optional[LayerIInformation] = None,
        modal: Optional[LayerMModal] = None,
        temporal: Optional[LayerItTemporal] = None,
        resonance: Optional[LayerRResonance] = None,
        emergence: Optional[LayerEEmergence] = None,
    ) -> None:
        self.T = topology or LayerTTopology()
        self.I = information or LayerIInformation()
        self.M = modal or LayerMModal()
        self.It = temporal or LayerItTemporal()
        self.R = resonance or LayerRResonance()
        self.E = emergence or LayerEEmergence()

    def forward(self, value: Any) -> Any:
        return self.E.forward(self.R.forward(self.It.forward(self.M.forward(self.I.forward(self.T.forward(value))))))


class TIMDR_AI_System:
    """Combines a transform pipeline with the epistemic guardrail."""

    def __init__(self, protocol: Optional[TIMDRProtocol] = None, model: Optional[FundamentalModelLTR] = None) -> None:
        self.protocol = protocol or TIMDRProtocol()
        self.model = model or FundamentalModelLTR()

    def run(
        self,
        raw_input: Any,
        hypothesis_cfg: Mapping[str, Any],
        *,
        controls: Optional[ControlResult] = None,
        evidence: Optional[TestEvidence] = None,
    ) -> dict[str, Any]:
        hypothesis = Hypothesis(
            name=str(hypothesis_cfg.get("name", "default_hypothesis")),
            description=str(hypothesis_cfg.get("description", "")),
            effect_description=str(hypothesis_cfg.get("effect_description", "")),
            params=deepcopy(dict(hypothesis_cfg.get("params", {}))),
        )
        preregistration = self.protocol.preregister(hypothesis)
        control_result = self.protocol.run_controls(controls)
        test_result = self.protocol.run_test(control_result, evidence)
        return {
            "preregistration": preregistration,
            "controls": control_result,
            "test_result": test_result,
            "model_output": self.model.forward(raw_input),
            "ai_role": "epistemic_filter_not_proof_or_empirical_authority",
        }


if __name__ == "__main__":
    system = TIMDR_AI_System()
    result = system.run(
        raw_input={"input": "example signal"},
        hypothesis_cfg={
            "name": "example",
            "description": "Demonstration of protocol gating.",
            "effect_description": "A pre-registered difference from background.",
            "params": {"n_runs": 1},
        },
    )
    print(result["preregistration"].fingerprint)
    print(result["test_result"])
