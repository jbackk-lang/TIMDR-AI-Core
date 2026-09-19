"""Three protocol outcomes. All numeric evidence here is synthetic demonstration data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from timdr_ai_core import (  # noqa: E402
    ControlResult,
    ProtocolCriteria,
    TestEvidence,
    TIMDR_AI_System,
    TIMDRProtocol,
)


def show(label, result):
    test = result["test_result"]
    print(f"{label}: {test.verdict}")
    print(f"  {test.reason}")


cfg = {
    "name": "synthetic_protocol_demo",
    "description": "Demonstrates protocol states; it is not an empirical TIMDR run.",
    "effect_description": "Difference from a pre-defined background.",
    "params": {"seed": 20260919, "dataset": "synthetic_demo_only"},
}

system = TIMDR_AI_System(
    protocol=TIMDRProtocol(ProtocolCriteria(alpha=0.05, min_abs_effect_size=0.30))
)

show("1. Missing controls", system.run({"signal": [1, 2, 3]}, cfg))

controls = ControlResult(True, True, {"kind": "synthetic controls"})
not_supported = system.run(
    {"signal": [1, 2, 3]}, cfg,
    controls=controls,
    evidence=TestEvidence(0.20, 0.45, "synthetic demonstration test"),
)
show("2. Controls pass, evidence does not", not_supported)

supported_demo = system.run(
    {"signal": [1, 2, 3]}, cfg,
    controls=controls,
    evidence=TestEvidence(0.01, 0.45, "synthetic demonstration test"),
)
show("3. Technical API demonstration only", supported_demo)
print("\nThe final line is not an empirical TIMDR result: its evidence is synthetic.")
