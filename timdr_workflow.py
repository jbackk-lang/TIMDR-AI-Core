"""Explicit TIMDR construction workflow, not a new neural architecture.

Domain adapters must implement the actual operations. No identity defaults,
no interpretation of text embeddings as physical phase/frequency. Plans are
immutable JSON; each stage receives separate data and frozen calibration.
"""
from dataclasses import dataclass
from copy import deepcopy
import json
from typing import Callable, Mapping, Any
from timdr_ai_core import ProtocolError

SIGNAL_STAGES = ("signal_type", "field", "resonance", "sieve", "self_correction", "geometry")
TEXT_STAGES = ("claims", "sources", "candidate", "validation")

@dataclass(frozen=True)
class WorkflowPlan:
    route: str
    clock: str
    anchor: str
    calibration_json: str
    skips_json: str

    @classmethod
    def freeze(cls, *, route, clock, anchor, calibration, skips=None):
        if route not in ("signal", "text") or not clock.strip() or not anchor.strip():
            raise ProtocolError("Explicit route, clock and anchor are required.")
        skips = dict(skips or {})
        allowed = set(SIGNAL_STAGES if route == "signal" else TEXT_STAGES)
        # Type selection and final text validation cannot be skipped.
        optional = allowed - {"signal_type", "claims", "sources", "candidate", "validation"}
        if set(skips) - optional or any(not str(v).strip() for v in skips.values()):
            raise ProtocolError("Invalid skip or missing domain-specific reason.")
        return cls(route, clock, anchor,
                   json.dumps(calibration, sort_keys=True, allow_nan=False),
                   json.dumps(skips, sort_keys=True))

class TIMDRWorkflow:
    def __init__(self, plan: WorkflowPlan, adapters: Mapping[str, Callable]):
        self.plan = plan
        self.adapters = dict(adapters)
        self.stages = SIGNAL_STAGES if plan.route == "signal" else TEXT_STAGES
        skips = json.loads(plan.skips_json)
        missing = set(self.stages) - set(skips) - set(adapters)
        if missing:
            raise ProtocolError("Missing actual domain adapters: " + ", ".join(sorted(missing)))

    def run(self, raw: Any):
        states = {}
        trace = []
        skips = json.loads(self.plan.skips_json)
        for stage in self.stages:
            if stage in skips:
                trace.append({"stage": stage, "status": "skipped", "reason": skips[stage]})
                continue
            # Copies prevent later stages overwriting raw data or prior outputs.
            ctx = {"raw": deepcopy(raw), "states": deepcopy(states),
                   "reference": json.loads(self.plan.calibration_json),
                   "clock": self.plan.clock, "anchor": self.plan.anchor}
            states[stage] = self.adapters[stage](ctx)
            trace.append({"stage": stage, "status": "computed"})
        # Producing features is not empirical confirmation.
        return {"route": self.plan.route, "states": states, "trace": trace,
                "empirical_verdict": "INCONCLUSIVE"}
