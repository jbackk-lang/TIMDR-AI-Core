"""Source-backed answers using the explicit TIMDR text workflow."""
from claim_graph_gate import decide, render, gate_candidate
from timdr_lora_responder import explain
from timdr_workflow import WorkflowPlan, TIMDRWorkflow


def answer(question: str, use_lora: bool = True) -> dict:
    plan = WorkflowPlan.freeze(route="text", clock="question turn",
                               anchor="hash-checked Claim Graph source records", calibration={})

    def candidate(ctx):
        decision = ctx["states"]["sources"]
        verified = render(decision)
        if not use_lora or decision.node_id is None:
            return {"text": "", "status": "not_requested"}
        try:
            return {"text": explain(question, verified, decision.verdict), "status": "generated"}
        except Exception:
            return {"text": "", "status": "unavailable"}

    def validate(ctx):
        decision = ctx["states"]["sources"]
        proposed = ctx["states"]["candidate"]
        verified = render(decision)
        if proposed["status"] != "generated":
            return {"text": verified, "status": proposed["status"]}
        text = proposed["text"]
        # Lexical checks do not establish semantic equivalence. Free paraphrases
        # remain unpublished until an independently tested validator exists.
        accepted = (gate_candidate(decision, text)["accepted"] and
                    " ".join(text.split()) == " ".join(verified.split()))
        return {"text": verified, "status": "verified_copy" if accepted else "rejected_unverified"}

    result = TIMDRWorkflow(plan, {
        "claims": lambda ctx: ctx["raw"],
        "sources": lambda ctx: decide(ctx["states"]["claims"]),
        "candidate": candidate, "validation": validate,
    }).run(question)
    decision = result["states"]["sources"]
    publication = result["states"]["validation"]
    return {"answer": publication["text"], "verdict": decision.verdict,
            "explanation_status": publication["status"],
            "record_ids": list(decision.record_ids), "is_in_scope": decision.node_id is not None,
            "workflow_trace": result["trace"]}
