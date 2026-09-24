"""Small self-contained TIMDR answer engine backed by the Claim Graph."""
from claim_graph_gate import decide, render


def answer(question: str) -> dict:
    """Return a source-backed answer without calling a language model."""
    decision = decide(question)
    return {
        "answer": render(decision),
        "verdict": decision.verdict,
        "record_ids": list(decision.record_ids),
        "is_in_scope": decision.node_id is not None,
    }
