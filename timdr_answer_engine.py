"""Small self-contained TIMDR answer engine backed by the Claim Graph."""
from claim_graph_gate import decide, render
from timdr_lora_responder import explain


def answer(question: str, use_lora: bool = True) -> dict:
    """Return a Claim Graph answer, optionally followed by a local LoRA explanation."""
    decision = decide(question)
    verified = render(decision)
    explanation = ""
    if use_lora and decision.node_id is not None:
        try:
            explanation = explain(question, verified, decision.verdict)
        except Exception as error:
            explanation = f"[Lokalny adapter niedostępny: {error}]"
    return {
        "answer": verified + (f"\n\nWyjaśnienie lokalnego Qwen + LoRA:\n{explanation}" if explanation else ""),
        "verdict": decision.verdict,
        "record_ids": list(decision.record_ids),
        "is_in_scope": decision.node_id is not None,
    }
