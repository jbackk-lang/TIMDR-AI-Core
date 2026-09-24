"""Source-backed claim graph and logical gate for TIMDR answers.

This is inspired by the pipeline discipline of math-validator, not by its
topology-of-equations filter.  Text cannot be given a mathematical continuity
domain by analogy; instead, claims are checked against explicit requirements
and forbidden conclusions.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPSULE = ROOT / "data" / "timdr_knowledge_injection_v0.2.json"
GRAPH = ROOT / "data" / "timdr_claim_graph_v0.1.json"
SKILL_CAPSULE = ROOT / "data" / "timdr_skill_capsule_v0.1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-ząćęłńóśźż]+", text.lower()))


@dataclass(frozen=True)
class Decision:
    node_id: str | None
    verdict: str
    answer: str
    record_ids: tuple[str, ...]
    required: tuple[str, ...]
    forbidden: tuple[str, ...]


def _load():
    capsule = json.loads(CAPSULE.read_text(encoding="utf-8"))
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    if _sha(CAPSULE) != graph["capsule"]["sha256"]:
        raise RuntimeError("INCONCLUSIVE_CAPSULE_CHANGED")
    skill_capsule = json.loads(SKILL_CAPSULE.read_text(encoding="utf-8"))
    if skill_capsule["source"]["sha256"] != graph["skill_capsule"]["source_sha256"]:
        raise RuntimeError("INCONCLUSIVE_SKILL_SOURCE_CHANGED")
    return capsule, {node["id"]: node for node in graph["nodes"]}


def _node(nodes, node_id, answer):
    node = nodes[node_id]
    return Decision(node_id, node["verdict"], answer, tuple(node["record_ids"]),
                    tuple(node["requires"]), tuple(node["forbids"]))


def decide(question: str) -> Decision:
    """Return a structured verdict; unknown queries never fall back to an LLM."""
    _, nodes = _load()
    q = _words(question)
    chrono_alias = {"wspólny", "czas"} <= q and bool(q & {"m", "s", "g", "k", "gałęzie", "gałąź"})
    if "chronoproces" in q or any(word.startswith("chronoproces") for word in q) or chrono_alias:
        return _node(nodes, "chrono-separate", "Nie. Chronoproces daje wspólny indeks czasu, ale zachowuje odrębne rzuty i operatory M/S, G oraz K.")
    gia_alias = "gia" in q or "pca" in q or ("tor" in q and any(word.startswith(("zamroż", "referenc")) for word in q))
    if gia_alias:
        return _node(nodes, "gia-frozen-path", "Tor GIA jest referencją z kalibracji: środek, kierunek PCA i promień trzeba zamrozić przed oceną. Selekcja zgodna z torem nie jest sama w sobie dowodem rezonansu modalnego.")
    if "ai" in q and ("timdr" in q or "supported" in q or "wynik" in q or "empiryczny" in q):
        return _node(nodes, "ai-epistemic-filter", "AI w TIMDR jest filtrem epistemicznym: może pilnować prerejestracji, kontroli i granic wniosków, ale nie jest autorytetem empirycznym i nie może samodzielnie ogłosić wyniku SUPPORTED.")
    if "weingarten" in q or ("krzywej" in q and "powierzchni" in q):
        return _node(nodes, "weingarten-surface", "Nie bez dodatkowej konstrukcji. Klasyczny operator Weingartena wymaga regularnej powierzchni; z krzywej można jawnie zbudować pomocniczą wstęgę albo rurę.")
    if "rezonans" in q and ("modalny" in q or "k" in q):
        return _node(nodes, "resonance-distinct", "Nie. Rezonans M jest koincydencją progową w sygnale, a rezonans K dotyczy wyrównania częstotliwości i fazy modalności.")
    if "fouriera" in q or "gaussowski" in q:
        return _node(nodes, "fourier-scoped", "Nie jako reguła ogólna. Most Fouriera ma tu zakres pojedynczego idealnego impulsu gaussowskiego, nie dowolnych realnych zdarzeń.")
    # More specific M/S-G forms must win before the broad MC/K-G pattern.
    # Otherwise the shared tokens "MC" and "G" route a correct M/S-G question
    # to the rejected K-G bridge.
    if "z0" in q or "gi" in q or {"m", "s", "g"} <= q or ("ms" in q and "g" in q):
        return _node(nodes, "mc-msg-diagnostic", "MC M/S-G #1 ma status ustalonej diagnostyki o zakresie domenowym: silny dla łożysk, częściowy dla sejsmiki i brak dla BTC. Nie jest selektorem danych nieoznaczonych.")
    if ("mc" in q or "k" in q) and ("kg" in q or "g" in q or "möbius" in q or "mobius" in q):
        return _node(nodes, "mc-kg-rejected", "MC K-G jest odrzucony w obecnej formie: wynik został zdominowany artefaktem gęstości kratownicy, więc nie jest wiarygodnym selektorem ani diagnostyką.")
    if "rho" in q or ("j" in q and ("źródła" in q or "źródło" in q)):
        return _node(nodes, "rho-j-separate", "Rho i J muszą mieć rozłączne źródła: rho opisuje odchylenia energii, a J zgodność kierunku krawędzi z torem. Nie dowodzi to ich niezależności statystycznej.")
    if any(word.startswith("holdou") for word in q) or "progu" in q:
        return _node(nodes, "freeze-holdout", "Zmiana progu po zobaczeniu holdoutu tworzy wersję eksploracyjną; nie potwierdza pierwotnego zamrożonego planu.")
    return Decision(None, "INCONCLUSIVE_NO_MATCH", "INCONCLUSIVE_NO_MATCH", (), (), ())


def render(decision: Decision) -> str:
    if decision.node_id is None:
        return decision.answer
    return f"{decision.answer}\nWerdykt: {decision.verdict}\nEvidence: [{', '.join(decision.record_ids)}]"


def gate_candidate(decision: Decision, candidate: str) -> dict:
    """Accept a model phrasing only if it preserves verdict, citations and constraints."""
    text = candidate.lower()
    if decision.node_id is None:
        return {"accepted": candidate.strip() == "INCONCLUSIVE_NO_MATCH", "reason": "unknown_query_requires_exact_refusal"}
    missing_citations = [record_id for record_id in decision.record_ids if record_id.lower() not in text]
    forbidden_hits = [phrase for phrase in decision.forbidden if phrase.lower() in text]
    verdict_ok = decision.verdict.lower() in text
    return {"accepted": not missing_citations and not forbidden_hits and verdict_ok,
            "missing_citations": missing_citations, "forbidden_hits": forbidden_hits, "verdict_present": verdict_ok}
