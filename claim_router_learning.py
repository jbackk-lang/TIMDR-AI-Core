"""Small learned router whose output is subordinate to the Claim Graph."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from claim_graph_gate import decide

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "claim_router_dataset_v0.1.json"


def tokens(text: str) -> list[str]:
    raw = re.findall(r"[a-ząćęłńóśźż]+", text.lower())
    # prefixes make small Polish inflection changes visible without external NLP.
    return raw + [word[:5] for word in raw if len(word) >= 5]


def train(rows):
    vocabulary, counts, totals, docs = set(), defaultdict(Counter), Counter(), Counter()
    for text, label in rows:
        docs[label] += 1
        for token in tokens(text):
            vocabulary.add(token); counts[label][token] += 1; totals[label] += 1
    return {"labels": sorted(docs), "vocabulary": sorted(vocabulary), "counts": {label: dict(c) for label, c in counts.items()}, "totals": dict(totals), "docs": dict(docs), "n_docs": sum(docs.values())}


def predict(model, text):
    vocab_size = max(1, len(model["vocabulary"])); toks = tokens(text); scores = {}
    for label in model["labels"]:
        score = math.log((model["docs"][label] + 1) / (model["n_docs"] + len(model["labels"])))
        total = model["totals"][label]
        for token in toks:
            score += math.log((model["counts"].get(label, {}).get(token, 0) + 1) / (total + vocab_size))
        scores[label] = score
    label = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else float("inf")
    return label, margin


def guarded_route(model, question):
    learned_label, margin = predict(model, question)
    logical = decide(question)
    logical_label = logical.node_id or "INCONCLUSIVE_NO_MATCH"
    return {"learned_label": learned_label, "margin": margin, "logical_label": logical_label,
            "agreement": learned_label == logical_label, "verdict": logical.verdict}


def save_model(model, path: Path):
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
