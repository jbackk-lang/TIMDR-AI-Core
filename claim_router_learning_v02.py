"""Claim Router v0.2: learned suggestion with confidence gate, never a verdict authority."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from claim_graph_gate import decide

ROOT = Path(__file__).resolve().parent
V1 = ROOT / "data" / "claim_router_dataset_v0.1.json"
PLAN = ROOT / "data" / "claim_router_v0.2_plan.json"


def tokens(text):
    raw = re.findall(r"[a-ząćęłńóśźż]+", text.lower())
    normalized = ["holdout" if word.startswith("holdout") else word for word in raw]
    return normalized + [word[:5] for word in normalized if len(word) >= 5]


def train(rows):
    vocabulary, counts, totals, docs = set(), defaultdict(Counter), Counter(), Counter()
    for text, label in rows:
        docs[label] += 1
        for token in tokens(text): vocabulary.add(token); counts[label][token] += 1; totals[label] += 1
    return {"labels":sorted(docs),"vocabulary":sorted(vocabulary),"counts":{k:dict(v) for k,v in counts.items()},"totals":dict(totals),"docs":dict(docs),"n_docs":sum(docs.values())}


def predict(model, question, minimum_margin=1.0):
    vocab=max(1,len(model["vocabulary"])); scores={}
    for label in model["labels"]:
        score=math.log((model["docs"][label]+1)/(model["n_docs"]+len(model["labels"])))
        for token in tokens(question): score += math.log((model["counts"].get(label,{}).get(token,0)+1)/(model["totals"][label]+vocab))
        scores[label]=score
    ordered=sorted(scores.items(),key=lambda item:item[1],reverse=True); label,score=ordered[0]; margin=score-ordered[1][1]
    return (label if margin >= minimum_margin else "INCONCLUSIVE_LOW_CONFIDENCE"), margin


def guarded_route(model, question):
    plan=json.loads(PLAN.read_text(encoding="utf-8")); learned,margin=predict(model,question)
    logical=decide(question); logical_label=logical.node_id or "INCONCLUSIVE_NO_MATCH"
    return {"learned_label":learned,"margin":margin,"logical_label":logical_label,"agreement":learned==logical_label,"verdict":logical.verdict}
