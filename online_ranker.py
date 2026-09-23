"""Rank online documents as research candidates without mixing TIMDR branches."""
from __future__ import annotations

import json
from pathlib import Path
import re


BRANCH_TERMS = {
    "M/S signal": {"signal", "time", "series", "anomaly", "defect", "transient", "bearing", "fault", "waveform"},
    "G geometry": {"geometry", "geometric", "topology", "topological", "curvature", "mesh", "weingarten"},
    "K modal": {"modal", "mode", "spectral", "spectrum", "frequency", "phase", "fourier", "laplacian"},
    "META-DYNAMICS": {"lambda", "tau", "rho", "meta", "dynamics", "dynamic", "aggregate", "operator"},
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_-]{3,}", text.lower()))


def rank_state(state_path: str | Path) -> dict:
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    ranked = []
    for key, document in state.get("documents", {}).items():
        observed = set(document.get("top_terms", {})) | _words(document.get("title", "")) | _words(document.get("url", ""))
        branches = {branch: len(words & observed) for branch, words in BRANCH_TERMS.items()}
        ranked.append({
            "document_key": key,
            "title": document.get("title", ""),
            "url": document.get("url", ""),
            "branch_matches": branches,
            "candidate_score": sum(branches.values()),
            "status": "candidate_only; not a hypothesis, result, or cross-branch bridge",
        })
    return {
        "schema": "timdr-online-source-ranking/1",
        "method": "keyword routing per independent TIMDR branch; no claim inference",
        "documents": sorted(ranked, key=lambda item: (-item["candidate_score"], item["document_key"])),
    }
