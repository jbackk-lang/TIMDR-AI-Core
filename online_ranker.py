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

# Canonical roles taken from the repository catalogue. These declarations
# outrank a README keyword match: a formalism does not become "UNROUTED"
# merely because its front page does not repeat its branch vocabulary.
EXPLICIT_REPOSITORY_ROUTES = {
    "GIA-TIMDR": ("FRAMEWORK_CORE", "formalization"),
    "TIMDR-Math-Formalism": ("M/S signal", "formalization"),
    "TIMDR-Geometry-Formalism": ("G geometry", "formalization"),
    "TIMDR-Modal-Formalism": ("K modal", "formalization"),
    "TIMDR-Time-Formalism": ("CHRONOPROCESS", "formalization"),
    "TIMDR-META-DYNAMICS": ("META-DYNAMICS", "engineering_tool"),
    "TIMDR-Sygnalizacja": ("M/S signal", "formalization"),
    "TIMDR-Earthquake-Core": ("M/S signal", "engineering_tool"),
    "TIMDR-fusion-tools": ("M/S signal", "engineering_tool"),
    "TIMDR-Grid-Monitor": ("META-DYNAMICS", "engineering_tool"),
    "TIMDR-Industrial-Predict": ("M/S signal", "engineering_tool"),
    "Synoptyk-v3": ("MULTI_BRANCH_TOOL", "engineering_tool"),
}


def _repository_name(title: str) -> str | None:
    prefix = "jbackk-lang/"
    suffix = " README"
    if title.startswith(prefix) and title.endswith(suffix):
        return title[len(prefix):-len(suffix)]
    return None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_-]{3,}", text.lower()))


def rank_state(state_path: str | Path) -> dict:
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    ranked = []
    for key, document in state.get("documents", {}).items():
        observed = set(document.get("top_terms", {})) | _words(document.get("title", "")) | _words(document.get("url", ""))
        branches = {branch: len(words & observed) for branch, words in BRANCH_TERMS.items()}
        repository = _repository_name(document.get("title", ""))
        explicit = EXPLICIT_REPOSITORY_ROUTES.get(repository)
        if explicit:
            branch, role = explicit
            routing = "catalogue declaration (takes precedence over keywords)"
        else:
            branch, role = None, "keyword_candidate"
            routing = "keyword routing only; requires catalogue review"
        ranked.append({
            "document_key": key,
            "title": document.get("title", ""),
            "url": document.get("url", ""),
            "branch_matches": branches,
            "candidate_score": sum(branches.values()),
            "declared_branch": branch,
            "repository_role": role,
            "routing": routing,
            "status": "candidate_only; not a hypothesis, result, or cross-branch bridge",
        })
    return {
        "schema": "timdr-online-source-ranking/2",
        "method": "catalogue declarations take precedence; other sources use keyword routing only; no claim inference",
        "documents": sorted(ranked, key=lambda item: (-item["candidate_score"], item["document_key"])),
    }
