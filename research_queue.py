"""Turn ranked online documents into auditable, branch-local research candidates."""
from __future__ import annotations

import json
from pathlib import Path


REQUIREMENTS = {
    "M/S signal": ["time-indexed signal", "data license", "pre-registered target and controls"],
    "G geometry": ["geometry/mesh or trajectory", "explicit time alignment", "data license and controls"],
    "K modal": ["frequency/phase representation", "aligned sampling metadata", "pre-registered target and controls"],
    "META-DYNAMICS": ["defined Lambda-tau-rho-J series", "independent channels", "pre-registered target and controls"],
    "UNROUTED": ["manual domain review", "identify one TIMDR branch before any hypothesis"],
}


def build_queue(ranking_path: str | Path, state_path: str | Path) -> dict:
    ranking = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    entries = []
    for item in ranking.get("documents", []):
        matches = item.get("branch_matches", {})
        best_score = max(matches.values(), default=0)
        branch = min((name for name, score in matches.items() if score == best_score), default="UNROUTED") if best_score else "UNROUTED"
        document = state.get("documents", {}).get(item["document_key"], {})
        entries.append({
            "candidate_id": item["document_key"],
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source_sha256": document.get("sha256"),
            "branch": branch,
            "branch_score": best_score,
            "status": "CANDIDATE_SOURCE_ONLY",
            "required_before_dataset_or_hypothesis": REQUIREMENTS[branch],
            "prohibited": ["automatic cross-branch bridge", "automatic TIMDR verdict", "holdout access"],
        })
    return {
        "schema": "timdr-research-queue/1",
        "rule": "Each source is routed to exactly one highest-scoring branch; ties use stable lexical order.",
        "entries": sorted(entries, key=lambda value: (-value["branch_score"], value["candidate_id"])),
    }
