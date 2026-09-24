"""Bounded, skill-guided learning from declared online source catalogs.

The module first uses :mod:`online_learning` to retrieve text-only documents
from explicit HTTPS catalogs.  It then creates candidate study cards using an
immutable TIMDR skill capsule.  It deliberately never edits the Claim Graph,
never runs downloaded code, and never creates an empirical TIMDR verdict.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from online_learning import run_cycle


ROOT = Path(__file__).resolve().parent
SKILL_CAPSULE = ROOT / "data" / "timdr_skill_capsule_v0.1.json"

BRANCH_TERMS = {
    "M/S signal": {"signal", "vibration", "seismic", "waveform", "fault", "anomaly", "time-series"},
    "G geometry": {"geometry", "curvature", "surface", "mesh", "topology", "trajectory"},
    "K modal": {"modal", "frequency", "phase", "spectrum", "resonance", "fft"},
    "META-DYNAMICS": {"dispersion", "temporal", "dynamics", "regime", "network", "state"},
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _route(top_terms: dict[str, int]) -> list[dict]:
    terms = set(top_terms)
    routes = []
    for branch, vocabulary in BRANCH_TERMS.items():
        hits = sorted(terms & vocabulary)
        if hits:
            routes.append({"branch": branch, "score": len(hits), "matched_terms": hits})
    return sorted(routes, key=lambda item: (-item["score"], item["branch"]))


def run_skill_guided_cycle(config_path: str | Path, cache_dir: str | Path) -> dict:
    """Fetch declared sources then write non-authoritative candidate study cards.

    ``cache_dir`` is expected to be ignored by Git.  The result is an index of
    possible reading/research inputs, not training examples for empirical
    claims and not a modification of any frozen artifact.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    online_report = run_cycle(config_path, cache / "online_learning_state.json")
    skill = _load_json(SKILL_CAPSULE)
    state = _load_json(cache / "online_learning_state.json")

    cards = []
    for key, document in sorted(state["documents"].items()):
        routes = _route(document.get("top_terms", {}))
        if not routes:
            continue
        cards.append({
            "document_key": key,
            "title": document.get("title", ""),
            "url": document["url"],
            "source_sha256": document["sha256"],
            "candidate_branches": routes,
            "status": "CANDIDATE_READING_ONLY",
            "restrictions": [
                "does_not_modify_claim_graph",
                "does_not_create_hypothesis_or_preregistration",
                "does_not_establish_empirical_status",
                "does_not_download_or_execute_remote_code",
            ],
        })

    report = {
        "schema": "timdr-skill-guided-learning-report/0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skill_source_sha256": skill["source"]["sha256"],
        "skill_records_used": [record["id"] for record in skill["records"]],
        "online_cycle": online_report,
        "candidate_cards": cards,
        "holdout_accessed": False,
        "claim_graph_modified": False,
    }
    output = cache / "skill_guided_learning_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"report_path": str(output), "candidate_cards": len(cards), **online_report}
