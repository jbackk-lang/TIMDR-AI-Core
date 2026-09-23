"""Small, automatic online learner for an external source graph.

It follows only explicit HTTPS catalog seeds, fetches a bounded number of text
documents, and updates a local provenance/term graph. It never executes remote
code, uploads local data, modifies preregistrations, or emits a TIMDR verdict.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

from environment_policy import BUDGET, EnvironmentPolicyError


class OnlineLearningError(RuntimeError):
    pass


def _https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise OnlineLearningError("Online learning accepts HTTPS catalog and document URLs only.")


def _read_url(url: str, limit: int) -> bytes:
    _https(url)
    received, blocks = 0, []
    try:
        with urlopen(Request(url, headers={"User-Agent": "TIMDR-AI-Core-online/0.1"}), timeout=30) as response:
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                received += len(block)
                if received > limit:
                    raise OnlineLearningError(f"Remote item exceeds {limit} byte budget.")
                blocks.append(block)
    except URLError as exc:
        raise OnlineLearningError(f"Cannot fetch {url}: {exc.reason}") from exc
    return b"".join(blocks)


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in re.findall(r"[A-Za-zÀ-ÿ0-9_-]{3,}", text))


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema": "timdr-online-learning-state/1", "documents": {}, "terms": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def run_cycle(config_path: str | Path, state_path: str | Path) -> dict:
    """Automatically update a tiny local knowledge graph from configured feeds.

    Catalog JSON format: {"items": [{"id": "...", "url": "...", "title": "..."}]}.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema") == "timdr-online-learning-config/1":
        catalogs = config.get("catalogs", [])
    elif isinstance(config.get("items"), list):
        # A compact local list of documents is also a valid one-source catalog.
        catalogs = [{"id": "local-items", "items": config["items"]}]
    else:
        raise OnlineLearningError(
            "Unsupported configuration. Use either schema timdr-online-learning-config/1 "
            "with catalogs, or a JSON object containing an items list."
        )
    state_file = Path(state_path)
    state = _load_state(state_file)
    processed = 0
    refreshed = 0
    for seed in catalogs:
        catalog = seed if "items" in seed else json.loads(
            _read_url(seed["url"], BUDGET.max_online_document_bytes).decode("utf-8")
        )
        for item in catalog.get("items", []):
            if processed >= BUDGET.max_online_documents_per_cycle:
                break
            item_id, url = str(item["id"]), str(item["url"])
            _https(url)
            document_key = sha256(f"{seed['id']}:{item_id}:{url}".encode()).hexdigest()
            existing = state["documents"].get(document_key)
            if existing is not None and "top_terms" in existing:
                continue
            raw = _read_url(url, BUDGET.max_online_document_bytes)
            text = raw.decode("utf-8", errors="replace")
            terms = _tokens(text)
            if existing is not None:
                existing["top_terms"] = dict(terms.most_common(128))
                existing["term_count"] = sum(terms.values())
                existing["refreshed_at"] = datetime.now(timezone.utc).isoformat()
                refreshed += 1
                continue
            state["documents"][document_key] = {
                "catalog_id": seed["id"], "item_id": item_id, "url": url,
                "title": str(item.get("title", "")), "sha256": sha256(raw).hexdigest(),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "term_count": sum(terms.values()),
                "top_terms": dict(terms.most_common(128)),
            }
            for term, count in terms.items():
                state["terms"][term] = state["terms"].get(term, 0) + count
            processed += 1
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "new_documents": processed,
        "refreshed_documents": refreshed,
        "total_documents": len(state["documents"]),
        "holdout_accessed": False,
    }
