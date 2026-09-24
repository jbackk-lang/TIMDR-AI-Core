"""Deterministic Phase 1B evidence selector; it never trains or edits model weights."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "timdr_knowledge_injection_v0.1.json"
GIA = ROOT.parent / "GIA-TIMDR"


def words(text):
    return set(re.findall(r"[a-ząćęłńóśźż]+", text.lower()))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieve(question, records, limit=3):
    query = words(question)
    # Explicit operator routing comes before lexical overlap. It prevents a
    # common Polish function word from being mistaken for evidence.
    routes = set()
    if query & {"rho", "źródła", "źródło", "kierunków", "energii"}:
        routes.add("branch-meta")
    if "chronoproces" in query:
        routes.add("chronoprocess")
    if query & {"weingarten", "krzywej", "powierzchni", "geometria"}:
        routes.add("branch-g")
    if "rezonans" in query:
        routes.update({"branch-ms", "branch-k"})
    if query & {"modalność", "modalny", "częstotliwość", "faza"}:
        routes.add("branch-k")
    if query & {"holdoutu", "holdout", "progu", "zamrożeniu", "eksploracją"}:
        routes.add("protocol-freeze")
    if query & {"sygnał", "anomalia", "defekt"}:
        routes.add("branch-ms")
    if routes:
        return [record for record in records if record["id"] in routes][:limit]
    # Conservative fallback: only meaningful terms may select a record.
    query = {token for token in query if len(token) >= 4}
    ranked = []
    for record in records:
        evidence = " ".join(str(record[key]) for key in ("branch", "modality", "operator", "claim"))
        score = len(query & {token for token in words(evidence) if len(token) >= 4})
        ranked.append((score, record))
    return [record for score, record in sorted(ranked, key=lambda pair: (-pair[0], pair[1]["id"])) if score > 0][:limit]


def trajectory(question, capsule):
    evidence = retrieve(question, capsule["records"])
    source_hashes = {}
    for record in evidence:
        source = GIA / record["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hashes[record["source"]] = sha(source)
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "question": question,
            "selected_record_ids": [record["id"] for record in evidence], "source_sha256": source_hashes,
            "evidence": [{key: record[key] for key in ("id", "branch", "modality", "operator", "claim", "source")} for record in evidence],
            "status": "EVIDENCE_READY" if evidence else "INCONCLUSIVE_NO_MATCH"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    capsule = json.loads(DATA.read_text(encoding="utf-8"))
    result = trajectory(args.question, capsule)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
