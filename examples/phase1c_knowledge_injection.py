"""Phase 1C deterministic evidence capsule; it never trains model weights."""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "timdr_knowledge_injection_v0.2.json"
GIA = ROOT.parent / "GIA-TIMDR"


def words(text):
    return set(re.findall(r"[a-ząćęłńóśźż]+", text.lower()))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(capsule):
    mismatches = []
    for relative, expected in capsule["source_manifest"].items():
        path = GIA / relative
        actual = digest(path) if path.is_file() else None
        if actual != expected:
            mismatches.append({"source": relative, "expected": expected, "actual": actual})
    return mismatches


def retrieve(question, records, limit=3):
    """Route only explicit operator questions; unknown questions remain empty."""
    query, routes = words(question), []
    def add(record_id):
        if record_id not in routes:
            routes.append(record_id)
    if query & {"rho", "źródła", "źródło", "energii", "krawędzi", "edge"}: add("meta-source-separation")
    if any(token.startswith("chronoproces") for token in query): add("chronoprocess-time")
    if query & {"weingarten", "krzywej", "powierzchni", "rura", "wstęga"}: add("g-weingarten")
    if query & {"gia", "pca", "toru", "ścieżki", "referencji"}: add("gia-frozen-path")
    if query & {"holdoutu", "holdout", "progu", "zamrożeniu", "kalibracji"}: add("protocol-freeze")
    if query & {"fouriera", "gauss", "gaussowski", "impulsu"}: add("fourier-bridge-scope")
    if query & {"mc", "mobius", "möbius", "kg"}: add("mc-kg-rejected")
    if query & {"z0", "gi", "łożysk", "sejsmiczny", "btc"}: add("mc-msg-diagnostic")
    if "rezonans" in query:
        add("ms-signal"); add("k-modal")
    elif query & {"modalny", "modalność", "częstotliwość", "faza"}: add("k-modal")
    elif query & {"sygnał", "anomalia", "defekt"}: add("ms-signal")
    by_id = {record["id"]: record for record in records}
    return [by_id[record_id] for record_id in routes if record_id in by_id][:limit]


def trajectory(question, capsule):
    mismatches = verify_manifest(capsule)
    if mismatches:
        return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "question": question, "selected_record_ids": [], "status": "INCONCLUSIVE_SOURCE_CHANGED", "source_manifest_mismatches": mismatches, "evidence": []}
    evidence = retrieve(question, capsule["records"])
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "question": question, "selected_record_ids": [record["id"] for record in evidence], "status": "EVIDENCE_READY" if evidence else "INCONCLUSIVE_NO_MATCH", "source_sha256": {record["source"]: capsule["source_manifest"][record["source"]] for record in evidence}, "evidence": [{key: record[key] for key in ("id", "branch", "modality", "object", "operator", "claim", "status", "scope", "limitation", "source")} for record in evidence]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("question"); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); capsule = json.loads(DATA.read_text(encoding="utf-8")); result = trajectory(args.question, capsule)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
