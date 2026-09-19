"""Frozen, metadata-only PROTECT-90 learning bundle.

The source is an EMT simulation benchmark, not live grid telemetry.  This module
prepares a reproducible four-class *research baseline* and does not implement a
protection relay or a TIMDR B4 verdict.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from learning_sandbox import Sample


DATASET_ID = "PROTECT-90 v1.0.0"
PREREG_SCHEMA = "timdr-protect90-learning-prereg/1"
PREREG_VERSION = "PROTECT90_MULTICLASS_v0.1"
SPLIT_SEED = "PROTECT90_MULTICLASS_v0.1:fixed-stratified"
FEATURE_COLUMNS = (
    "t_evnt_start", "t_evnt_end", "sc_location", "phase_select",
    "fault_resistance", "line_1_2_a_length", "line_2_3_a_length",
    "ext_grid_1_u_setp", "ext_grid_1_short_circuit_power",
    "load_3_plini", "load_3_qlini",
)


class Protect90Error(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _source_paths(source_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(source_root)
    base = root / "data" / "protect90"
    return (
        base / "frozen" / "B4_GRID_PROTECT90_FROZEN_MANIFEST.json",
        base / "frozen" / "B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json",
        base / "raw" / "hv_double_line_90kv_labels.csv",
    )


def _labels(labels_path: Path) -> dict[int, dict[str, str]]:
    with labels_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    answer: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            sample_id = int(row["sample_id"])
            int(row["sc_type"])
            for column in FEATURE_COLUMNS:
                float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise Protect90Error(f"Invalid label row: {row!r}") from exc
        answer[sample_id] = row
    return answer


def _episode_ids(ids_payload: dict[str, Any]) -> list[int]:
    candidates = ids_payload.get("episode_ids", ids_payload.get("sample_ids"))
    if not isinstance(candidates, list) or not all(isinstance(value, int) for value in candidates):
        raise Protect90Error("Frozen episode-id file has no integer episode_ids/sample_ids list.")
    if len(candidates) != len(set(candidates)):
        raise Protect90Error("Frozen episode-id file contains duplicate identifiers.")
    return candidates


def _split_ids(ids: list[int], rows: dict[int, dict[str, str]]) -> dict[str, list[int]]:
    by_class: dict[int, list[int]] = {}
    for sample_id in ids:
        if sample_id not in rows:
            raise Protect90Error(f"Frozen sample {sample_id} is absent from labels CSV.")
        by_class.setdefault(int(rows[sample_id]["sc_type"]), []).append(sample_id)
    if set(by_class) != {0, 1, 2, 3}:
        raise Protect90Error(f"Expected four sc_type classes 0..3; found {sorted(by_class)}.")
    result = {"train": [], "calibration": [], "holdout": []}
    for label, class_ids in sorted(by_class.items()):
        ordered = sorted(class_ids, key=lambda value: sha256(f"{SPLIT_SEED}:{label}:{value}".encode()).hexdigest())
        n = len(ordered)
        train_end, calibration_end = (n * 3) // 5, (n * 4) // 5
        result["train"].extend(ordered[:train_end])
        result["calibration"].extend(ordered[train_end:calibration_end])
        result["holdout"].extend(ordered[calibration_end:])
    return {split: sorted(values) for split, values in result.items()}


def build_preregistration(source_root: str | Path) -> dict[str, Any]:
    manifest_path, ids_path, labels_path = _source_paths(source_root)
    if not all(path.is_file() for path in (manifest_path, ids_path, labels_path)):
        raise Protect90Error("Missing PROTECT-90 frozen manifest, IDs, or labels CSV in source repository.")
    manifest, ids_payload, rows = _read_json(manifest_path), _read_json(ids_path), _labels(labels_path)
    if manifest.get("dataset_id") != DATASET_ID:
        raise Protect90Error(f"Unexpected dataset_id: {manifest.get('dataset_id')!r}")
    ids = _episode_ids(ids_payload)
    splits = _split_ids(ids, rows)
    return {
        "schema": PREREG_SCHEMA,
        "version": PREREG_VERSION,
        "state": "FROZEN_BEFORE_FIRST_FIT",
        "dataset": {
            "dataset_id": DATASET_ID,
            "source_type": "physically grounded EMT simulation; not field data",
            "frozen_manifest_sha256": _sha256_file(manifest_path),
            "frozen_episode_ids_sha256": _sha256_file(ids_path),
            "labels_csv_sha256": _sha256_file(labels_path),
            "n_selected_episodes": len(ids),
        },
        "task": {
            "name": "classify_sc_type",
            "label": "sc_type",
            "classes": [0, 1, 2, 3],
            "features": list(FEATURE_COLUMNS),
            "model": "nearest_centroid_multiclass",
        },
        "split": {
            "seed": SPLIT_SEED,
            "rule": "stratify sc_type; SHA-256 order; first 60% train, next 20% calibration, final 20% holdout per class",
            "episode_ids": splits,
            "counts": {name: len(values) for name, values in splits.items()},
        },
        "holdout_policy": "Holdout must not be accessed by fitting, model selection, or calibration.",
        "scope": "Research baseline only. It cannot operate a grid protection device and is not a TIMDR B4 result.",
    }


def write_preregistration(source_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    payload = build_preregistration(source_root)
    target = Path(output_path)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != serialized:
        raise Protect90Error(f"Refusing to overwrite a different frozen preregistration: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(serialized, encoding="utf-8")
    return payload


def samples_from_preregistration(source_root: str | Path, prereg_path: str | Path) -> list[Sample]:
    prereg = _read_json(Path(prereg_path))
    if prereg.get("schema") != PREREG_SCHEMA or prereg.get("state") != "FROZEN_BEFORE_FIRST_FIT":
        raise Protect90Error("Invalid or unfrozen PROTECT-90 preregistration.")
    manifest_path, ids_path, labels_path = _source_paths(source_root)
    expected = prereg["dataset"]
    actual_hashes = {
        "frozen_manifest_sha256": _sha256_file(manifest_path),
        "frozen_episode_ids_sha256": _sha256_file(ids_path),
        "labels_csv_sha256": _sha256_file(labels_path),
    }
    if any(expected.get(name) != value for name, value in actual_hashes.items()):
        raise Protect90Error("Source artifact hashes differ from the frozen preregistration.")
    rows = _labels(labels_path)
    result: list[Sample] = []
    for split, sample_ids in prereg["split"]["episode_ids"].items():
        if split not in ("train", "calibration", "holdout"):
            raise Protect90Error(f"Unknown split: {split}")
        for sample_id in sample_ids:
            row = rows[int(sample_id)]
            result.append(Sample(tuple(float(row[name]) for name in FEATURE_COLUMNS), int(row["sc_type"]), split))
    return result
