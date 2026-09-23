"""Freeze a Paderborn file-level split from RAR metadata, never waveform values."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


VERSION = "PADERBORN_MS_REPLICATION_v0.1"
SEED = "PADERBORN_MS_REPLICATION_v0.1:file-metadata-only"


class PaderbornPreregError(ValueError):
    pass


def _archive_members(path: Path) -> list[str]:
    try:
        import rarfile
    except ImportError as exc:
        raise PaderbornPreregError("Install archive metadata support: pip install -e '.[paderborn]'") from exc
    return sorted(info.filename.replace("\\", "/") for info in rarfile.RarFile(path).infolist() if info.filename.lower().endswith(".mat"))


def build(source_root: str | Path) -> dict:
    root = Path(source_root)
    manifest_path = root / "data" / "paderborn_candidate" / "PADERBORN_CANDIDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, str]]] = {}
    for archive in manifest["archives"]:
        class_name = archive["class"]
        members = _archive_members(root / "data" / "paderborn_candidate" / "raw" / archive["file"])
        if len(members) != 80:
            raise PaderbornPreregError(f"Expected 80 MATLAB measurements in {archive['file']}; found {len(members)}.")
        groups[class_name] = [{"archive": archive["file"], "member": member} for member in members]
    splits = {"train": [], "calibration": [], "holdout": []}
    for class_name, items in sorted(groups.items()):
        ordered = sorted(items, key=lambda item: sha256(f"{SEED}:{class_name}:{item['member']}".encode()).hexdigest())
        splits["train"].extend(ordered[:48])
        splits["calibration"].extend(ordered[48:64])
        splits["holdout"].extend(ordered[64:])
    return {
        "schema": "timdr-paderborn-dataset-freeze/1",
        "version": VERSION,
        "state": "FROZEN_BEFORE_WAVEFORM_EXTRACTION",
        "selection_method": "RAR member names only; no MATLAB file payload was read.",
        "source_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "classes": sorted(groups),
        "split": {
            "seed": SEED,
            "rule": "per class: SHA-256 order of member name; first 48 train, next 16 calibration, final 16 holdout",
            "counts": {name: len(items) for name, items in splits.items()},
            "members": splits,
        },
        "holdout_policy": "Holdout MATLAB payloads must not be extracted, inspected, used for feature design, trained on, or calibrated on.",
        "next_required_artifact": "A separate M/S hypothesis preregistration defining one signal channel, windows, features, controls, and evaluation criteria.",
    }


def write(source_root: str | Path, target: str | Path) -> dict:
    payload = build(source_root)
    destination = Path(target)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded:
        raise PaderbornPreregError(f"Refusing to overwrite frozen selection: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(encoded, encoding="utf-8")
    return payload
