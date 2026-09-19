"""Immutable evidence reports for pre-registered TIMDR runs.

The runner verifies internal consistency and provenance hashes.  It does not
re-run a source experiment and therefore never upgrades an imported claim.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


class EvidenceError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"Cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"Artifact must contain a JSON object: {path}")
    return value


def import_b4_kitchen_v03(result_path: Path, prereg_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Create a provenance-preserving import of B4-Kitchen v0.3.

    This validates only what can be checked from the three supplied artifacts.
    Archive hashes in the manifest are retained as declarations, not rehashed.
    """
    result, manifest = _json(result_path), _json(manifest_path)
    prereg_text = prereg_path.read_text(encoding="utf-8")
    required = ("dataset_id", "prereg_version", "method", "verdict", "controls", "main_test", "alpha")
    missing = [key for key in required if key not in result]
    if missing:
        raise EvidenceError(f"Result artifact misses fields: {', '.join(missing)}")
    if result["prereg_version"] != "B4_KITCHEN_v0.3":
        raise EvidenceError("Only B4_KITCHEN_v0.3 may be imported by this adapter.")
    if result["dataset_id"] != manifest.get("dataset_id"):
        raise EvidenceError("Result and manifest refer to different datasets.")
    if "B4-Kitchen v0.3" not in prereg_text:
        raise EvidenceError("The supplied preregistration is not B4-Kitchen v0.3.")

    controls = result["controls"]
    main = result["main_test"]
    pvalue, alpha = float(main["pvalue"]), float(result["alpha"])
    positive_ok = float(controls["positive"]["pvalue"]) < alpha
    negative_ok = float(controls["negative"]["pvalue"]) >= alpha
    passed = positive_ok and negative_ok
    if bool(controls.get("passed")) != passed:
        raise EvidenceError("Declared control gate is inconsistent with individual control p-values.")
    expected = "INCONCLUSIVE" if not passed else ("SUPPORTED" if pvalue < alpha else "NOT_SUPPORTED")
    if result["verdict"] != expected:
        raise EvidenceError("Declared verdict is inconsistent with the frozen B4-Kitchen v0.3 rule.")

    return {
        "schema": "timdr-evidence/1",
        "artifact_kind": "imported_pre_registered_run",
        "epistemic_status": "IMPORTED_CLAIM_NOT_INDEPENDENTLY_REEXECUTED",
        "dataset_id": result["dataset_id"],
        "preregistration": {"source": str(prereg_path), "sha256": file_sha256(prereg_path)},
        "manifest": {
            "source": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "raw_file_hashes_declared": manifest.get("raw_file_hashes", {}),
            "raw_data_rehashed_by_this_runner": False,
        },
        "source_result": {"source": str(result_path), "sha256": file_sha256(result_path)},
        "method": result["method"],
        "criteria": {"alpha": alpha, "min_abs_effect_size": 0.0},
        "controls": {"positive_ok": positive_ok, "negative_ok": negative_ok, "source": controls},
        "test_evidence": {"p_value": pvalue, "effect_size": float(main["rho"]), "source": main},
        "declared_verdict": result["verdict"],
        "integrity_check": {"expected_verdict": expected, "passed": True},
        "warning": "This is an imported artifact. It is not an independent re-execution or new empirical confirmation.",
    }


def validate_report(report: Mapping[str, Any]) -> list[str]:
    """Self-audit the report contract; returns violations rather than a verdict.

    It validates protocol metadata only. It cannot establish the truth of the
    source experiment or replace independent reproduction of raw data.
    """
    errors: list[str] = []
    if report.get("schema") != "timdr-evidence/1":
        errors.append("Unknown evidence schema.")
    controls = report.get("controls")
    evidence = report.get("test_evidence")
    criteria = report.get("criteria")
    if not isinstance(controls, Mapping) or not isinstance(evidence, Mapping) or not isinstance(criteria, Mapping):
        return errors + ["Missing controls, test evidence, or criteria."]
    try:
        alpha, pvalue = float(criteria["alpha"]), float(evidence["p_value"])
    except (KeyError, TypeError, ValueError):
        return errors + ["Invalid alpha or p-value."]
    if not 0.0 < alpha < 1.0 or not 0.0 <= pvalue <= 1.0:
        errors.append("Alpha or p-value is outside its valid range.")
    passed = bool(controls.get("positive_ok")) and bool(controls.get("negative_ok"))
    expected = "INCONCLUSIVE" if not passed else ("SUPPORTED" if pvalue < alpha else "NOT_SUPPORTED")
    if report.get("declared_verdict") != expected:
        errors.append("Declared verdict conflicts with controls or pre-registered alpha.")
    for section in ("preregistration", "manifest", "source_result"):
        value = report.get(section)
        if not isinstance(value, Mapping) or not value.get("sha256"):
            errors.append(f"Missing provenance hash: {section}.")
    return errors


def write_immutable_report(report: dict[str, Any], destination: Path) -> Path:
    """Write once; an existing different report is an integrity failure."""
    errors = validate_report(report)
    if errors:
        raise EvidenceError("Invalid evidence report: " + " | ".join(errors))
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise EvidenceError(f"Refusing to overwrite a different evidence report: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    return destination
