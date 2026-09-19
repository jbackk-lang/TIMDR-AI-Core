import json
from pathlib import Path

from evidence_runner import EvidenceError, import_b4_kitchen_v03, validate_report, write_immutable_report


def _write(path: Path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_b4_import_checks_declared_verdict_and_is_immutable(tmp_path):
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    prereg = tmp_path / "prereg.md"
    _write(result, {"dataset_id":"d", "prereg_version":"B4_KITCHEN_v0.3", "method":"m", "verdict":"SUPPORTED", "controls":{"passed":True}, "main_test":{"pvalue":0.01,"rho":0.2}, "alpha":0.05})
    _write(manifest, {"dataset_id":"d"})
    prereg.write_text("# B4-Kitchen v0.3", encoding="utf-8")
    report = import_b4_kitchen_v03(result, prereg, manifest)
    assert validate_report(report) == []
    destination = write_immutable_report(report, tmp_path / "evidence.json")
    assert destination.exists()
    assert report["epistemic_status"] == "IMPORTED_CLAIM_NOT_INDEPENDENTLY_REEXECUTED"
    report["method"] = "changed"
    try:
        write_immutable_report(report, destination)
    except EvidenceError:
        pass
    else:
        raise AssertionError("Expected immutable report protection")
