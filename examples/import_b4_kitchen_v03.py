"""Import an existing B4-Kitchen v0.3 result without re-running it.

Usage: python examples/import_b4_kitchen_v03.py C:\\path\\to\\GIA-TIMDR
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evidence_runner import import_b4_kitchen_v03, validate_report, write_immutable_report  # noqa: E402


if len(sys.argv) != 2:
    raise SystemExit("Usage: python examples/import_b4_kitchen_v03.py C:\\path\\to\\GIA-TIMDR")

source = Path(sys.argv[1]).resolve() / "docs" / "geometry"
report = import_b4_kitchen_v03(
    source / "B4_KITCHEN_RESULT_v0.3.json",
    source / "PREREG_B4_KITCHEN_v0.3.md",
    source / "b4_kitchen_manifest_template.json",
)
destination = ROOT / "evidence" / "B4_KITCHEN_v0.3_import.json"
write_immutable_report(report, destination)
print(destination)
print(report["epistemic_status"])
print("Declared verdict:", report["declared_verdict"])
print("Self-audit:", "OK" if not validate_report(report) else validate_report(report))
