"""Inspect one authorized Paderborn training member without touching holdout."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paderborn_extractor import training_schema

print(training_schema(ROOT))
print("Holdout accessed: False")
