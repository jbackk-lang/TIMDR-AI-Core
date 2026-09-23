"""Freeze the Paderborn selection using archive member names only."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paderborn_prereg import PaderbornPreregError, write

try:
    plan = write(ROOT, ROOT / "prereg" / "PADERBORN_MS_REPLICATION_v0.1.json")
except PaderbornPreregError as exc:
    raise SystemExit(f"Cannot freeze Paderborn selection: {exc}") from exc
print(f"Frozen Paderborn selection: {plan['split']['counts']}")
print("Only archive member names were read; no waveform payload was opened.")
