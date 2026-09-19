"""Download one already-declared external research source into the local cache."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment_policy import EnvironmentPolicyError, download_declared_source

if len(sys.argv) != 3:
    raise SystemExit("Usage: python examples/download_external_evidence.py manifest.json source_id")
manifest_path, source_id = Path(sys.argv[1]), sys.argv[2]
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
source = next((item for item in payload.get("sources", []) if item.get("id") == source_id), None)
if source is None:
    raise SystemExit(f"No declared source with id {source_id!r}.")
try:
    saved = download_declared_source(source["url"], source["sha256"], ROOT / "external_cache" / source["filename"])
except (EnvironmentPolicyError, KeyError) as exc:
    raise SystemExit(f"Download rejected: {exc}") from exc
print(f"Verified external source saved locally: {saved}")
print("It is not training data until a separate preregistration authorizes it.")
