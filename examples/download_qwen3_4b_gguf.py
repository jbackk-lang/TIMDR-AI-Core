"""Download the official CPU-friendly Qwen3 4B GGUF file and verify it.

This script downloads weights only. It neither trains the model nor makes it
an authority for TIMDR verdicts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-4B-GGUF"
FILENAME = "Qwen3-4B-Q4_K_M.gguf"
DESTINATION = ROOT.parent / "data" / "models" / "Qwen3-4B-GGUF"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).model_info(MODEL_ID, files_metadata=True)
    upstream = next((item for item in info.siblings if item.rfilename == FILENAME), None)
    if upstream is None or upstream.lfs is None or not upstream.lfs.sha256:
        raise RuntimeError("The official upstream manifest has no verifiable Q4_K_M file.")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / FILENAME
    if target.exists() and target.stat().st_size == upstream.size and sha256(target) == upstream.lfs.sha256:
        print(f"Already verified: {target}")
    else:
        print(f"Downloading official {MODEL_ID}/{FILENAME}")
        print(f"Expected size: {upstream.size:,} bytes", flush=True)
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=FILENAME,
            local_dir=str(DESTINATION),
            token=False,
        )
        if not target.exists() or target.stat().st_size != upstream.size:
            raise RuntimeError("Downloaded file size does not match the official manifest.")
        digest = sha256(target)
        if digest != upstream.lfs.sha256:
            raise RuntimeError("SHA-256 mismatch: deleting the untrusted model file.")
        print("SHA-256 verified.")

    manifest = {
        "model_id": MODEL_ID,
        "file": FILENAME,
        "bytes": upstream.size,
        "sha256": upstream.lfs.sha256,
        "source": f"https://huggingface.co/{MODEL_ID}",
        "quantization": "Q4_K_M",
        "hardware_target": "CPU, 15.9 GB RAM class",
        "training_performed": False,
        "generation_tested": False,
        "timdr_role": "optional language component; never a TIMDR verdict authority",
        "status": "DOWNLOADED_HASH_VERIFIED_RUNTIME_NOT_INSTALLED",
    }
    (DESTINATION / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"READY: {target}")
    print("Weights only. A separate CPU runtime check is required before chat use.")


if __name__ == "__main__":
    main()
