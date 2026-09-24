"""Download pinned trainable generative weights; no training or GPU allocation."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DESTINATION = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"
FILES = ["LICENSE", "README.md", "config.json", "generation_config.json",
         "merges.txt", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def main():
    from huggingface_hub import snapshot_download, HfApi
    info = HfApi(token=False).model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    expected = {item.rfilename: item for item in info.siblings if item.rfilename in FILES}
    if set(expected) != set(FILES):
        raise RuntimeError("Missing files in upstream model manifest")
    print(f"Downloading {MODEL_ID}; bytes={sum(item.size for item in expected.values())}", flush=True)
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(DESTINATION),
                      allow_patterns=FILES, max_workers=2, token=False)
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "training_performed": False,
                "source": f"https://huggingface.co/{MODEL_ID}", "files": {}}
    for name in FILES:
        path = DESTINATION / name
        if path.stat().st_size != expected[name].size:
            raise RuntimeError(f"Size mismatch: {name}")
        print(f"Verifying {name}...", flush=True)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        lfs = expected[name].lfs
        if lfs is not None and digest != lfs.sha256:
            raise RuntimeError(f"Upstream SHA-256 mismatch: {name}")
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": digest}
    # Validate the safetensors header without loading multi-GB weights into RAM.
    from safetensors import safe_open
    with safe_open(str(DESTINATION / "model.safetensors"), framework="numpy") as weights:
        manifest["tensor_count"] = len(list(weights.keys()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(DESTINATION), local_files_only=True,
                                              trust_remote_code=False)
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Wyjaśnij cztery gałęzie TIMDR."}],
        tokenize=True, add_generation_prompt=True)
    manifest["tokenizer_smoke_tokens"] = len(tokens)
    manifest["status"] = "DOWNLOADED_HASH_VERIFIED_TOKENIZER_CHECKED"
    manifest["generation_tested"] = False
    (DESTINATION / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"READY: {DESTINATION}")
    print("Weights verified. Training and generation were not started.")


if __name__ == "__main__":
    main()
