r"""Download a pinned multilingual encoder, then verify local CPU inference.

This is an installation smoke test, not a TIMDR accuracy experiment.
Run from the repository: .venv\Scripts\python.exe examples/prepare_text_encoder.py --download
Subsequent runs without --download use only local files.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_DIR = ROOT.parent / "data" / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
FILES = ["model.safetensors", "config.json", "config_sentence_transformers.json",
         "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model",
         "1_Pooling/config.json", "README.md"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download
        print(f"Downloading {MODEL_ID} at {REVISION}", flush=True)
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=str(MODEL_DIR),
                          allow_patterns=FILES, max_workers=2, token=False)
        manifest = {"model_id": MODEL_ID, "revision": REVISION,
                    "source": f"https://huggingface.co/{MODEL_ID}", "files": {}}
        for name in FILES:
            path = MODEL_DIR / name
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": checksum}
        (MODEL_DIR / "DOWNLOAD_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise SystemExit("Model is not installed; first run with --download.")

    # All model/tokenizer loading below is offline and uses safetensors weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("Loading CPU libraries...", flush=True)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    print("Loading local encoder weights...", flush=True)
    model = SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True,
                                trust_remote_code=False,
                                model_kwargs={"use_safetensors": True})
    texts = ["Dzisiaj w Warszawie pada deszcz.",
             "W Warszawie mamy dziś deszczową pogodę.",
             "Łożysko silnika wykazuje zwiększone drgania."]
    print("Encoding three Polish sentences...", flush=True)
    embeddings = model.encode(texts, batch_size=3, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)
    if embeddings.shape != (3, 384) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Unexpected or invalid embeddings: {embeddings.shape}")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise RuntimeError("Embedding normalization failed")
    report = {"status": "INSTALLATION_SMOKE_TEST_PASSED", "model_id": MODEL_ID,
              "revision": REVISION, "model_directory": str(MODEL_DIR),
              "device": "cpu", "torch_threads": 2, "shape": list(embeddings.shape),
              "texts": texts, "cosine_similarities": (embeddings @ embeddings.T).tolist(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ["torch", "numpy", "sentence-transformers", "transformers", "huggingface-hub"]},
              "note": "Technical smoke test only; no anomaly threshold, training or holdout evaluation."}
    destination = ROOT / "external_cache" / "text_encoder_smoke.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Offline CPU inference OK: {embeddings.shape}")
    print(f"Model: {MODEL_DIR}")
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()
