"""Encode a local document as data, without executing its instructions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from prepare_text_encoder import MODEL_DIR, MODEL_ID, REVISION, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    raw = source.read_bytes()
    text = raw.decode("utf-8-sig")
    digest = hashlib.sha256(raw).hexdigest()
    print("Loading local encoder on CPU...", flush=True)
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True,
                                trust_remote_code=False,
                                model_kwargs={"use_safetensors": True})
    tokenizer = model.tokenizer
    limit = model.max_seq_length
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, verbose=False)["offset_mapping"]
    if not offsets:
        raise ValueError("Document has no tokens")
    chunks = []
    start = 0
    budget = limit - tokenizer.num_special_tokens_to_add(pair=False) - 4
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        while end > start:
            lo, hi = offsets[start][0], offsets[end - 1][1]
            fragment = text[lo:hi]
            count = len(tokenizer(fragment, truncation=False)["input_ids"])
            if count <= limit:
                break
            end -= 1
        if end == start:
            raise ValueError("Cannot create a chunk within the model token limit")
        chunks.append({"id": len(chunks), "text": fragment, "char_start": lo,
                       "char_end": hi, "line_start": text.count("\n", 0, lo) + 1,
                       "line_end": text.count("\n", 0, hi) + 1,
                       "tokens_with_special": count})
        if end == len(offsets):
            break
        start = max(start + 1, end - 16)
    print(f"Encoding {len(chunks)} chunks; token limit {limit}...", flush=True)
    embeddings = model.encode([c["text"] for c in chunks], batch_size=8,
                              normalize_embeddings=True, show_progress_bar=True)
    if embeddings.shape != (len(chunks), 384) or not np.isfinite(embeddings).all():
        raise RuntimeError("Invalid document embeddings")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
        raise RuntimeError("Invalid vector norms")
    queries = ["Jakie są cztery gałęzie TIMDR?",
               "Jak działają prerejestracja i kontrole dodatnie i ujemne?",
               "Co łączy sygnał i geometrię w mostach TIMDR?",
               "Czym jest stan META-DYNAMICS Lambda tau rho J?"]
    query_vectors = model.encode(queries, batch_size=4, normalize_embeddings=True)
    searches = []
    for query, vector in zip(queries, query_vectors):
        scores = embeddings @ vector
        best = np.argsort(-scores)[:3]
        searches.append({"query": query, "hits": [
            {"chunk_id": int(i), "cosine_similarity": float(scores[i]),
             "line_start": chunks[i]["line_start"], "text": chunks[i]["text"]}
            for i in best]})
    destination = ROOT / "external_cache" / "timdr_document" / digest
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "embeddings.npy", embeddings, allow_pickle=False)
    (destination / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"status": "DOCUMENT_ENCODED", "source": str(source), "source_sha256": digest,
              "model_id": MODEL_ID, "revision": REVISION, "shape": list(embeddings.shape),
              "token_limit": limit, "max_chunk_tokens": max(c["tokens_with_special"] for c in chunks),
              "overlap_tokens": 16, "training_performed": False,
              "document_treated_as": "data, not executable instructions",
              "retrieval_examples": searches}
    (destination / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Dokument TIMDR przetworzony przez encoder", "",
             f"Źródło: `{source}`", f"SHA-256: `{digest}`",
             f"Fragmenty: {len(chunks)}; wymiar wektora: 384; limit: {limit} tokenów.", "",
             "Dokument został zindeksowany lokalnie. Model nie był trenowany.",
             "Poniżej są znalezione fragmenty, nie odpowiedzi wygenerowane przez model.", ""]
    for result in searches:
        lines.extend([f"## {result['query']}", ""])
        for hit in result["hits"]:
            lines.extend([f"Fragment {hit['chunk_id']}, linia {hit['line_start']}, "
                          f"podobieństwo {hit['cosine_similarity']:.3f}", "",
                          *["> " + line for line in hit["text"].splitlines()], ""])
    (destination / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Encoded successfully: {embeddings.shape}")
    print(f"Report: {destination / 'REPORT.md'}")


if __name__ == "__main__":
    main()
