"""Build a hashed, source-only Qwen corpus from canonical TIMDR documents.

This is data preparation, not an empirical TIMDR test and not a source of
instructions for the training runner.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIA = ROOT.parent / "GIA-TIMDR"
BASE = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"
CANONICAL_TRAIN = [
    "ARCHITEKTURA_TIMDR.md", "README.md",
    "docs/theory/TIMDR_Branch_Specification.md",
    "docs/theory/TIMDR_Chronoprocess.md",
    "docs/theory/TIMDR_EventGraph_Branch_Construction.md",
    "docs/theory/TIMDR_Signal_From_EventGraph.md",
    "docs/theory/Axioms_S_TIMDR_Signal.md",
    "docs/theory/Axioms_G_TIMDR_Geometry.md",
    "docs/theory/Axioms_K_TIMDR.md",
    "docs/theory/Axioms_META_TIMDR.md",
    "docs/theory/TIMDR_Twists.md",
]
CANONICAL_EVALUATION = [
    "docs/theory/TIMDR_CALIBRATION_FREEZE_RULE.md",
    "docs/theory/TIMDR_Geometry_From_EventGraph.md",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks(tokenizer, text, max_tokens):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    for start in range(0, len(ids), max_tokens):
        part = ids[start:start + max_tokens]
        if len(part) >= 32:
            yield tokenizer.decode(part, skip_special_tokens=True)


def rows_for(tokenizer, rel_paths, split, max_tokens):
    rows = []
    for relative in rel_paths:
        path = GIA / relative
        text = path.read_text(encoding="utf-8")
        for number, text_chunk in enumerate(chunks(tokenizer, text, max_tokens)):
            rows.append({"id": f"{split}-{path.stem}-{number:04d}", "source": relative,
                         "text": text_chunk})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "timdr_source_lora_v0.1.json")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if not 64 <= args.max_tokens <= 512:
        raise ValueError("max-tokens must be 64..512")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True, trust_remote_code=False)
    all_sources = CANONICAL_TRAIN + CANONICAL_EVALUATION
    missing = [relative for relative in all_sources if not (GIA / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source documents: {missing}")
    output = {"schema": "timdr-source-lora/0.1", "purpose": "Source-only continued pretraining corpus for local Qwen LoRA.",
              "max_tokens": args.max_tokens,
              "source_sha256": {relative: digest(GIA / relative) for relative in all_sources},
              "train": rows_for(tokenizer, CANONICAL_TRAIN, "train", args.max_tokens),
              "evaluation": rows_for(tokenizer, CANONICAL_EVALUATION, "evaluation", args.max_tokens)}
    if not output["train"] or not output["evaluation"]:
        raise RuntimeError("Both train and evaluation need chunks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "train_chunks": len(output["train"]),
                      "evaluation_chunks": len(output["evaluation"]), "sources": len(all_sources)}, indent=2))


if __name__ == "__main__":
    main()
