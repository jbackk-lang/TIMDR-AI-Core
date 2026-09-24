"""Local Qwen 1.5B + LoRA explanation layer for Claim Graph decisions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "data" / "models" / "Qwen2.5-1.5B-Instruct"
RUNS = ROOT / "learning_runs" / "timdr_lora_cpu"
_model = None
_tokenizer = None


def _adapter() -> Path:
    candidates = sorted(RUNS.glob("*/adapter/adapter_config.json"))
    if not candidates:
        raise FileNotFoundError("Nie znaleziono adaptera LoRA TIMDR.")
    return candidates[-1].parent


def _load():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    _tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    _model = AutoModelForCausalLM.from_pretrained(
        str(BASE), local_files_only=True, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    _model = PeftModel.from_pretrained(_model, str(_adapter()), is_trainable=False)
    _model.eval()
    return _model, _tokenizer


def explain(question: str, verified_answer: str, verdict: str) -> str:
    """Generate a short Polish explanation without changing Claim Graph facts."""
    model, tokenizer = _load()
    prompt = (
        "Wyjaśnij krótko po polsku odpowiedź TIMDR. Użyj wyłącznie faktów poniżej. "
        "Nie zmieniaj statusu ani nie dodawaj nowych twierdzeń.\n"
        f"Pytanie: {question}\nStatus: {verdict}\n"
        f"Zweryfikowana odpowiedź: {verified_answer}"
    )
    import torch
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=96, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
