"""
core/generation.py — Model loading and text generation.
Supports both HuggingFace (4-bit NF4) and OpenAI GPT-4o routes.
"""

import gc
import json
import os

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList,
)

from core.schemas import SYSTEM_PROMPT


# ── JSON stopping criterion ───────────────────────────────────────────────────

class JsonStop(StoppingCriteria):
    def __init__(self, start_idx: int, tokenizer):
        self.start_idx = start_idx
        self.tok       = tokenizer
        self._started  = False

    def __call__(self, input_ids, scores, **kwargs):
        txt = self.tok.decode(input_ids[0][self.start_idx:], skip_special_tokens=True)
        if "{" in txt:
            self._started = True
        if not self._started:
            return False
        depth = in_str = escape = 0, False, False
        depth  = 0
        in_str = False
        escape = False
        for ch in txt:
            if escape:     escape = False; continue
            if ch == "\\": escape = True;  continue
            if ch == '"':  in_str = not in_str; continue
            if in_str:     continue
            if ch == "{":  depth += 1
            elif ch == "}":
                depth -= 1
                if depth <= 0:
                    return True
        return False


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_last_json(text: str):
    """Extracts the last complete JSON object from raw model output."""
    stack, start, last = [], None, None
    in_str = escape = False
    for i, ch in enumerate(text):
        if escape:     escape = False; continue
        if ch == "\\": escape = True;  continue
        if ch == '"':  in_str = not in_str; continue
        if in_str:     continue
        if ch == "{":
            if not stack: start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    try:    last = json.loads(text[start:i + 1])
                    except: pass
                    start = None
    return last


def parse_output(raw: str):
    """Try direct parse, fall back to extraction. Returns (parsed, is_valid)."""
    try:
        parsed = json.loads(raw)
        return parsed, True
    except json.JSONDecodeError:
        parsed = extract_last_json(raw)
        return parsed, parsed is not None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_hf_model(model_id: str):
    """
    Loads a model in 4-bit NF4. Two-attempt strategy:
    Attempt 1 — single GPU (cuda:0)
    Attempt 2 — auto device_map across all visible GPUs (on OOM)
    """
    from config import HF_TOKEN
    if HF_TOKEN:
        from huggingface_hub import login
        login(HF_TOKEN, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if n_gpus > 0:
        for dev in range(n_gpus):
            free_b, total_b = torch.cuda.mem_get_info(dev)
            print(f"  [GPU {dev}] {free_b/1024**3:.1f}GB free / {total_b/1024**3:.1f}GB total")

    try:
        print(f"  Loading {model_id} on cuda:0 (4-bit NF4)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            device_map={"": "cuda:0"}, trust_remote_code=True, low_cpu_mem_usage=True,
        )
        return model, tokenizer
    except torch.OutOfMemoryError as e:
        print(f"  [WARN] Single-GPU OOM: {e}")
        if n_gpus < 2:
            raise RuntimeError("OOM on single GPU; no second GPU available.")
        gc.collect(); torch.cuda.empty_cache()
        mem = {i: f"{int(torch.cuda.mem_get_info(i)[0]/1024**3)-4}GiB" for i in range(n_gpus)}
        print(f"  Retrying across {n_gpus} GPUs: max_memory={mem}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto",
            max_memory=mem, trust_remote_code=True, low_cpu_mem_usage=True,
        )
        return model, tokenizer


def unload_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ── Generation ────────────────────────────────────────────────────────────────

def generate_one(model, tokenizer, prompt: str, params: dict,
                 rag_context: str = "",
                 use_json_stop: bool = True,
                 model_id: str = "") -> str:
    """
    Generates one record. Handles both GPT-4o and HuggingFace routes.
    rag_context: if non-empty, prepended to user message as retrieved evidence.
    """
    # GPT-4o route
    if "gpt" in model_id.lower():
        from config import OPENAI_API_KEY
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        user_msg = (f"[Retrieved Context]\n{rag_context}\n\n{prompt}"
                    if rag_context else prompt)
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=params.get("max_new_tokens", 1024),
                seed=params.get("seed", 42),
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  [GPT error] {e}")
            return ""

    # HuggingFace route
    device   = next(model.parameters()).device
    user_msg = (f"[Retrieved Context]\n{rag_context}\n\n{prompt}"
                if rag_context else prompt)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}]

    has_template = getattr(tokenizer, "chat_template", None) is not None
    try:
        if has_template:
            enc       = tokenizer.apply_chat_template(
                messages, return_dict=True, return_tensors="pt",
                add_generation_prompt=True).to(device)
            input_ids = enc["input_ids"]
            attn_mask = enc.get("attention_mask", torch.ones_like(input_ids))
        else:
            raw_text  = f"System: {SYSTEM_PROMPT}\nUser: {user_msg}\nAssistant: "
            enc       = tokenizer(raw_text, return_tensors="pt").to(device)
            input_ids, attn_mask = enc.input_ids, enc.attention_mask

        stops = (StoppingCriteriaList([JsonStop(input_ids.shape[1], tokenizer)])
                 if use_json_stop else None)

        hf_kw = dict(
            input_ids=input_ids, attention_mask=attn_mask,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stops,
            max_new_tokens=params.get("max_new_tokens", 1024),
        )
        for k in ("temperature", "top_p", "top_k", "do_sample"):
            if k in params:
                hf_kw[k] = params[k]

        with torch.inference_mode():
            out = model.generate(**hf_kw)
        raw_out = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

    except Exception as e:
        print(f"  [HF gen error] {e}")
        raw_out = ""
    finally:
        try:
            del input_ids, attn_mask, enc, out
        except NameError:
            pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return raw_out
