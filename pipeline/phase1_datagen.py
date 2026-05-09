# Phase1_fixed.py
# DATAGEN Phase 1 — Synthetic Data Generation
#
#   This script creates results/ and results/master_results.jsonl automatically.
#   Re-running APPENDS to master_results.jsonl (safe for multi-run experiments).
#   To start fresh: delete results/master_results.jsonl before running.
#
# ── CHANGELOG ─────────────────────────────────────────────────────────────────
# FIX 1 [CRITICAL] Label uniformity:
#   Original run_controlled() used a single hardcoded case ("65yo Male, T2N0M0")
#   for ALL controlled runs. Every valid record inherited T2/N0/M0, making the
#   Tier 3 Golden corpus label-uniform. The adapter then learned a degenerate prior.
#   Fix: replaced the single case with a 32-cell TNM grid (T1-T4 x N0-N3 x M0/M1).
#   run_controlled() now maps each ablation row round-robin across the grid,
#   producing a balanced staging distribution in the final corpus.
#
# FIX 2 [IMPORTANT] Post-generation diversity gate:
#   audit_corpus_diversity() runs after all generation is complete and reports
#   Shannon entropy per TNM dimension. Values below DIVERSITY_ENTROPY_FLOOR
#   trigger a FAIL warning so label collapse is caught before Phase 4 training.
#
# FIX 3 [MINOR] Exploratory runs now request a random stage per run:
#   Original exploratory prompt was static. Now each run explicitly requests
#   a randomly sampled T/N/M target, giving open-source models a stronger
#   signal to produce diverse outputs.
#
# FIX 4 [LOADER] Two-attempt 4-bit loader for GPU Optimization strategy analysis
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import json
import uuid
import gc
import random
import itertools
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from scipy.stats import entropy as shannon_entropy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

_HF_CACHE = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
os.environ["HF_HOME"]               = _HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_CACHE

from openai import OpenAI
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from huggingface_hub import login

# ── OUTPUT PATHS ──────────────────────────────────────────────────────────────
RESULTS_DIR       = "results"
RESULTS_FILE      = os.path.join(RESULTS_DIR, "master_results.jsonl")
CHECKPOINT_FILE   = os.path.join(RESULTS_DIR, "completed_runs.txt")

# ── GPU SANITY CHECK ──────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — DATAGEN SYNTHETIC DATA GENERATION")
print("=" * 60)
if torch.cuda.is_available():
    print(f"Visible GPU count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
              f"Free: {free/1024**3:.1f}GB / Total: {total/1024**3:.1f}GB")
else:
    print("CUDA not available — running on CPU.")
print(f"Results will be written to: {os.path.abspath(RESULTS_FILE)}")
print("=" * 60)

# ── AUTH ──────────────────────────────────────────────────────────────────────
HF_TOKEN       = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BIOPORTAL_KEY  = os.getenv("BIOPORTAL_API_KEY")

if HF_TOKEN:
    login(HF_TOKEN, add_to_git_credential=False)
    print("HuggingFace: authenticated.")
else:
    print("HuggingFace: HF_TOKEN not set — gated models will fail.")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not openai_client:
    print("OpenAI: OPENAI_API_KEY not set — GPT-4o runs will be skipped.")
if not BIOPORTAL_KEY:
    print("BioPortal: BIOPORTAL_API_KEY not set — SNOMED annotation will return [].")

# ── DIVERSITY GATE THRESHOLDS ─────────────────────────────────────────────────
DIVERSITY_ENTROPY_FLOOR = {
    "T": 0.80 * 1.386,   # 1.109
    "N": 0.80 * 1.386,   # 1.109
    "M": 0.80 * 0.693,   # 0.554
}

# ── TNM STAGING GRID (FIX 1) ─────────────────────────────────────────────────
T_VALUES = ["T1", "T2", "T3", "T4"]
N_VALUES = ["N0", "N1", "N2", "N3"]
M_VALUES = ["M0", "M1"]
TNM_GRID = list(itertools.product(T_VALUES, N_VALUES, M_VALUES))  # 32 cells


def infer_stage_group(t: str, n: str, m: str) -> str:
    """AJCC 8th Edition stage group lookup (simplified). M1 always -> Stage IV."""
    if m == "M1":
        return "IV"
    mapping = {
        ("T1", "N0"): "IA",   ("T2", "N0"): "IB",
        ("T1", "N1"): "IIA",  ("T2", "N1"): "IIB",
        ("T3", "N0"): "IIB",  ("T3", "N1"): "IIIA",
        ("T4", "N0"): "IIIA", ("T4", "N1"): "IIIA",
        ("T1", "N2"): "IIIA", ("T2", "N2"): "IIIA",
        ("T3", "N2"): "IIIB", ("T4", "N2"): "IIIB",
        ("T1", "N3"): "IIIB", ("T2", "N3"): "IIIB",
        ("T3", "N3"): "IIIC", ("T4", "N3"): "IIIC",
    }
    return mapping.get((t, n), "Unknown")


def build_controlled_case(t: str, n: str, m: str) -> str:
    """Builds a randomised natural-language seed case for the given TNM values."""
    size_map = {"1": "1.5 cm", "2": "3 cm",
                "3": "6 cm",  "4": "8 cm (chest wall invasion)"}
    n_desc = {
        "0": "no lymph node involvement",
        "1": "ipsilateral peribronchial nodes",
        "2": "ipsilateral mediastinal nodes",
        "3": "contralateral or supraclavicular nodes",
    }
    m_desc = {"0": "no distant metastasis", "1": "distant metastasis present (liver)"}
    age   = random.randint(45, 80)
    sex   = random.choice(["Male", "Female"])
    hist  = random.choice(["Adenocarcinoma", "Squamous Cell Carcinoma",
                            "Large Cell Carcinoma", "Small Cell Lung Cancer"])
    stage = infer_stage_group(t, n, m)
    return (
        f"Patient: {age}yo {sex}. Lung {hist}. "
        f"Primary tumor {size_map[t[1:]]} ({t}), "
        f"{n_desc[n[1:]]} ({n}), "
        f"{m_desc[m[1:]]} ({m}). "
        f"Clinical Stage {stage}."
    )


# ── SCHEMAS ───────────────────────────────────────────────────────────────────
BASE_SCHEMA = {
    "$schema_version": "rigid.v3",
    "notes": [{
        "staging":      {"prefix": "", "T": "", "N": "", "M": "", "stage_group": ""},
        "histology":    {"name": "", "icdo3": "", "snomed": ""},
        "molecular":    {"drivers": [{"gene": "", "result": "", "variant": None,
                                      "snomed_qualifier": ""}]},
        "demographics": {"age": "", "sex": "", "race/ethnicity": ""},
        "imaging":      [{"modality": "", "finding": "", "snomed": ""}],
        "treatment":    {"intent": "", "modalities": [{"type": "", "snomed": ""}]},
        "equity":       {"factors": [{"issue": "", "snomed": ""}]},
        "free_text":    "",
    }],
}

TIMELINE_SCHEMA = {
    "$schema_version": "longitudinal.v1",
    "patient_id": "SYNTH-001",
    "notes": [{
        "encounter_type":  "Initial / Follow-up / Relapse",
        "time_offset":     "Month X",
        "staging":         {"prefix": "", "T": "", "N": "", "M": "", "stage_group": ""},
        "clinical_status": "Progression / Response / Stable",
        "treatment":       {"intent": "", "modalities": [{"type": "", "snomed": ""}]},
        "free_text":       "",
    }],
}

SYSTEM_PROMPT = (
    "You are an expert oncologist. "
    "You must return ONLY a valid JSON object that strictly conforms to the provided JSON Schema. "
    "Do not include prose, markdown, or code fences. "
    "Use realistic values; avoid null and empty strings. "
    "Follow AJCC 8th Edition strictly."
)

SYSTEM_PROMPT_EXPLORATORY = (
    "You are an expert oncologist. Generate a realistic and plausible lung cancer case. "
    "You may use free text and standard clinical narrative. "
    "No need to strictly follow a schema."
)


# ── CHECKPOINTING ─────────────────────────────────────────────────────────────
def load_completed_run_ids() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_run_completed(run_key: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(run_key + "\n")


# ── RESULT LOGGING ────────────────────────────────────────────────────────────
def log_master_result(result_obj: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result_obj) + "\n")


# ── BIOPORTAL SNOMED ANNOTATION ───────────────────────────────────────────────
def bioportal_annotate_snomed(text: str) -> list:
    if not BIOPORTAL_KEY or not text.strip():
        return []
    url     = "https://data.bioontology.org/annotator"
    params  = {
        "text": text, "ontologies": "SNOMEDCT",
        "longest_only": "true", "exclude_numbers": "true", "whole_word_only": "true",
    }
    headers = {"Authorization": f"apikey token={BIOPORTAL_KEY}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        out = {}
        for ann in r.json():
            clz        = ann.get("annotatedClass", {})
            pref_label = clz.get("prefLabel", "Unknown")
            iri        = clz.get("@id", "")
            snomed_id  = iri.rsplit("/", 1)[-1] if iri else ""
            for loc in ann.get("annotations", []):
                matched = loc.get("text", "")
                key     = f"{snomed_id}_{matched}"
                if key not in out:
                    out[key] = {"snomed_id": snomed_id,
                                "prefLabel": pref_label,
                                "matched_text": matched}
        return list(out.values())
    except Exception as e:
        print(f"  [BioPortal] Error: {e}")
        return []


# ── UTILITY ───────────────────────────────────────────────────────────────────
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_last_json(text: str):
    stack, start, last = [], None, None
    in_string, escape  = False, False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if not stack:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    try:
                        last = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    start = None
    return last


def validate_longitudinal(json_obj: dict) -> list:
    if not json_obj or "notes" not in json_obj:
        return ["Invalid schema structure — 'notes' key missing."]
    errors = []
    for idx, note in enumerate(json_obj.get("notes", [])):
        stg     = note.get("staging", {})
        m_stage = str(stg.get("M", "")).upper()
        group   = str(stg.get("stage_group", "")).upper()
        if "M1" in m_stage and "IV" not in group:
            errors.append(
                f"Visit {idx}: M1 staging assigned but stage_group='{group}' "
                f"(expected Stage IV per AJCC 8th Edition)."
            )
    return errors


# ── STOPPING CRITERION ────────────────────────────────────────────────────────
class JsonStopOnBalancedClose(StoppingCriteria):
    def __init__(self, start_idx: int, tokenizer):
        self.start_idx = start_idx
        self.tok       = tokenizer
        self.started   = False

    def __call__(self, input_ids, scores, **kwargs):
        txt = self.tok.decode(input_ids[0][self.start_idx:], skip_special_tokens=True)
        if "{" in txt:
            self.started = True
        if not self.started:
            return False
        depth, in_string, escape = 0, False, False
        for ch in txt:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth <= 0:
                    return True
        return False


# ── GENERATION ROUTER ─────────────────────────────────────────────────────────
def generate_structured(
    model_id:    str,
    hf_artifacts,
    prompt:      str,
    params:      dict,
    layer:       str,
    prompt_type: str,
) -> dict:
    raw_output = ""
    seed       = params.get("seed", 42)
    set_global_seed(seed)
    active_sys = SYSTEM_PROMPT_EXPLORATORY if layer == "exploratory" else SYSTEM_PROMPT

    # ── OpenAI route ──────────────────────────────────────────────────────────
    if "gpt" in model_id.lower():
        if not openai_client:
            print(f"  [SKIP] {model_id}: OPENAI_API_KEY not set.")
            return {}
        try:
            gpt_kwargs = {
                "model":    model_id,
                "messages": [
                    {"role": "system", "content": active_sys},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": params.get("max_new_tokens", 1000),
                "seed":       seed,
            }
            if layer != "exploratory" and params.get("strict_json", True):
                gpt_kwargs["response_format"] = {"type": "json_object"}
            resp       = openai_client.chat.completions.create(**gpt_kwargs)
            raw_output = resp.choices[0].message.content
        except Exception as e:
            print(f"  [GPT Error] {e}")

    # ── HuggingFace route ─────────────────────────────────────────────────────
    else:
        model, tokenizer = hf_artifacts
        # next(model.parameters()).device is safe for both single- and multi-GPU
        # models (device_map="auto" spreads layers; model.device may be "meta").
        device = next(model.parameters()).device
        messages = [
            {"role": "system", "content": active_sys},
            {"role": "user",   "content": prompt},
        ]
        has_chat_template  = getattr(tokenizer, "chat_template", None) is not None
        want_template      = params.get("use_chat_template", True)
        effective_template = want_template and has_chat_template
        if want_template and not has_chat_template:
            print(f"  [INFO] {model_id}: no chat_template — plain-text fallback.")

        try:
            if effective_template:
                inputs    = tokenizer.apply_chat_template(
                    messages, return_dict=True, return_tensors="pt",
                    add_generation_prompt=True,
                ).to(device)
                input_ids = inputs["input_ids"]
                attn_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            else:
                full_text = (f"System: {active_sys}\n"
                             f"User: {prompt}\n"
                             f"Assistant: ")
                inputs    = tokenizer(full_text, return_tensors="pt").to(device)
                input_ids = inputs.input_ids
                attn_mask = inputs.attention_mask

            use_stopper = (
                params.get("strict_json", True)
                if layer == "controlled"
                else (layer != "exploratory")
            )
            stops = (
                StoppingCriteriaList([JsonStopOnBalancedClose(input_ids.shape[1], tokenizer)])
                if use_stopper else None
            )
            hf_kwargs = {
                "input_ids":         input_ids,
                "attention_mask":    attn_mask,
                "pad_token_id":      tokenizer.eos_token_id,
                "stopping_criteria": stops,
                "max_new_tokens":    params.get("max_new_tokens", 512),
            }
            for k in ("temperature", "top_p", "top_k", "do_sample"):
                if k in params:
                    hf_kwargs[k] = params[k]

            with torch.inference_mode():
                out_ids = model.generate(**hf_kwargs)
            raw_output = tokenizer.decode(
                out_ids[0][input_ids.shape[1]:], skip_special_tokens=True
            )
        except Exception as e:
            print(f"  [HF Gen Error] {e}")
        finally:
            # Free tensors immediately to prevent VRAM fragmentation
            for _name in ("inputs", "input_ids", "attn_mask", "out_ids"):
                try:
                    del locals()[_name]
                except KeyError:
                    pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    # ── Post-processing ───────────────────────────────────────────────────────
    parsed_json, is_valid = None, False
    try:
        parsed_json = json.loads(raw_output)
        is_valid    = True
    except json.JSONDecodeError:
        parsed_json = extract_last_json(raw_output)
        if parsed_json is not None:
            is_valid = True

    snomed_hits = bioportal_annotate_snomed(raw_output)
    val_errors  = (
        validate_longitudinal(parsed_json)
        if layer == "longitudinal" and is_valid
        else []
    )

    result_obj = {
        "run_id":              str(uuid.uuid4()),
        "timestamp":           datetime.now().isoformat(),
        "layer":               layer,
        "model":               model_id,
        "prompt_type":         prompt_type,
        "params":              params,
        "raw_output":          raw_output,
        "parsed_json_valid":   is_valid,
        "parsed_json":         parsed_json,
        "validation_errors":   val_errors,
        "snomed_codes":        snomed_hits,
        "snomed_source_layer": layer,
    }
    log_master_result(result_obj)
    return result_obj


# ── LAYER RUNNERS ─────────────────────────────────────────────────────────────
def run_exploratory(model_id: str, hf_artifacts, num_runs: int = 5,
                    completed: set = None):
    """FIX 3: Each run requests a randomly sampled T/N/M target."""
    if completed is None:
        completed = set()
    print(f"\n--- Layer 1: Exploratory ({model_id}, {num_runs} runs) ---")
    for i in range(num_runs):
        run_key = f"{model_id}::exploratory::{i}"
        if run_key in completed:
            print(f"  [SKIP] Run {i+1}/{num_runs} already completed.")
            continue
        t = random.choice(T_VALUES)
        n = random.choice(N_VALUES)
        m = random.choice(M_VALUES)
        prompt = (
            f"SCHEMA:\n{json.dumps(BASE_SCHEMA, indent=2)}\n\n"
            f"Generate 1 diverse synthetic lung cancer case at clinical stage {t}/{n}/{m}. "
            "Vary the patient demographics, histology subtype, molecular drivers, "
            "imaging findings, and treatment approach. "
            "Avoid defaulting to the most common presentation."
        )
        print(f"  Run {i+1}/{num_runs} — target: {t}/{n}/{m}")
        generate_structured(
            model_id, hf_artifacts, prompt,
            {"seed": 42 + i},
            "exploratory", "unrestricted",
        )
        mark_run_completed(run_key)


def run_controlled(model_id: str, hf_artifacts, csv_path: str,
                   completed: set = None):
    """FIX 1: Each ablation row gets a different TNM cell via round-robin."""
    if completed is None:
        completed = set()
    print(f"\n--- Layer 2: Controlled ({model_id}) — {csv_path} ---")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} ablation rows from {csv_path}")

    for idx, row in df.iterrows():
        run_key = f"{model_id}::controlled::{os.path.basename(csv_path)}::{idx}"
        if run_key in completed:
            print(f"  [SKIP] Row {idx+1} already completed.")
            continue
        t, n, m   = TNM_GRID[idx % len(TNM_GRID)]
        base_case = build_controlled_case(t, n, m)
        prompt    = (
            f"SCHEMA:\n{json.dumps(BASE_SCHEMA, indent=2)}\n\n"
            f"CASE:\n{base_case}\n\n"
            "Map this case to the schema exactly."
        )

        use_chat_template = str(row.get("use_chat_template", "true")).lower() in ("true", "1", "yes")
        strict_json       = str(row.get("strict_json",       "true")).lower() in ("true", "1", "yes")

        params = {
            "max_new_tokens":    int(row.get("max_length",   1000)),
            "temperature":       float(row.get("temperature",  0.7)),
            "top_p":             float(row.get("top_p",         0.9)),
            "top_k":             int(row.get("top_k",           50)),
            "do_sample":         str(row.get("do_sample", "true")).lower() in ("true", "1", "yes"),
            "use_chat_template": use_chat_template,
            "strict_json":       strict_json,
            "seed":              42,
            "intended_T":        t,
            "intended_N":        n,
            "intended_M":        m,
        }

        print(
            f"  Row {idx+1:>3}/{len(df)} | Target: {t}/{n}/{m} | "
            f"Temp={params['temperature']} | "
            f"ChatTpl={use_chat_template} | Strict={strict_json}"
        )
        generate_structured(
            model_id, hf_artifacts, prompt, params,
            "controlled", "zero-shot-anchored",
        )
        mark_run_completed(run_key)


def run_longitudinal(model_id: str, hf_artifacts, num_runs: int = 5,
                     completed: set = None):
    """Generates multi-encounter patient timelines with AJCC-compliant progression."""
    if completed is None:
        completed = set()
    print(f"\n--- Layer 3: Longitudinal ({model_id}, {num_runs} runs) ---")
    prompt = (
        f"SCHEMA:\n{json.dumps(TIMELINE_SCHEMA, indent=2)}\n\n"
        "Generate a longitudinal timeline for ONE synthetic lung cancer patient. "
        "Return exactly 3 items in the 'notes' array in chronological order:\n"
        "  1) Initial Diagnosis (Month 0)\n"
        "  2) Post-Treatment Follow-up (Month 6)\n"
        "  3) Disease Progression or Surveillance (Month 12)\n"
        "Ensure TNM values and stage_group are internally consistent at every visit."
    )
    for i in range(num_runs):
        run_key = f"{model_id}::longitudinal::{i}"
        if run_key in completed:
            print(f"  [SKIP] Run {i+1}/{num_runs} already completed.")
            continue
        print(f"  Run {i+1}/{num_runs}")
        generate_structured(
            model_id, hf_artifacts, prompt,
            {"temperature": 0.4, "do_sample": True, "seed": 100 + i},
            "longitudinal", "timeline-3-visits",
        )
        mark_run_completed(run_key)


# ── DIVERSITY AUDIT (FIX 2) ───────────────────────────────────────────────────
def _normalise_tnm(value: str, prefix: str) -> str:
    import re as _re
    v = str(value).strip().upper()
    if not v or v in ("UNKNOWN", "NONE", "NULL", ""):
        return "UNKNOWN"
    if v.startswith(prefix.upper()):
        v = v[len(prefix):]
    m = _re.match(r"^(\d+|IS|X|SN)", v)
    if m:
        return prefix.upper() + m.group(1)
    return prefix.upper() + v


def audit_corpus_diversity(jsonl_path: str = RESULTS_FILE):
    print("\n" + "=" * 60)
    print("POST-GENERATION DIVERSITY AUDIT")
    print("=" * 60)

    if not os.path.exists(jsonl_path):
        print(f"[WARN] {jsonl_path} not found — skipping audit.")
        return

    T_labels, N_labels, M_labels = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("parsed_json_valid"):
                continue
            if row.get("layer") not in ("controlled", "longitudinal"):
                continue
            pj    = row.get("parsed_json", {})
            notes = pj.get("notes", [])
            if not notes:
                continue
            stg = notes[0].get("staging", {})
            T_labels.append(_normalise_tnm(stg.get("T", "Unknown"), "T"))
            N_labels.append(_normalise_tnm(stg.get("N", "Unknown"), "N"))
            M_labels.append(_normalise_tnm(stg.get("M", "Unknown"), "M"))

    if not T_labels:
        print("[WARN] No valid controlled/longitudinal records found.")
        print("       Run Phase 1 generation before auditing.")
        return

    print(f"Valid Golden records audited: {len(T_labels)}\n")
    all_pass = True
    for dim, labels, key in [("T", T_labels, "T"), ("N", N_labels, "N"), ("M", M_labels, "M")]:
        counts = pd.Series(labels).value_counts()
        ent    = shannon_entropy(counts) if len(counts) > 1 else 0.0
        floor  = DIVERSITY_ENTROPY_FLOOR[key]
        status = "✓  PASS" if ent >= floor else "✗  FAIL — label collapse risk"
        if ent < floor:
            all_pass = False
        print(f"  [{dim}]  Entropy: {ent:.3f}  (floor: {floor:.3f})  {status}")
        print(f"        Distribution: {counts.to_dict()}")

    print()
    expected = {"T": ["T1","T2","T3","T4"], "N": ["N0","N1","N2","N3"], "M": ["M0","M1"]}
    if all_pass:
        print("All TNM dimensions pass the diversity threshold.")
        print("Safe to proceed to Phase 2 -> Phase 3 -> Phase 4.")
    else:
        print("One or more dimensions failed the diversity threshold.")
        for dim, labels, key in [("T", T_labels, "T"), ("N", N_labels, "N"), ("M", M_labels, "M")]:
            counts = pd.Series(labels).value_counts()
            ent    = shannon_entropy(counts) if len(counts) > 1 else 0.0
            if ent < DIVERSITY_ENTROPY_FLOOR[key]:
                missing = [c for c in expected[key] if counts.get(c, 0) == 0]
                scarce  = [c for c in expected[key]
                           if 0 < counts.get(c, 0) < len(labels) / (len(expected[key]) * 2)]
                print(f"   [{dim}] Missing: {missing or 'none'}  |  Scarce: {scarce or 'none'}")
        print()
        print("   Actions:")
        print("   1. Delete results/master_results.jsonl and re-run Phase 1 for a full fresh run.")
        print("   2. OR target only missing classes: run GPT-4o or Llama-3.3 (not ClinicalCamel)")
        print("      ClinicalCamel ignores T1/T2 prompts on the plain-text path.")
        print("   3. Do NOT proceed to Phase 4 fine-tuning until all dimensions PASS.")
    print("=" * 60)


# ── MODEL LOADER (FIX 4) ──────────────────────────────────────────────────────
def load_hf_model(model_name: str):
    """
    Loads a 70B model in 4-bit NF4 with a two-attempt strategy.

    Attempt 1 — single GPU (cuda:0).
        Works when one card has ~35GB free. Fast path, no inter-GPU communication.

    Attempt 2 — all visible GPUs via device_map='auto'.
        Triggered automatically on single-GPU OOM (e.g. ClinicalCamel on 48GB cards).
        Reserves 4GB headroom per GPU for activations and KV-cache.

    WHY 4-BIT NOT 8-BIT:
        The Transformers loader materialises each layer in bfloat16 inside a thread
        pool before BnB quantizes it. For 8-bit the bfloat16 peak per layer is
        ~twice that of 4-bit, pushing a 70B model past 48GB on a single GPU.
        4-bit keeps the peak well within budget and quality loss is negligible
        for clinical text generation.
    """
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Pre-flight VRAM report
    if n_gpus > 0:
        for dev in range(n_gpus):
            free_b, total_b = torch.cuda.mem_get_info(dev)
            print(f"  [Pre-load] GPU {dev}: {free_b/1024**3:.1f}GB free / "
                  f"{total_b/1024**3:.1f}GB total")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Defined once — shared by both attempts
    bnb_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # ── Attempt 1: single GPU ─────────────────────────────────────────────────
    try:
        print(f"  Attempt 1: loading {model_name} on cuda:0 (4-bit NF4)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_4bit,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if n_gpus > 0:
            free_b, total_b = torch.cuda.mem_get_info(0)
            print(f"  Loaded on GPU 0. Free: {free_b/1024**3:.1f}GB / "
                  f"{total_b/1024**3:.1f}GB")
        return model, tokenizer

    except torch.OutOfMemoryError as oom1:
        print(f"  [WARN] Single-GPU 4-bit OOM: {oom1}")
        print(f"  Clearing cache and retrying across {n_gpus} GPU(s)...")
        gc.collect()
        torch.cuda.empty_cache()

    # ── Attempt 2: all visible GPUs ───────────────────────────────────────────
    if n_gpus < 2:
        raise RuntimeError(
            "Single-GPU 4-bit load failed and only 1 GPU is visible. "
            "Free VRAM (nvidia-smi / kill -9 <PID>) then re-run."
        )

    # Reserve 4GB headroom per GPU for activations / KV-cache during inference
    mem_per_gpu = {
        i: f"{int(torch.cuda.mem_get_info(i)[0] / 1024**3) - 4}GiB"
        for i in range(n_gpus)
    }
    print(f"  Attempt 2: device_map='auto', max_memory={mem_per_gpu}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_4bit,
        device_map="auto",
        max_memory=mem_per_gpu,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    for dev in range(n_gpus):
        free_b, total_b = torch.cuda.mem_get_info(dev)
        print(f"  Loaded (multi-GPU). GPU {dev} free: {free_b/1024**3:.1f}GB / "
              f"{total_b/1024**3:.1f}GB")
    return model, tokenizer


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    MODELS = [
        #"gpt-4o",
        #"meta-llama/Llama-3.3-70B-Instruct",
        "wanglab/ClinicalCamel-70B",
    ]

    # ── Startup validation ────────────────────────────────────────────────────
    for csv_path in ["data/one_factor_at_a_time.csv", "data/full_factorial.csv"]:
        if not os.path.exists(csv_path):
            print(f"[WARN] Ablation CSV not found: {csv_path}  "
                  f"— controlled layer will be skipped for this file.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            existing = sum(1 for line in f if line.strip())
        print(f"Appending to existing {RESULTS_FILE} ({existing} records already logged).")
    else:
        print(f"Creating {RESULTS_FILE} — new run.")

    completed = load_completed_run_ids()
    if completed:
        print(f"Checkpoint loaded: {len(completed)} run(s) already done — will skip these.")

    for model_id in MODELS:
        print(f"\n{'='*60}\nMODEL: {model_id}\n{'='*60}")
        hf_artifacts = model = tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for _dev in range(torch.cuda.device_count()):
                _free, _total = torch.cuda.mem_get_info(_dev)
                print(f"[Pre-load]  GPU {_dev} free: {_free/1024**3:.1f}GB / "
                      f"{_total/1024**3:.1f}GB")

        try:
            if "gpt" not in model_id.lower():
                hf_artifacts     = load_hf_model(model_id)
                model, tokenizer = hf_artifacts

            run_exploratory(model_id, hf_artifacts, num_runs=5, completed=completed)
            run_controlled(model_id,  hf_artifacts, "data/one_factor_at_a_time.csv", completed=completed)
            run_controlled(model_id,  hf_artifacts, "data/full_factorial.csv",        completed=completed)
            run_longitudinal(model_id, hf_artifacts, num_runs=5, completed=completed)
            print(f"\n>> Finished {model_id} successfully.")

        except Exception as e:
            print(f"\n>> ERROR during {model_id}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if hf_artifacts: del hf_artifacts
            if model:        del model
            if tokenizer:    del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for _dev in range(torch.cuda.device_count()):
                    _free, _total = torch.cuda.mem_get_info(_dev)
                    print(f"[Post-clean] GPU {_dev} free: {_free/1024**3:.1f}GB / "
                          f"{_total/1024**3:.1f}GB")
            time.sleep(5)
            print(">> VRAM cleared. Ready for next model.")

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            total_records = sum(1 for line in f if line.strip())
        print(f"\nTotal records in {RESULTS_FILE}: {total_records}")

    audit_corpus_diversity()

    print("\nPhase 1 complete.")
    print(f"Results: {os.path.abspath(RESULTS_FILE)}")
    print("Next: run Phase2_fixed.py for statistical analysis.")


if __name__ == "__main__":
    main()
