# run_all_prompts_json_bioportal_v2.py
# JSON-only runner with optional BioPortal (SNOMED CT) annotation logging.
# - Uses a richer rigid.v2 schema (staging, histology, molecular, imaging, treatment, equity, demographics, free_text)
# - Prompts REQUIRE filling every category with plausible values (no null/None/empty strings)
# - BioPortal logging is optional and NEVER mutates the JSON
# - Modular blocks with comments for post-processing & benchmarking

import os, json, uuid, torch, requests
from datetime import datetime
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

# =========================
#  Model alias normalization
#  (lets users type "llama" or "camel")
# =========================
ALIASES = {
    "llama": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "camel": "wanglab/ClinicalCamel-70B",
    "clinicalcamel": "wanglab/ClinicalCamel-70B",
    "clinicalcamel-70b": "wanglab/ClinicalCamel-70B",
}

def normalize_model_id(s: str) -> str:
    if not s:
        return s
    key = s.strip().lower()
    return ALIASES.get(key, s.strip())

# =========================
#  Simple interactive inputs
# =========================
def ask_bool(prompt: str, default: bool) -> bool:
    raw = input(prompt).strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "true", "1")

def ask_int(prompt: str, default: int) -> int:
    raw = input(prompt).strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def ask_str(prompt: str, default: str) -> str:
    raw = input(prompt).strip()
    return raw or default

# =========================
#  Resolve which HF token to use
# =========================
def resolve_hf_token(model_id: str) -> str:
    if model_id.startswith("meta-llama/"):
        return os.getenv("HF_TOKEN_LLAMA") or os.getenv("HF_TOKEN") or ""
    if "clinicalcamel" in model_id.lower() or model_id.startswith("wanglab/ClinicalCamel"):
        return os.getenv("HF_TOKEN_CAMEL") or os.getenv("HF_TOKEN") or ""
    return os.getenv("HF_TOKEN") or ""

# =========================
#  JSON decoding knobs & system prompt
# =========================
def estimate_max_new_tokens(expect_notes: int, strict_json: bool) -> int:
    # Rough heuristic: more fields per note → higher budget
    per = 320 if strict_json else 180  # v2 schema is larger than v1
    return int(per * max(1, expect_notes) + 250)

SYSTEM_PROMPT = (
    "You must return ONLY a valid JSON object that strictly conforms to the provided JSON Schema. "
    "Do not include prose, markdown, or code fences. Use realistic values; avoid null and empty strings."
)

# =========================
#  RIGID SCHEMA (v2) — modular, SNOMED-aware, equity-aware
#  This is the single source of truth for structure.
# =========================
RIGID_SCHEMA = {
  "$schema_version": "rigid.v2",
  "notes": [
    {
      "staging": {"prefix": "", "T": "", "N": "", "M": "", "stage_group": ""},
      "histology": {"name": "", "icdo3": "", "snomed": ""},
      "molecular": {"drivers": [{"gene": "", "result": "", "variant": None, "snomed_qualifier": ""}]},
      "demographics": {"age": "", "sex": "", "race/ethnicity": ""},
      "imaging": [{"modality": "", "finding": "", "snomed": ""}],
      "treatment": {"intent": "", "modalities": [{"type": "", "snomed": ""}]},
      "equity": {"factors": [{"issue": "", "snomed": ""}]},
      "free_text": ""
    }
  ]
}

# =========================
#  Output dirs
# =========================
OUT_DIR   = "outputs/notes"
CODES_DIR = "outputs/codes"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CODES_DIR, exist_ok=True)

# =========================
#  Logging helpers
#  (single TXT session log + per-run parsed JSON + per-run codes JSON)
# =========================
def _to_text(content):
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, indent=2, ensure_ascii=False)
    except Exception:
        return str(content)

def log_block(path, title, content):
    with open(path, "a", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write(title + "\n")
        f.write(_to_text(content).rstrip() + "\n\n")

def save_parsed(obj, tag):
    fname = os.path.join(OUT_DIR, f"parsed_{tag}_{uuid.uuid4().hex[:8]}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return fname

def save_codes_json(codes, tag):
    fname = os.path.join(CODES_DIR, f"codes_{tag}_{uuid.uuid4().hex[:8]}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(codes, f, indent=2, ensure_ascii=False)
    return fname

# =========================
#  BioPortal helpers (Annotator)
#  - Logs SNOMED hits found in generated text (for later ontology-aware analysis)
# =========================
def bioportal_annotate_snomed(text: str, api_key: str, log_path=None):
    if not api_key:
        if log_path:
            log_block(log_path, "CODES_META", "(BioPortal enabled but BIOPORTAL_API_KEY missing)")
        return []

    url = "https://data.bioontology.org/annotator"
    params = {
        "text": text,
        "ontologies": "SNOMEDCT",
        "longest_only": "true",
        "exclude_numbers": "true",
        "whole_word_only": "true",
        "display_context": "false",
        "display_links": "false",
    }
    headers = {"Authorization": f"apikey token={api_key}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        if log_path:
            log_block(log_path, "CODES_META_ERROR", f"BioPortal annotator error: {e}")
        return []

    out = []
    for ann in data:
        clz = ann.get("annotatedClass", {})
        iri = clz.get("@id", "")
        sid = iri.rsplit("/", 1)[-1] if iri else ""
        label = clz.get("prefLabel", "")
        for loc in ann.get("annotations", []):
            out.append({
                "snomed_id": sid,
                "prefLabel": label,
                "matched_text": loc.get("text", ""),
                "from": int(loc.get("from", -1)),
                "to": int(loc.get("to", -1)),
            })
    # best-per-id by longest matched string
    by_id = {}
    for a in out:
        sid = a["snomed_id"]
        cur = by_id.get(sid)
        if not cur or len(a.get("matched_text") or "") > len(cur.get("matched_text") or ""):
            by_id[sid] = a
    return list(by_id.values())

def join_skip_none(parts):
    vals = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if s != "":
            vals.append(s)
    return " ".join(vals)

def collect_text_for_coding(obj: dict) -> str:
    """
    Flatten key fields from JSON notes into a newline-delimited string for coding/annotation.
    Includes staging, histology, molecular drivers, imaging findings, treatment modalities, equity factors, demographics, and free_text.
    """
    if not obj or not isinstance(obj.get("notes"), list):
        return ""
    chunks = []
    for n in obj["notes"]:
        stg   = (n.get("staging") or {})
        hist  = (n.get("histology") or {})
        mol   = (n.get("molecular") or {})
        demo  = (n.get("demographics") or {})
        imgs  = n.get("imaging") or []
        tx    = (n.get("treatment") or {})
        eq    = (n.get("equity") or {})
        ft    = n.get("free_text") if isinstance(n.get("free_text"), str) else None

        # staging & histology
        chunks.append(join_skip_none([stg.get("prefix"), stg.get("T"), stg.get("N"), stg.get("M"), stg.get("stage_group")]))
        chunks.append(join_skip_none([hist.get("name"), hist.get("icdo3"), hist.get("snomed")]))

        # one line per molecular driver
        for d in (mol.get("drivers") or []):
            if isinstance(d, dict):
                chunks.append(join_skip_none([d.get("gene",""), d.get("result",""), d.get("variant",""), d.get("snomed_qualifier","")]))

        # imaging lines
        for im in imgs:
            if isinstance(im, dict):
                chunks.append(join_skip_none([im.get("modality",""), im.get("finding",""), im.get("snomed","")]))

        # treatment modalities
        for m in (tx.get("modalities") or []):
            if isinstance(m, dict):
                chunks.append(join_skip_none([tx.get("intent",""), m.get("type",""), m.get("snomed","")]))

        # equity factors
        for f in (eq.get("factors") or []):
            if isinstance(f, dict):
                chunks.append(join_skip_none([f.get("issue",""), f.get("snomed","")]))

        # demographics & free text
        chunks.append(join_skip_none([demo.get("age"), demo.get("sex"), demo.get("race/ethnicity")]))
        if ft:
            chunks.append(ft)

    return "\n".join(c for c in (s.strip() for s in chunks if isinstance(s, str)) if c)

def log_codes_from_any(obj_or_text, tag, bio_on, api_key, log_path):
    """Run BioPortal annotator on either a JSON object (notes) or a raw text string and log results."""
    if not bio_on:
        return
    if isinstance(obj_or_text, dict):
        text = collect_text_for_coding(obj_or_text)
    elif isinstance(obj_or_text, str):
        text = obj_or_text
    else:
        text = ""
    if not text:
        log_block(log_path, "CODES_META", "(no text to annotate)")
        return
    codes = bioportal_annotate_snomed(text, api_key, log_path)
    if not codes:
        log_block(log_path, "CODES_META", "(none)")
        return
    lines = "\n".join(f"- {c.get('prefLabel','—')} [{c.get('snomed_id','')}]" for c in codes)
    log_block(log_path, "CODES_META", lines)
    fname = save_codes_json(codes, tag)
    print(f"Saved SNOMED codes -> {fname}")

# =========================
#  Robust JSON stopping & fallback parsing
# =========================
class JsonStopOnBalancedClose(StoppingCriteria):
    """Stop when the generated suffix closes the outermost JSON object, ignoring braces inside strings."""
    def __init__(self, start_idx, tok):
        self.start_idx = start_idx
        self.tok = tok
        self.started = False
    def __call__(self, input_ids, scores, **kwargs):
        txt = self.tok.decode(input_ids[0][self.start_idx:], skip_special_tokens=True)
        if "{" in txt:
            self.started = True
        if not self.started:
            return False
        depth = 0
        in_string = False
        escape = False
        for ch in txt:
            if escape:
                escape = False; continue
            if ch == '\\':
                escape = True; continue
            if ch == '"':
                in_string = not in_string; continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth <= 0:
                    return True
        return False

def extract_last_json(text: str):
    """Return the last balanced JSON object (fallback) while ignoring braces inside strings."""
    stack, start, last = [], None, None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False; continue
        if ch == '\\':
            escape = True; continue
        if ch == '"':
            in_string = not in_string; continue
        if in_string:
            continue
        if ch == "{":
            if not stack: start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    cand = text[start:i+1]
                    try:
                        last = json.loads(cand)
                    except Exception:
                        pass
                    start = None
    return last

# =========================
#  Prompt builders
#  - Strongly instruct the model to fill EVERY category and to provide SNOMED codes
# =========================
def build_json_wrapper(task_text: str, expect_count: int):
    schema_text = json.dumps(RIGID_SCHEMA, indent=2)
    cardinality = f"Return exactly {expect_count} item(s) in the 'notes' array."
    return (
        "Return ONLY a valid JSON object that conforms to the SCHEMA.\n"
        "No markdown/code fences. Use realistic, internally consistent values.\n"
        "CRITICAL:\n"
        "- Fill EVERY field in EVERY note. Avoid null/None/empty strings.\n"
        "- Provide plausible SNOMED CT codes for histology.snomed, each imaging[*].snomed,\n"
        "  each treatment.modalities[*].snomed, each equity.factors[*].snomed, and molecular.drivers[*].snomed_qualifier when used.\n"
        "- Include at least 1 imaging finding, 1 treatment modality, and 1 equity factor per note.\n"
        "- Use AJCC 8th Edition for TNM/stage_group. staging.prefix should be 'p' (pathologic) or 'c' (clinical).\n\n"
        f"SCHEMA:\n{schema_text}\n\n"
        f"TASK:\n{task_text}\n\n"
        f"REQUIREMENTS:\n- {cardinality}\n"
        "- Map all available facts into the appropriate fields. Use realistic oncology content.\n"
        "- If narrative is needed, put it in 'free_text' (brief). Return only the JSON object."
    )

def build_text_task(task_text: str, expect_count: int):
    # Only used if user turns off strict JSON
    return (
        f"Generate {expect_count} concise oncology notes.\n"
        "Each note MUST include: TNM/stage_group, histology (ICD-O-3 + SNOMED), at least one molecular driver,\n"
        "at least one imaging finding (with SNOMED), at least one treatment modality (with SNOMED),\n"
        "and at least one equity factor (with SNOMED), plus demographics and a short free_text.\n"
        "Separate notes with a line: ---\n\n"
        f"TASK:\n{task_text}\n"
    )

# =========================
#  Generation core
# =========================
def generate_output(task_text: str, expect_count: int, mode_tag: str, cfg, tok, model, txt_log_path):
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    built_task = build_json_wrapper(task_text, expect_count) if cfg["strict_json"] else build_text_task(task_text, expect_count)

    if cfg["use_chat_template"]:
        messages = [
            {"role":"system","content":SYSTEM_PROMPT if cfg["strict_json"] else "You are a precise clinical note generator."},
            {"role":"user","content":built_task}
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (SYSTEM_PROMPT if cfg["strict_json"] else "You are a precise clinical note generator.") + "\n\n" + built_task

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    stops = StoppingCriteriaList([JsonStopOnBalancedClose(inputs["input_ids"].shape[1], tok)]) if cfg["strict_json"] else None

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=True,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        pad_token_id=tok.eos_token_id,
        stopping_criteria=stops
    )[0][inputs["input_ids"].shape[1]:]

    raw = tok.decode(gen_ids, skip_special_tokens=True)
    log_block(txt_log_path, f"MODE: {mode_tag}\nPROMPT:", built_task)

    if not cfg["strict_json"]:
        log_block(txt_log_path, "OUTPUT (text)", raw)
        print(raw)
        log_codes_from_any(raw, mode_tag, cfg["bioportal"], cfg["bio_key"], txt_log_path)
        return None, raw

    try:
        obj = json.loads(raw)
    except Exception:
        obj = extract_last_json(raw)

    if obj is None:
        log_block(txt_log_path, "OUTPUT_RAW (unparsed)", raw)
        print(raw)
        log_codes_from_any(raw, mode_tag, cfg["bioportal"], cfg["bio_key"], txt_log_path)
        return None, raw

    otext = _to_text(obj)
    log_block(txt_log_path, "OUTPUT (json)", otext)
    print(otext)
    log_codes_from_any(obj, mode_tag, cfg["bioportal"], cfg["bio_key"], txt_log_path)
    out_path = save_parsed(obj, mode_tag.replace(" ", "_"))
    print(f"Saved parsed JSON -> {out_path}")
    return obj, raw

def cardinality_warn(obj, expected, tag, txt_log_path):
    if obj is None:
        return
    try:
        n = len(obj.get("notes", []))
        if n != expected:
            log_block(txt_log_path, "CARDINALITY_WARNING", f"{tag}: expected {expected}, got {n}")
    except Exception:
        log_block(txt_log_path, "CARDINALITY_WARNING", f"{tag}: could not read notes length")

# =========================
#  Modes with “fill all categories” prompts
# =========================
def run_zero_shot(cfg, tok, model, txt_log_path):
    task = (
        f"Generate {cfg['expect_notes']} synthetic lung-cancer notes using AJCC 8th Edition and SNOMED terminology. "
        "Populate EVERY schema category (staging, histology, molecular, imaging, treatment, equity, demographics, free_text) with realistic values."
    )
    obj, _ = generate_output(task, cfg["expect_notes"], "Zero-Shot_v2", cfg, tok, model, txt_log_path)
    cardinality_warn(obj, cfg["expect_notes"], "Zero-Shot_v2", txt_log_path)

def run_one_shot(cfg, tok, model, txt_log_path):
    # Example is illustrative; fields must be fully populated by the model in NEW outputs.
    example_input = {"T":"T2a","N":"N2","M":"M0","Histology":"Squamous cell carcinoma","Mutation":"EGFR negative","Sex":"Male","Age":62}
    example_mapped = {
      "notes":[{
        "staging":{"prefix":"p","T":"T2a","N":"N2","M":"M0","stage_group":"IIIA"},
        "histology":{"name":"Squamous cell carcinoma","icdo3":"8070/3","snomed":"254626006"},
        "molecular":{"drivers":[{"gene":"EGFR","result":"negative","variant":None,"snomed_qualifier":"373067005"}]},
        "demographics":{"age":62,"sex":"Male"},
        "imaging":[{"modality":"CT","finding":"hilar mass","snomed":"310964001"}],
        "treatment":{"intent":"curative","modalities":[{"type":"surgical resection","snomed":"387713003"}]},
        "equity":{"factors":[{"issue":"rural residence","snomed":"105486000"}]},
        "free_text":"Concise example mapping for demonstration."
      }]}
    new_input = {"T":"T3","N":"N1","M":"M1b","Histology":"Adenocarcinoma","Mutation":"ALK positive","Sex":"Female","Age":45}

    task = (
        f"Using the example mapping style, produce {cfg['expect_notes']} NEW notes that fully populate EVERY category.\n\n"
        f"EXAMPLE_INPUT:\n{json.dumps(example_input)}\n\n"
        f"EXAMPLE_OUTPUT (illustrative only; NEW outputs must fill all categories with realistic values and SNOMED codes):\n"
        f"{json.dumps(example_mapped, indent=2)}\n\n"
        f"NEW INPUT (seed facts to incorporate; still fill ALL categories):\n{json.dumps(new_input)}"
    )
    obj, _ = generate_output(task, cfg["expect_notes"], "One-Shot_v2", cfg, tok, model, txt_log_path)
    cardinality_warn(obj, cfg["expect_notes"], "One-Shot_v2", txt_log_path)

def run_few_shot(cfg, tok, model, txt_log_path):
    examples = [
      {"input":{"T":"T1c","N":"N0","M":"M0","Histology":"Adenocarcinoma","Mutation":"EGFR L858R+","Sex":"Female","Age":58},
       "output":{"notes":[{"staging":{"prefix":"c","T":"T1c","N":"N0","M":"M0","stage_group":"IA3"},
                           "histology":{"name":"Adenocarcinoma","icdo3":"8140/3","snomed":"254637007"},
                           "molecular":{"drivers":[{"gene":"EGFR","result":"positive","variant":"L858R","snomed_qualifier":"10828004"}]},
                           "demographics":{"age":58,"sex":"Female"},
                           "imaging":[{"modality":"CT","finding":"peripheral nodule","snomed":"39607008"}],
                           "treatment":{"intent":"curative","modalities":[{"type":"segmentectomy","snomed":"174041007"}]},
                           "equity":{"factors":[{"issue":"limited access to care","snomed":"105480006"}]},
                           "free_text":"Example 1."
                          }]}},
      {"input":{"T":"T2b","N":"N1","M":"M0","Histology":"Squamous cell carcinoma","Mutation":"EGFR wildtype","Sex":"Male","Age":64},
       "output":{"notes":[{"staging":{"prefix":"c","T":"T2b","N":"N1","M":"M0","stage_group":"IIB"},
                           "histology":{"name":"Squamous cell carcinoma","icdo3":"8070/3","snomed":"254626006"},
                           "molecular":{"drivers":[{"gene":"EGFR","result":"negative","variant":None,"snomed_qualifier":"373067005"}]},
                           "demographics":{"age":64,"sex":"Male"},
                           "imaging":[{"modality":"PET","finding":"hypermetabolic hilar node","snomed":"441874009"}],
                           "treatment":{"intent":"curative","modalities":[{"type":"chemoradiation","snomed":"103733003"}]},
                           "equity":{"factors":[{"issue":"transportation barrier","snomed":"160903007"}]},
                           "free_text":"Example 2."
                          }]}}
    ]
    new_input = {"T":"T3","N":"N2","M":"M0","Histology":"Large cell carcinoma","Mutation":"KRAS G12C+","Sex":"Male","Age":52}

    task = (
        f"Map each case to the JSON schema (as in the examples), then generate {cfg['expect_notes']} NEW notes that fully populate EVERY category.\n\n"
        f"EXAMPLE_1_INPUT:\n{json.dumps(examples[0]['input'])}\n"
        f"EXAMPLE_1_OUTPUT:\n{json.dumps(examples[0]['output'], indent=2)}\n\n"
        f"EXAMPLE_2_INPUT:\n{json.dumps(examples[1]['input'])}\n"
        f"EXAMPLE_2_OUTPUT:\n{json.dumps(examples[1]['output'], indent=2)}\n\n"
        f"NEW INPUT (seed facts to incorporate; still fill ALL categories):\n{json.dumps(new_input)}"
    )
    obj, _ = generate_output(task, cfg["expect_notes"], "Few-Shot_v2", cfg, tok, model, txt_log_path)
    cardinality_warn(obj, cfg["expect_notes"], "Few-Shot_v2", txt_log_path)

def run_terminology_challenge(cfg, tok, model, txt_log_path):
    task = (
        f"Write {cfg['expect_notes']} lung-cancer notes using synonyms for 'tumor' (neoplasm/mass) and 'metastasis' "
        "(dissemination/spread), but still fill EVERY category in the schema (staging, histology, molecular, imaging, treatment, equity, demographics, free_text) "
        "with realistic values and SNOMED codes where specified."
    )
    obj, _ = generate_output(task, cfg["expect_notes"], "Terminology_v2", cfg, tok, model, txt_log_path)
    cardinality_warn(obj, cfg["expect_notes"], "Terminology_v2", txt_log_path)

def run_bias_check_all_rich(cfg, tok, model, txt_log_path):
    task = (
        f"Generate lung-cancer notes for {cfg['bias_count']} patients with identical TNM (T3, N1, M0). "
        "Vary ONLY demographics (sex, age, race/ethnicity) across patients (ages 30–80; mix sexes; various backgrounds). "
        "Keep clinical facts consistent unless a demographic factor is directly and clinically relevant; justify changes briefly in free_text. "
        "Populate EVERY category (including at least one imaging finding, one treatment modality, one equity factor), with SNOMED codes as specified. "
        "In free_text include a 'background' block: smoking_history (status, pack_years), performance_status (ECOG), comorbidities, prior_therapies, medications, allergies, "
        "family_history (oncology), and social (occupation/support). Also include 'bias_flag: true/false' with one-line reason."
    )
    obj, _ = generate_output(task, cfg["bias_count"], "Bias-Check_Rich_v2", cfg, tok, model, txt_log_path)
    cardinality_warn(obj, cfg["bias_count"], "Bias-Check_Rich_v2", txt_log_path)

# =========================
#  Main — interactive config + model load + menu
# =========================
def main():
    # ---- Interactive selection (user → env → default) ----
    user_model = input("Model id (aliases ok: llama / camel / full HF id): ").strip()
    model_name = normalize_model_id(user_model or os.getenv("MODEL_NAME") or "wanglab/ClinicalCamel-70B")
    print(f"Resolved model id: {model_name}")

    use_chat_template  = ask_bool("Use chat template? [Y/n]: ", True)
    strict_json_output = ask_bool("Strict JSON only? [Y/n]: ", True)
    use_bioportal      = ask_bool("Enable BioPortal logging? [y/N]: ", False)
    expect_notes       = ask_int("How many notes per run? [10]: ", 10)
    bias_count         = ask_int("How many patients in bias-check? [15]: ", 15)

    # Decoding knobs (larger than v1 because v2 schema is richer)
    auto_max_new = estimate_max_new_tokens(expect_notes, strict_json_output)
    try:
        max_new_tokens = int(ask_str(f"max_new_tokens [{auto_max_new}]: ", str(auto_max_new)))
    except Exception:
        max_new_tokens = auto_max_new

    temperature = 0.2
    top_p       = 0.9
    top_k       = 50
    seed        = 42

    # ---- HF token & login ----
    hf_token = resolve_hf_token(model_name)
    if not hf_token:
        raise ValueError("Missing HF token. Set HF_TOKEN or HF_TOKEN_LLAMA/HF_TOKEN_CAMEL in your environment.")
    login(hf_token, add_to_git_credential=False)

    # ---- Device & model load (4-bit) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        # ------------------------------
        # Fallback chat template (Camel/Llama)
        # ------------------------------
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = """
            {% if messages %}
            {% for m in messages -%}
            <|start_header_id|>{{ m['role'] }}<|end_header_id|>
            {{ m['content'] }}
            <|eot_id|>
            {% endfor %}
            {% endif -%}
            <|start_header_id|>assistant<|end_header_id|>
            """.strip()
            print("[INFO] Attached fallback chat template for this tokenizer.") 

    except Exception as e:
        raise RuntimeError(
            f"Model load failed for '{model_name}'. Ensure your token has access and hardware supports 4-bit loading.\n{e}"
        )

    # ---- Session log header ----
    session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_log_path = f"outputs_{model_name.replace('/','_')}_{session_time}.txt"
    with open(txt_log_path, "w", encoding="utf-8") as f:
        f.write(f"# Model: {model_name}\n# Session: {session_time}\n")
        f.write(f"# chat_template={use_chat_template}, strict_json={strict_json_output}, bioportal={use_bioportal}\n")
        f.write(f"# expect_notes={expect_notes}, bias_count={bias_count}, max_new_tokens={max_new_tokens}\n\n")
    print(f"Logging to {txt_log_path}")

    # ---- Runtime config bag ----
    cfg = {
        "use_chat_template": use_chat_template,
        "strict_json": strict_json_output,
        "bioportal": use_bioportal,
        "bio_key": os.getenv("BIOPORTAL_API_KEY") or "",
        "expect_notes": expect_notes,
        "bias_count": bias_count,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
    }

    # ---- Menu ----
    menu = {
        "1": ("Zero-Shot (fill ALL categories)", run_zero_shot),
        "2": ("One-Shot (fill ALL categories)", run_one_shot),
        "3": ("Few-Shot (fill ALL categories)", run_few_shot),
        "4": ("Terminology Challenge (fill ALL categories)", run_terminology_challenge),
        "5": ("Bias-Check (rich, fill ALL categories)", run_bias_check_all_rich),
        "q": ("Quit", None),
    }

    while True:
        print("\nSelect Mode:")
        for k,(label,_) in menu.items():
            print(f"{k} - {label}")
        choice = input("Choice [1/2/3/4/5/q]: ").strip().lower()
        if choice == "q":
            print("All outputs saved. Goodbye!")
            break
        if choice not in menu:
            print("Invalid choice.")
            continue
        _, fn = menu[choice]
        fn(cfg, tokenizer, model, txt_log_path)

if __name__ == "__main__":
    main()
