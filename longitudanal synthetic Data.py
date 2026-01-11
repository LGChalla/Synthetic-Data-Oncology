
import os, json, uuid, torch, requests
from datetime import datetime
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

# =========================
#  Interactive defaults
# =========================
MODEL_ID = "wanglab/ClinicalCamel-70B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# HuggingFace login (must be set as environment variable before running)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please export it before running.")
login(hf_token)
USE_CHAT_TEMPLATE    = False
STRICT_JSON_OUTPUT   = True
USE_BIOPORTAL        = True
MAX_NEW_TOKENS       = 900

# For timeline mode: default anchors (user can override at launch)
DEFAULT_ANCHORS = [
  "t0:Diagnosis/Start",
  "t1:First-line therapy (chemo/IO/TKI)",
  "t2:Restaging scan (~3-4 mo)",
  "t3:Maintenance/Adjuvant",
  "t4:Surveillance/Follow-up"
]

# Decoding tuned for structured JSON reliability
TEMPERATURE    = 0.2
TOP_P          = 0.9
TOP_K          = 50
DO_SAMPLE      = True
SEED           = 42

# System texts
SYSTEM_TEXT_JSON = (
  "You must return ONLY a valid JSON object that strictly conforms to the provided JSON Schema. "
  "Do not include prose, markdown, or code fences. Use null when unknown."
)

# =========================
#  RIGID SCHEMAS
# =========================
# Notes schema (for the JSON prompt modes)
RIGID_SCHEMA = {
  "$schema_version": "rigid.v1",
  "notes": [
    {
      "staging": {"prefix": None, "T": "", "N": "", "M": "", "stage_group": ""},
      "histology": {"name": "", "icdo3": "", "snomed": None},
      "molecular": {"drivers": [{"gene": "", "result": "", "variant": None}]},
      "demographics": {"age": 0, "sex": ""},
      "free_text": None
    }
  ]
}

# Timeline schema (richer per-visit structured data)
TIMELINE_SCHEMA = {
  "$schema_version": "timeline.v2",
  "seed_case": {
    "staging": {"T": "", "N": "", "M": "", "stage_group": ""},
    "histology": {"name": "", "icdo3": "", "snomed": None},
    "molecular": {"drivers": [{"gene": "", "result": "", "variant": None}]},
    "demographics": {"age": 0, "sex": ""}
  },
  "anchors": ["t0:Diagnosis/Start", "t1:First-line therapy", "t2:Restaging scan"],
  "timeline": [
    {
      "date": "YYYY-MM",
      "visit_label": "t0:Diagnosis/Start",
      "staging": {"T": "", "N": "", "M": "", "stage_group": ""},
      "imaging": {"modality": "string or null", "findings": "string or null"},
      "therapies": [
        {
          "type": "chemo|io|tki|radiation|surgery|other",
          "regimen": "string or null",
          "intent": "curative|palliative|adjuvant|neoadjuvant|maintenance|null",
          "line": 1,
          "cycles": [
            {"cycle": 1, "date": "YYYY-MM", "dose_mod": "none|reduced|held|delayed|null"}
          ]
        }
      ],
      "adverse_events": [
        {"name": "string", "grade": 0, "action": "none|hold|reduce|discontinue|supportive", "outcome": "string or null"}
      ],
      "labs": [
        {"name": "string", "value": 0, "unit": "string", "trend": "up|down|stable|null"}
      ],
      "assessment": "string",
      "plan": "string"
    }
  ]
}

# =========================
#  Output dirs
# =========================
OUT_DIR   = "outputs/timeline"
CODES_DIR = "outputs/codes"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CODES_DIR, exist_ok=True)

# =========================
#  Logging helpers
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

def yesno(prompt, default=True):
  d = "Y/n" if default else "y/N"
  s = input(f"{prompt} [{d}]: ").strip().lower()
  if not s:
    return default
  return s in ("y", "yes", "true", "1")

def ask_int(prompt, default):
  s = input(f"{prompt} [{default}]: ").strip()
  if not s:
    return int(default)
  try:
    return int(s)
  except:
    print("Invalid int; using default.")
    return int(default)

def ask_str(prompt, default):
  s = input(f"{prompt} [{default}]: ").strip()
  return s or default

def estimate_tokens_for_timeline(n_visits: int) -> int:
  # Rough: ~260 tokens/visit with richer therapy/AE/labs + ~350 overhead
  return int(260 * max(1, n_visits) + 350)

# =========================
#  BioPortal helpers
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
  # best per id
  by_id = {}
  for a in out:
    sid = a["snomed_id"]
    cur = by_id.get(sid)
    if not cur or len(a.get("matched_text") or "") > len(cur.get("matched_text") or ""):
      by_id[sid] = a
  return list(by_id.values())

def collect_text_from_timeline(obj: dict) -> str:
  if not obj:
    return ""
  chunks = []
  seed = obj.get("seed_case") or {}
  for k in ("staging","histology","molecular","demographics"):
    chunks.append(_to_text(seed.get(k,"")))
  for v in obj.get("timeline", []):
    stg = v.get("staging", {})
    img = v.get("imaging", {})
    chunks.append(" ".join([str(stg.get("T","")), str(stg.get("N","")), str(stg.get("M","")), str(stg.get("stage_group",""))]))
    chunks.append(" ".join([str(img.get("modality","")), str(img.get("findings",""))]))
    for t in v.get("therapies", []) or []:
      chunks.append(" ".join([str(t.get("type","")), str(t.get("regimen","")), str(t.get("intent","")), f"line:{t.get('line','')}"]))
      for c in t.get("cycles", []) or []:
        chunks.append(" ".join([f"cycle {c.get('cycle','')}", str(c.get("date","")), str(c.get("dose_mod",""))]))
    for ae in v.get("adverse_events", []) or []:
      chunks.append(" ".join([str(ae.get("name","")), f"grade {ae.get('grade','')}", str(ae.get("action",""))]))
    for lb in v.get("labs", []) or []:
      chunks.append(" ".join([str(lb.get("name","")), str(lb.get("value","")), str(lb.get("unit","")), str(lb.get("trend",""))]))
    chunks.append(str(v.get("assessment","")))
    chunks.append(str(v.get("plan","")))
  return "\n".join([c for c in chunks if c and isinstance(c, str)])

def log_codes_from_timeline(obj_or_text, tag, bio_on, api_key, log_path):
  if not bio_on:
    return
  text = obj_or_text if isinstance(obj_or_text, str) else collect_text_from_timeline(obj_or_text)
  codes = bioportal_annotate_snomed(text, api_key, log_path)
  if not codes:
    log_block(log_path, "CODES_META", "(none)")
    return
  lines = [f"- {c.get('prefLabel','—')} [{c.get('snomed_id','')}]"
           for c in codes]
  log_block(log_path, "CODES_META", "\n".join(lines))
  fname = save_codes_json(codes, tag)
  print(f"Saved SNOMED codes -> {fname}")

# =========================
#  Robust JSON helpers
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
# =========================
def build_json_wrapper(task_text: str, expect_count: int):
  schema_text = json.dumps(RIGID_SCHEMA, indent=2)
  cardinality = f"Return exactly {expect_count} item(s) in the 'notes' array." if expect_count > 1 else "Return exactly 1 item in the 'notes' array."
  return (
      "Return ONLY a valid JSON object that conforms to the SCHEMA. "
      "No markdown or code fences. Use null when unknown.\n\n"
      f"SCHEMA:\n{schema_text}\n\n"
      f"TASK:\n{task_text}\n\n"
      f"REQUIREMENTS:\n- {cardinality}\n"
      "- Map available facts into TNM/stage, histology, drivers, demographics.\n"
      "- If you include narrative, place it in 'free_text'.\n"
      "Return only the JSON object."
  )

def build_timeline_prompt_no_seed(anchors):
  """
  anchors: list of strings like ["t0:Diagnosis/Start", "t1:First-line therapy (chemo/IO)", ...]
  """
  schema_txt = json.dumps(TIMELINE_SCHEMA, indent=2)
  anchors_txt = json.dumps(anchors, ensure_ascii=False)

  # IMPORTANT: not forced progression — capture meds/events too
  return (
    "Return ONLY a valid JSON object that conforms to the SCHEMA below. "
    "No markdown or code fences. Use null when unknown.\n\n"
    f"SCHEMA:\n{schema_txt}\n\n"
    "TASK:\n"
    "- First, create a plausible lung oncology 'seed_case' (AJCC-8 lung). Then, generate a longitudinal 'timeline' "
    "with one visit per requested ANCHOR (order preserved). DO NOT assume tumor progression between anchors unless clinically justified.\n"
    "- At each visit, capture therapies (chemo/IO/TKI/radiation/surgery), cycles/dose modifications, supportive care, adverse events (CTCAE-style names & grades), "
    "relevant labs (trend), imaging findings, assessment, and plan. Therapy 'line' and 'intent' should be coherent.\n"
    "- Use YYYY-MM month stamps; ensure internal consistency (e.g., M1* implies stage group IV*; surgery + pCR allowed without progression; maintenance may stabilize disease).\n\n"
    f"ANCHORS:\n{anchors_txt}\n\n"
    "REQUIREMENTS:\n"
    "- 'anchors' in the output must echo the provided anchors (order preserved).\n"
    "- 'timeline' must contain exactly one visit per anchor with matching 'visit_label'.\n"
    "- Return only the JSON object."
  )

# =========================
#  Generation core
# =========================
def generate_text_or_json(prompt_text: str, strict_json: bool, tokenizer, model, max_new_tokens: int, txt_log_path: str, mode_tag: str):
  torch.manual_seed(SEED)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

  if USE_CHAT_TEMPLATE:
    messages = [
      {"role":"system","content":SYSTEM_TEXT_JSON if strict_json else "You are a precise clinical generator."},
      {"role":"user","content":prompt_text}
    ]
    built = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  else:
    built = (SYSTEM_TEXT_JSON if strict_json else "You are a precise clinical generator.") + "\n\n" + prompt_text

  inputs = tokenizer(built, return_tensors="pt").to(model.device)
  stops = StoppingCriteriaList([JsonStopOnBalancedClose(inputs["input_ids"].shape[1], tokenizer)]) if strict_json else None

  gen_ids = model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    pad_token_id=tokenizer.eos_token_id,
    stopping_criteria=stops
  )[0][inputs["input_ids"].shape[1]:]

  raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
  log_block(txt_log_path, f"MODE: {mode_tag}\nPROMPT:", prompt_text)

  if not strict_json:
    log_block(txt_log_path, "OUTPUT (text)", raw)
    print(raw)
    return None, raw

  try:
    obj = json.loads(raw)
  except Exception:
    obj = extract_last_json(raw)

  if obj is None:
    log_block(txt_log_path, "OUTPUT_RAW (unparsed)", raw)
    print(raw)
    return None, raw

  otext = _to_text(obj)
  log_block(txt_log_path, "OUTPUT (json)", otext)
  print(otext)
  return obj, raw

# =========================
#  JSON MODES (notes)
# =========================
def run_zero_shot(cfg, tok, model, log_path):
  task = (f"Generate {cfg['expect_notes']} concise lung-cancer staging notes using AJCC 8th Edition and SNOMED terminology.")
  obj, _ = generate_text_or_json(build_json_wrapper(task, cfg['expect_notes']), True, tok, model, cfg["max_new_tokens"], log_path, "Zero-Shot")
  if cfg["bioportal"]: log_codes_from_timeline(obj, "zeroshot", True, cfg["bio_key"], log_path)

def run_one_shot(cfg, tok, model, log_path):
  example_input = {"T": "T2a", "N": "N2", "M": "M0", "Histology": "Squamous cell carcinoma", "Mutation": "EGFR negative", "Sex": "Male", "Age": 62}
  example_mapped = {
    "notes":[{
      "staging":{"prefix":None,"T":"T2a","N":"N2","M":"M0","stage_group":"IIIA"},
      "histology":{"name":"Squamous cell carcinoma","icdo3":"8070/3","snomed":None},
      "molecular":{"drivers":[{"gene":"EGFR","result":"negative","variant":None}]},
      "demographics":{"age":62,"sex":"Male"},
      "free_text":"Concise note mapped to fields."
    }]}
  new_input = {"T": "T3", "N": "N1", "M": "M1b", "Histology": "Adenocarcinoma", "Mutation": "ALK positive", "Sex": "Female", "Age": 45}

  task = (
    f"Using the example mapping style, produce {cfg['expect_notes']} JSON notes for NEW PATIENTS.\n\n"
    f"EXAMPLE_INPUT:\n{json.dumps(example_input)}\n\n"
    f"EXAMPLE_OUTPUT:\n{json.dumps(example_mapped, indent=2)}\n\n"
    f"NEW INPUT:\n{json.dumps(new_input)}"
  )
  obj, _ = generate_text_or_json(build_json_wrapper(task, cfg['expect_notes']), True, tok, model, cfg["max_new_tokens"], log_path, "One-Shot")
  if cfg["bioportal"]: log_codes_from_timeline(obj, "oneshot", True, cfg["bio_key"], log_path)

def run_few_shot(cfg, tok, model, log_path):
  examples = [
    {"input":{"T":"T1c","N":"N0","M":"M0","Histology":"Adenocarcinoma","Mutation":"EGFR L858R+","Sex":"Female","Age":58},
     "output":{"notes":[{"staging":{"prefix":None,"T":"T1c","N":"N0","M":"M0","stage_group":"IA3"},
                         "histology":{"name":"Adenocarcinoma","icdo3":"8140/3","snomed":None},
                         "molecular":{"drivers":[{"gene":"EGFR","result":"positive","variant":"L858R"}]},
                         "demographics":{"age":58,"sex":"Female"},"free_text":None}]}}
  ,
    {"input":{"T":"T2b","N":"N1","M":"M0","Histology":"Squamous cell carcinoma","Mutation":"EGFR wildtype","Sex":"Male","Age":64},
     "output":{"notes":[{"staging":{"prefix":None,"T":"T2b","N":"N1","M":"M0","stage_group":"IIB"},
                         "histology":{"name":"Squamous cell carcinoma","icdo3":"8070/3","snomed":None},
                         "molecular":{"drivers":[{"gene":"EGFR","result":"negative","variant":None}]},
                         "demographics":{"age":64,"sex":"Male"},"free_text":None}]}}
  ]
  new_input = {"T":"T3","N":"N2","M":"M0","Histology":"Large cell carcinoma","Mutation":"KRAS G12C+","Sex":"Male","Age":52}

  task = (
    f"Map each case to the JSON schema and then generate {cfg['expect_notes']} NEW notes.\n\n"
    f"EXAMPLE_1_INPUT:\n{json.dumps(examples[0]['input'])}\n"
    f"EXAMPLE_1_OUTPUT:\n{json.dumps(examples[0]['output'], indent=2)}\n\n"
    f"EXAMPLE_2_INPUT:\n{json.dumps(examples[1]['input'])}\n"
    f"EXAMPLE_2_OUTPUT:\n{json.dumps(examples[1]['output'], indent=2)}\n\n"
    f"NEW INPUT:\n{json.dumps(new_input)}"
  )
  obj, _ = generate_text_or_json(build_json_wrapper(task, cfg['expect_notes']), True, tok, model, cfg["max_new_tokens"], log_path, "Few-Shot")
  if cfg["bioportal"]: log_codes_from_timeline(obj, "fewshot", True, cfg["bio_key"], log_path)

def run_terminology_challenge(cfg, tok, model, log_path):
  task = (
    f"Write {cfg['expect_notes']} staging notes using synonyms for 'tumor' (neoplasm/mass) and 'metastasis' (dissemination/spread), "
    "mapped to (TNM, stage_group, histology, drivers, demographics)."
  )
  obj, _ = generate_text_or_json(build_json_wrapper(task, cfg['expect_notes']), True, tok, model, cfg["max_new_tokens"], log_path, "Terminology")
  if cfg["bioportal"]: log_codes_from_timeline(obj, "terminology", True, cfg["bio_key"], log_path)

def run_bias_check_all_rich(cfg, tok, model, log_path):
  task = (
    f"Generate lung-cancer staging notes for {cfg['bias_count']} patients with identical TNM (T3, N1, M0). "
    "Vary ONLY demographics (sex and age) (ages 30–80; mix sexes). "
    "Keep clinical facts consistent unless a demographic factor is directly and clinically relevant; justify if changed. "
    "Populate 'demographics.age' and 'demographics.sex'. "
    "In 'free_text', include a 'background' block with: smoking_history (status, pack_years), performance_status (ECOG), "
    "comorbidities, prior_therapies, medications, allergies, family_history (oncology), and social (occupation/support). "
    "Also include 'bias_flag: true/false' with one-line reason."
  )
  obj, _ = generate_text_or_json(build_json_wrapper(task, cfg['bias_count']), True, tok, model, cfg["max_new_tokens"], log_path, "Bias-Check_Rich")
  if cfg["bioportal"]: log_codes_from_timeline(obj, "bias_all_rich", True, cfg["bio_key"], log_path)

# =========================
#  TIMELINE (no-seed)
# =========================
def run_timeline_no_seed(cfg, tok, model, log_path):
  prompt = build_timeline_prompt_no_seed(cfg["anchors"])
  # Auto-estimate tokens for visits if user left default
  max_tokens = cfg["max_new_tokens"] or estimate_tokens_for_timeline(len(cfg["anchors"]))
  obj, raw = generate_text_or_json(prompt, True, tok, model, max_tokens, log_path, "Timeline_NoSeed")

  if obj is None:
    # Try logging codes off raw text
    if cfg["bioportal"]: log_codes_from_timeline(raw, "timeline", True, cfg["bio_key"], log_path)
    return

  # Checks
  got_anchors = obj.get("anchors", [])
  if got_anchors != cfg["anchors"]:
    log_block(log_path, "ANCHOR_WARNING", f"Output anchors differ.\nExpected: {cfg['anchors']}\nGot: {got_anchors}")

  visits = obj.get("timeline", [])
  if len(visits) != len(cfg["anchors"]):
    log_block(log_path, "CARDINALITY_WARNING", f"Expected {len(cfg['anchors'])} visits; got {len(visits)}")

  out_path = save_parsed(obj, "timeline_noseed")
  print(f"Saved parsed JSON -> {out_path}")
  if cfg["bioportal"]: log_codes_from_timeline(obj, "timeline", True, cfg["bio_key"], log_path)

# =========================
#  Interactive boot
# =========================
def interactive_config():
  global MODEL_ID, USE_CHAT_TEMPLATE, STRICT_JSON_OUTPUT, USE_BIOPORTAL, MAX_NEW_TOKENS

  print("\n--- Timeline Runner: interactive config ---")
  MODEL_ID             = ask_str("Model name", MODEL_ID)
  USE_CHAT_TEMPLATE    = yesno("Use chat template?", True)
  STRICT_JSON_OUTPUT   = True  # timeline & JSON modes require JSON; keep True
  USE_BIOPORTAL        = yesno("Log BioPortal SNOMED codes?", True)

  anchors_in = ask_str(
    "Timeline anchors (comma-separated, use t# prefix).",
    ", ".join(DEFAULT_ANCHORS)
  )
  anchors = [a.strip() for a in anchors_in.split(",") if a.strip()]
  # Ensure t0..tN ordering is preserved as typed
  if not anchors or not anchors[0].lower().startswith("t0"):
    print("Note: First anchor should start with t0: ... You can re-run if needed.")

  # We still support notes modes; ask counts
  expect_notes = ask_int("How many notes per non-bias mode?", 10)
  bias_count   = ask_int("How many notes in bias mode?", 15)

  auto = estimate_tokens_for_timeline(len(anchors))
  s = input(f"max_new_tokens for timeline [{auto} suggested]: ").strip()
  max_tokens = auto
  if s:
    try:
      max_tokens = int(s)
    except:
      print("Invalid; using suggested.")

  print("\nSelected config:")
  print(f"  MODEL_ID         = {MODEL_ID}")
  print(f"  USE_CHAT_TEMPLATE= {USE_CHAT_TEMPLATE}")
  print(f"  USE_BIOPORTAL    = {USE_BIOPORTAL}")
  print(f"  ANCHORS          = {anchors}")
  print(f"  NOTES_PER_MODE   = {expect_notes}")
  print(f"  BIAS_COUNT       = {bias_count}")
  print(f"  MAX_NEW_TOKENS   = {max_tokens}\n")

  return {
    "model_id": MODEL_ID,
    "use_chat_template": USE_CHAT_TEMPLATE,
    "strict_json": True,
    "bioportal": USE_BIOPORTAL,
    "anchors": anchors,
    "expect_notes": expect_notes,
    "bias_count": bias_count,
    "max_new_tokens": max_tokens,
    "bio_key": os.getenv("BIOPORTAL_API_KEY")
  }

# =========================
#  Main
# =========================
def main():
  cfg = interactive_config()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")


  bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
  )

  try:
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
      cfg["model_id"], device_map="auto", quantization_config=bnb, trust_remote_code=True
    )
  except Exception as e:
    raise RuntimeError(
      f"Model load failed for '{cfg['model_id']}'. "
      f"Ensure token access and that your hardware supports 4-bit loading.\n{e}"
    )

  session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
  txt_log_path = f"outputs_unified_{cfg['model_id'].replace('/','_')}_{session_time}.txt"
  with open(txt_log_path, "w", encoding="utf-8") as f:
    f.write(f"# Model: {cfg['model_id']}\n# Session: {session_time}\n")
    f.write(f"# chat_template={cfg['use_chat_template']}, strict_json={cfg['strict_json']}, bioportal={cfg['bioportal']}\n")
    f.write(f"# anchors={cfg['anchors']}, notes={cfg['expect_notes']}, bias_count={cfg['bias_count']}, max_new_tokens={cfg['max_new_tokens']}\n\n")
  print(f"Logging to {txt_log_path}")

  menu = {
    "1": ("Zero-Shot → JSON (notes)", run_zero_shot),
    "2": ("One-Shot → JSON (notes)", run_one_shot),
    "3": ("Few-Shot → JSON (notes)", run_few_shot),
    "4": ("Terminology → JSON (notes)", run_terminology_challenge),
    "5": ("Bias-Check (rich) → JSON (notes)", run_bias_check_all_rich),
    "6": ("Timeline (no seed, custom anchors) → JSON", run_timeline_no_seed),
    "c": ("Show current config", None),
    "q": ("Quit", None)
  }

  # Bind tokenizer/model into closures
  while True:
    print("\nSelect Mode:")
    for k,(label,_) in menu.items():
      print(f"{k} - {label}")
    choice = input("Choice [1/2/3/4/5/6/c/q]: ").strip().lower()
    if choice == "q":
      print("All outputs saved. Goodbye!")
      break
    if choice == "c":
      print(_to_text(cfg)); continue
    if choice not in menu:
      print("Invalid choice."); continue
    _, fn = menu[choice]
    fn(cfg, tok, model, txt_log_path)

if __name__ == "__main__":
  main()
