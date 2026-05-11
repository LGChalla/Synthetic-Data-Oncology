"""
config.py — Central configuration for the Ablation-Oncology project.
All paths, model IDs, and hyperparameters live here.
Override any value via environment variable before running.
"""

import os

# ── HuggingFace cache ──────────────────────────────────────────────────────────
HF_HOME = os.getenv("HF_HOME",
          os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
os.environ["HF_HOME"]               = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HOME
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# ── API keys ───────────────────────────────────────────────────────────────────
HF_TOKEN       = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BIOPORTAL_KEY  = os.getenv("BIOPORTAL_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
GENERATOR_MODEL  = os.getenv("GENERATOR_MODEL",  "meta-llama/Llama-3.3-70B-Instruct")
FINETUNE_MODEL   = os.getenv("FINETUNE_MODEL",   "meta-llama/Meta-Llama-3-8B-Instruct")
EMBED_MODEL      = os.getenv("EMBED_MODEL",      "ncbi/MedCPT-Query-Encoder")

# ── Output paths ───────────────────────────────────────────────────────────────
ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR      = os.path.join(ROOT_DIR, "results")
DATA_DIR         = os.path.join(ROOT_DIR, "data")
ADAPTERS_DIR     = os.path.join(ROOT_DIR, "adapters")
DATA_SPLITS_DIR  = os.path.join(ROOT_DIR, "data_splits")

# ── Generation hyperparameters ─────────────────────────────────────────────────
GEN_PARAMS = {
    "temperature":   0.7,
    "top_p":         0.9,
    "top_k":         50,
    "do_sample":     True,
    "max_new_tokens": 1024,
}

# ── Ablation run sizes ─────────────────────────────────────────────────────────
# Each "run" generates one record for one TNM cell (round-robin over 32-cell grid).
# 64 runs = 2 full passes; 128 = 4 full passes.
GATE_ABLATION_RUNS = int(os.getenv("GATE_ABLATION_RUNS", "64"))
RAG_ABLATION_RUNS  = int(os.getenv("RAG_ABLATION_RUNS",  "32"))   # per condition

# ── Diversity gates (80% of theoretical max entropy) ──────────────────────────
ENTROPY_FLOORS = {"T": 1.109, "N": 1.109, "M": 0.554}

# ── QLoRA hyperparameters ──────────────────────────────────────────────────────
LORA_CONFIG = {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none", "task_type": "CAUSAL_LM",
}
TRAIN_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "optim": "paged_adamw_8bit",
    "fp16": True,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "report_to": "none",
    "seed": 42,
}

# ── MedCPT FAISS index paths ───────────────────────────────────────────────────
# Set via environment variable or pass --faiss-index to ablation_rag_vs_norag.py
FAISS_INDEX_PATH   = os.getenv("FAISS_INDEX_PATH", "")
FAISS_TEXTS_PATH   = os.getenv("FAISS_TEXTS_PATH", "")

# ── Ablation adapter names (used as output dir names and labels) ───────────────
ADAPTER_NAMES = {
    "ungated":      "adapter_A_ungated",
    "schema_only":  "adapter_B_schema",
    "schema_onto":  "adapter_C_schema_onto",
    "full_norag":   "adapter_D_full_norag",
    "full_rag":     "adapter_E_full_rag",
}

# ── MTSamples benchmark ────────────────────────────────────────────────────────
MTSAMPLES_DATASET  = "harishnair04/mtsamples"
MTSAMPLES_LUNG_CSV = os.path.join(DATA_SPLITS_DIR, "mtsamples_lung_gold.csv")
MTSAMPLES_ALL_CSV  = os.path.join(DATA_SPLITS_DIR, "mtsamples_all_cancer_gold.csv")
TSTR_BOOTSTRAP_N   = 1000
