# pipeline/phase4_finetuning.py
# DATAGEN Phase 4 — QLoRA Fine-Tuning
#
# ── CHANGELOG ─────────────────────────────────────────────────────────────────
# FIX 1 [CRITICAL] Pre-training label diversity check (check_label_diversity).
# FIX 2 [MINOR] Training set stats logged at startup.
# FIX 3 [CRITICAL] Adapter A' (label-scrambled control) via seed-42 derangement.
# FIX 4 [CRITICAL] Robust TNM normalization — handles pT1/cN2/T2a/bare-number
#   label forms in the real corpus. Strips leading p/c, drops a/b/c subdivisions,
#   prefixes bare numbers, blanks -> UNKNOWN. Must match Phase 3 exactly so the
#   scrambled derangement and entropy checks operate on clean 4-class labels.
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import random
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

import torch
import pandas as pd
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from scipy.stats import entropy as shannon_entropy

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

DIVERSITY_ENTROPY_FLOOR = {"T": 1.11, "N": 1.11, "M": 0.55}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_tnm_label(value: str, prefix: str) -> str:
    """
    FIX 4: Robust TNM normalization — identical to Phase 3.
      pT1 -> T1, cN2 -> N2, T2a -> T2, bare 3 -> T3, blank -> <PREFIX>UNKNOWN.
    """
    v = str(value).strip().upper()
    if v in ("", "NONE", "NULL", "NAN", "UNKNOWN"):
        return prefix.upper() + "UNKNOWN"
    v = re.sub(r"^[PC]", "", v)
    if v.startswith(prefix.upper()):
        v = v[len(prefix):]
    m = re.match(r"(IS|X|[0-4])", v)
    if not m:
        return prefix.upper() + "UNKNOWN"
    return prefix.upper() + m.group(1)


def check_label_diversity(df: pd.DataFrame, csv_path: str, abort_on_fail: bool = False) -> bool:
    print("\n" + "="*60)
    print(f"PRE-TRAINING DIVERSITY AUDIT: {csv_path}")
    print("="*60)

    all_pass = True
    for col, pfx, key in [("T_target", "T", "T"), ("N_target", "N", "N"), ("M_target", "M", "M")]:
        if col not in df.columns:
            print(f"  [{col}] MISSING — skipping"); continue
        labels = df[col].fillna("Unknown").astype(str).apply(
            lambda v: normalize_tnm_label(v, pfx))
        counts = labels.value_counts()
        ent    = shannon_entropy(counts) if len(counts) > 1 else 0.0
        floor  = DIVERSITY_ENTROPY_FLOOR[key]
        status = "✓ PASS" if ent >= floor else "✗ FAIL ⚠️"
        if ent < floor: all_pass = False
        print(f"  [{col}] entropy={ent:.3f}  floor={floor:.3f}  {status}")
        print(f"           distribution: {counts.to_dict()}")

    if not all_pass:
        msg = (
            "\n⚠️  LABEL DIVERSITY FAILURE DETECTED.\n"
            "The training corpus has one or more single-class TNM dimensions.\n"
            "The adapter will learn a degenerate prior (e.g. always predict T2/N0/M0).\n"
            "Recommendation: re-run Phase 1 with the stratified TNM grid.\n"
        )
        print(msg)
        if abort_on_fail:
            raise RuntimeError("Training aborted due to label diversity failure. "
                               "Set abort_on_fail=False to train anyway.")
    else:
        print("\n  ✓ All TNM dimensions pass diversity threshold. Proceeding to training.")
    print("="*60)
    return all_pass


def _make_derangement(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    indices = list(range(n))
    for attempt in range(1000):
        perm = indices[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm
    return indices[1:] + [indices[0]]


def prepare_scrambled_dataset(csv_path: str, tokenizer, seed: int = 42):
    if not os.path.exists(csv_path):
        print(f"Skipping scrambled dataset: {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path).reset_index(drop=True)
    n  = len(df)
    print(f"\n[Adapter A'] Scrambling labels for {n} records (seed={seed})...")

    perm = _make_derangement(n, seed=seed)

    df_scrambled = df.copy()
    df_scrambled["T_target"] = df["T_target"].iloc[perm].values
    df_scrambled["N_target"] = df["N_target"].iloc[perm].values
    df_scrambled["M_target"] = df["M_target"].iloc[perm].values

    retained = sum(
        df_scrambled["T_target"].iloc[i] == df["T_target"].iloc[i] and
        df_scrambled["N_target"].iloc[i] == df["N_target"].iloc[i] and
        df_scrambled["M_target"].iloc[i] == df["M_target"].iloc[i]
        for i in range(n)
    )
    print(f"  Records retaining original label triple: {retained} "
          f"({'✓ true derangement' if retained == 0 else '⚠️  not a derangement'})")

    print("\n  Entropy comparison (original vs scrambled):")
    for col, pfx, key in [("T_target", "T", "T"), ("N_target", "N", "N"), ("M_target", "M", "M")]:
        orig_ent = shannon_entropy(df[col].value_counts())
        scr_ent  = shannon_entropy(df_scrambled[col].value_counts())
        print(f"    [{col}] original={orig_ent:.4f}  scrambled={scr_ent:.4f}  "
              f"{'✓ identical' if abs(orig_ent - scr_ent) < 1e-10 else '⚠️  differs'}")

    formatted_texts = []
    for _, row in df_scrambled.iterrows():
        text = row["free_text"]
        t = normalize_tnm_label(str(row.get("T_target", "Unknown")), "T")
        n_ = normalize_tnm_label(str(row.get("N_target", "Unknown")), "N")
        m = normalize_tnm_label(str(row.get("M_target", "Unknown")), "M")

        prompt = (
            "You are a clinical data extractor. Read the clinical note and extract the TNM staging. "
            "Return a strictly formatted JSON object with keys 'T', 'N', and 'M'. "
            "Always use the full prefixed format: e.g. 'T2', 'N1', 'M0'. "
            "If a value is not found, use 'Unknown'.\n\n"
            f"NOTE: {text}"
        )
        completion = json.dumps({"T": t, "N": n_, "M": m})
        full_text  = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{completion}<|eot_id|>"
        )
        formatted_texts.append(full_text)

    dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def prepare_dataset(csv_path: str, tokenizer):
    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)

    print(f"\nTraining set: {len(df)} records from {csv_path}")
    check_label_diversity(df, csv_path, abort_on_fail=False)

    formatted_texts = []
    for _, row in df.iterrows():
        text = row["free_text"]
        t = normalize_tnm_label(str(row.get("T_target", "Unknown")), "T")
        n = normalize_tnm_label(str(row.get("N_target", "Unknown")), "N")
        m = normalize_tnm_label(str(row.get("M_target", "Unknown")), "M")

        prompt = (
            "You are a clinical data extractor. Read the clinical note and extract the TNM staging. "
            "Return a strictly formatted JSON object with keys 'T', 'N', and 'M'. "
            "Always use the full prefixed format: e.g. 'T2', 'N1', 'M0'. "
            "If a value is not found, use 'Unknown'.\n\n"
            f"NOTE: {text}"
        )
        completion = json.dumps({"T": t, "N": n, "M": m})
        full_text  = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{completion}<|eot_id|>"
        )
        formatted_texts.append(full_text)

    dataset = Dataset.from_dict({"text": formatted_texts})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


def train_qlora_adapter(dataset, output_dir: str, run_name: str, tokenizer, seed: int = 42):
    set_seed(seed)
    print(f"\n{'='*60}\nSTARTING TRAINING: {run_name}\n{'='*60}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    print("Loading base model into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto")

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        seed=seed,
        data_seed=seed,
        report_to="none",
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, train_dataset=dataset,
        data_collator=data_collator, args=training_args,
    )

    print(f"Training {run_name} adapter...")
    trainer.train()

    final_save = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_save)
    tokenizer.save_pretrained(final_save)
    print(f">> SUCCESS: {run_name} adapter saved to {final_save}")

    del model; del trainer
    torch.cuda.empty_cache(); gc.collect()


def main():
    tier1_csv = "data_splits/train_tier1_raw.csv"
    tier3_csv = "data_splits/train_tier3_golden.csv"

    missing = [p for p in (tier1_csv, tier3_csv) if not os.path.exists(p)]
    if missing:
        print(f"[ERROR] Missing input file(s): {missing}")
        print("        Run pipeline/phase3_benchmark.py first to produce data_splits/.")
        return

    os.makedirs("adapters/tier1_raw",       exist_ok=True)
    os.makedirs("adapters/tier3_scrambled", exist_ok=True)
    os.makedirs("adapters/tier3_golden",    exist_ok=True)

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "="*60)
    print("ADAPTER A — Tier 1 Raw (uninformative labels)")
    print("="*60)
    dataset_tier1 = prepare_dataset(tier1_csv, tokenizer)
    if dataset_tier1:
        train_qlora_adapter(dataset_tier1, "adapters/tier1_raw",
                            "Adapter A — Tier 1 (Raw)", tokenizer, seed=42)

    print("\n" + "="*60)
    print("ADAPTER A' — Tier 3 Golden (label-permuted, entropy preserved)")
    print("="*60)
    _df3 = pd.read_csv(tier3_csv)
    check_label_diversity(_df3, tier3_csv, abort_on_fail=True)
    dataset_scrambled = prepare_scrambled_dataset(tier3_csv, tokenizer, seed=42)
    if dataset_scrambled:
        train_qlora_adapter(dataset_scrambled, "adapters/tier3_scrambled",
                            "Adapter A' — Tier 3 (Scrambled)", tokenizer, seed=42)

    print("\n" + "="*60)
    print("ADAPTER B — Tier 3 Golden (intact content-label correspondence)")
    print("="*60)
    check_label_diversity(_df3, tier3_csv, abort_on_fail=True)
    dataset_tier3 = prepare_dataset(tier3_csv, tokenizer)
    if dataset_tier3:
        train_qlora_adapter(dataset_tier3, "adapters/tier3_golden",
                            "Adapter B — Tier 3 (Golden)", tokenizer, seed=42)

    print("\n" + "="*60)
    print("All three adapters trained:")
    print("  adapters/tier1_raw/final_adapter       — Adapter A")
    print("  adapters/tier3_scrambled/final_adapter — Adapter A'")
    print("  adapters/tier3_golden/final_adapter    — Adapter B")
    print("Next: run pipeline/phase3_benchmark.py with each adapter path.")
    print("="*60)


if __name__ == "__main__":
    main()
