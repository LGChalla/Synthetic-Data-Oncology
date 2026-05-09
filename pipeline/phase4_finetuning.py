# Phase4_fixed.py
# DATAGEN Phase 4 — QLoRA Fine-Tuning
#
# ── CHANGELOG ─────────────────────────────────────────────────────────────────
# FIX 1 [CRITICAL] Pre-training label diversity check:
#   Original silently trained on whatever CSV was handed to it. If the training
#   CSV is label-uniform (e.g. all T2/N0/M0), the adapter will learn a degenerate
#   prior. Now check_label_diversity() runs BEFORE training and raises a clear
#   warning (with option to abort) if any TNM dimension is single-class.
#
# FIX 2 [MINOR] Training set stats logged at startup:
#   Prints label distribution so the researcher can see diversity at a glance
#   without re-running Phase 2.
#
# All other logic (QLoRA config, optimizer, adapter saving) is unchanged.
# ──────────────────────────────────────────────────────────────────────────────

import os
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# Minimum entropy thresholds (same as Phase 1 and 2)
DIVERSITY_ENTROPY_FLOOR = {"T": 1.11, "N": 1.11, "M": 0.55}


def normalize_tnm_label(value: str, prefix: str) -> str:
    v = str(value).strip().upper()
    if not v.startswith(prefix.upper()):
        v = prefix.upper() + v
    return v


def check_label_diversity(df: pd.DataFrame, csv_path: str, abort_on_fail: bool = False) -> bool:
    """
    FIX 1: Audits T/N/M label distributions before training begins.
    Prints PASS/FAIL per dimension. If abort_on_fail=True and any dimension
    fails, raises RuntimeError to stop training on a degenerate corpus.

    Returns True if all dimensions pass, False otherwise.
    """
    print("\n" + "="*60)
    print(f"PRE-TRAINING DIVERSITY AUDIT: {csv_path}")
    print("="*60)

    all_pass = True
    for col, pfx, key in [("T_target", "T", "T"), ("N_target", "N", "N"), ("M_target", "M", "M")]:
        if col not in df.columns:
            print(f"  [{col}] MISSING — skipping"); continue
        labels  = df[col].fillna("Unknown").astype(str).apply(
            lambda v: normalize_tnm_label(v, pfx))
        counts  = labels.value_counts()
        ent     = shannon_entropy(counts) if len(counts) > 1 else 0.0
        floor   = DIVERSITY_ENTROPY_FLOOR[key]
        status  = "✓ PASS" if ent >= floor else "✗ FAIL ⚠️"
        if ent < floor: all_pass = False
        print(f"  [{col}] entropy={ent:.3f}  floor={floor:.3f}  {status}")
        print(f"           distribution: {counts.to_dict()}")

    if not all_pass:
        msg = (
            "\n⚠️  LABEL DIVERSITY FAILURE DETECTED.\n"
            "The training corpus has one or more single-class TNM dimensions.\n"
            "The adapter will learn a degenerate prior (e.g. always predict T2/N0/M0).\n"
            "Recommendation: re-run Phase 1 with the stratified TNM grid (Phase1_fixed.py)\n"
            "to produce a balanced corpus before fine-tuning."
        )
        print(msg)
        if abort_on_fail:
            raise RuntimeError("Training aborted due to label diversity failure. "
                               "Set abort_on_fail=False to train anyway.")
    else:
        print("\n  ✓ All TNM dimensions pass diversity threshold. Proceeding to training.")
    print("="*60)
    return all_pass


def prepare_dataset(csv_path, tokenizer):
    """Loads CSV, formats Prompt-Completion pairs, tokenizes."""
    if not os.path.exists(csv_path):
        print(f"Skipping: {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)

    # FIX 2: Log training set stats
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
        completion = f'{{"T": "{t}", "N": "{n}", "M": "{m}"}}'
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


def train_qlora_adapter(dataset, output_dir, run_name, tokenizer):
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

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Adapter A — Tier 1 Raw
    dataset_tier1 = prepare_dataset(tier1_csv, tokenizer)
    if dataset_tier1:
        train_qlora_adapter(dataset_tier1, "adapters/tier1_raw",    "Tier 1 (Raw)",    tokenizer)

    # Adapter B — Tier 3 Golden
    dataset_tier3 = prepare_dataset(tier3_csv, tokenizer)
    if dataset_tier3:
        train_qlora_adapter(dataset_tier3, "adapters/tier3_golden", "Tier 3 (Golden)", tokenizer)

    print("\nAll adapters trained.")
    print("Next: run Phase3_fixed.py to benchmark.")


if __name__ == "__main__":
    main()
