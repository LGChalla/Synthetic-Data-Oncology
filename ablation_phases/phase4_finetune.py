"""
Phase 4 — QLoRA Adapter Training (One per Ablation Condition)
==============================================================
Trains five adapters, one per ablation corpus. Each adapter sees only
the records admitted under its condition's gate, so downstream TSTR
performance directly reflects the quality of that gating strategy.

Adapter map:
  adapter_A_ungated      <- condition=ungated       (all records, no gate)
  adapter_B_schema       <- condition=schema_only   (JSON schema only)
  adapter_C_schema_onto  <- condition=schema_onto   (schema + SNOMED CT)
  adapter_D_full_norag   <- condition=full_norag    (full G(x), no RAG)
  adapter_E_full_rag     <- condition=full_rag      (full G(x) + RAG)

Pre-training checks:
  - Aborts if T/N/M diversity fails entropy floor for D and E (golden corpora).
  - Warns (does not abort) for A, B, C — some diversity failure is expected.

Usage:
  python phases/phase4_finetune.py
  python phases/phase4_finetune.py --condition full_rag
  python phases/phase4_finetune.py --condition all
  python phases/phase4_finetune.py --results-dir /path --adapters-dir /path
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from core.logging_utils import load_jsonl
from core.tnm_grid      import audit_diversity, print_diversity_report

CONDITION_TO_ADAPTER = {
    "ungated":     "adapter_A_ungated",
    "schema_only": "adapter_B_schema",
    "schema_onto": "adapter_C_schema_onto",
    "full_norag":  "adapter_D_full_norag",
    "full_rag":    "adapter_E_full_rag",
}
# For golden corpora (D, E), abort training on diversity failure.
STRICT_CONDITIONS = {"full_norag", "full_rag"}


# ── Data preparation ──────────────────────────────────────────────────────────

def load_corpus(condition: str, results_dir: str) -> list:
    path = os.path.join(results_dir, f"phase1_{condition}.jsonl")
    recs = load_jsonl(path)
    return [r for r in recs if r.get("admitted")]


def build_training_rows(records: list) -> list:
    rows = []
    for r in records:
        pj  = r.get("parsed_json") or {}
        try:
            note = pj.get("notes", [{}])[0]
        except (IndexError, AttributeError):
            note = {}
        free_text = note.get("free_text", "") or r.get("raw_output", "")
        t = str(r.get("T", "Unknown")).upper()
        n = str(r.get("N", "Unknown")).upper()
        m = str(r.get("M", "Unknown")).upper()
        if not free_text or "Unknown" in (t, n, m):
            continue
        rows.append({"free_text": free_text, "T": t, "N": n, "M": m})
    return rows


def tokenize_dataset(rows: list, tokenizer) -> Dataset:
    texts = []
    for row in rows:
        prompt = (
            "You are a clinical data extractor. Read the clinical note and "
            "extract the TNM staging. Return a JSON object with keys 'T', 'N', 'M'.\n\n"
            f"NOTE: {row['free_text']}"
        )
        completion = json.dumps({"T": row["T"], "N": row["N"], "M": row["M"]})
        full = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{completion}<|eot_id|>"
        )
        texts.append(full)

    ds = Dataset.from_dict({"text": texts})
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)
    return ds.map(tokenize, batched=True, remove_columns=["text"])


# ── Diversity gate ────────────────────────────────────────────────────────────

def check_corpus_diversity(records: list, condition: str) -> bool:
    report = audit_diversity(records)
    print_diversity_report(report, label=condition)
    if not report["all_pass"]:
        if condition in STRICT_CONDITIONS:
            raise RuntimeError(
                f"[{condition}] Diversity gate FAILED. "
                "Training aborted — re-run Phase 1 to generate more records "
                "for under-represented TNM cells before fine-tuning."
            )
        else:
            print(f"  [WARN] [{condition}] Diversity FAILED — proceeding anyway "
                  f"(expected for less-filtered corpora).")
    return report["all_pass"]


# ── Training ──────────────────────────────────────────────────────────────────

def train_adapter(condition: str, results_dir: str, adapters_dir: str):
    adapter_name = CONDITION_TO_ADAPTER[condition]
    output_dir   = os.path.join(adapters_dir, adapter_name)
    final_dir    = os.path.join(output_dir, "final_adapter")

    print(f"\n{'='*65}")
    print(f"PHASE 4 — Training  |  {condition}  ->  {adapter_name}")
    print(f"{'='*65}")

    records = load_corpus(condition, results_dir)
    if not records:
        print(f"  [SKIP] No admitted records for condition='{condition}'.")
        print(f"         Run: python phases/phase1_generate.py --condition {condition}")
        return

    rows = build_training_rows(records)
    if not rows:
        print(f"  [SKIP] No valid training rows (missing free_text or TNM labels).")
        return

    print(f"  Corpus: {len(records)} admitted records -> {len(rows)} training rows")
    check_corpus_diversity(records, condition)

    # Tokenizer
    print(f"  Loading tokenizer: {cfg.FINETUNE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.FINETUNE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = tokenize_dataset(rows, tokenizer)
    print(f"  Tokenized dataset: {len(dataset)} examples")

    # Model
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"  Loading base model (4-bit NF4)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.FINETUNE_MODEL, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**cfg.LORA_CONFIG))
    model.print_trainable_parameters()

    train_cfg = cfg.TRAIN_CONFIG.copy()
    train_cfg["output_dir"] = output_dir
    args = TrainingArguments(**train_cfg)

    trainer = Trainer(
        model=model, train_dataset=dataset, args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"  Training {adapter_name}...")
    trainer.train()

    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  Adapter saved -> {final_dir}")

    del model, trainer
    torch.cuda.empty_cache()
    import gc; gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: QLoRA Adapter Training")
    parser.add_argument("--condition",    default="all",
                        help="Condition to train (or 'all' for all five)")
    parser.add_argument("--results-dir",  default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir", default=cfg.ADAPTERS_DIR)
    args = parser.parse_args()

    os.makedirs(args.adapters_dir, exist_ok=True)
    conditions = (list(CONDITION_TO_ADAPTER.keys())
                  if args.condition == "all"
                  else [args.condition])

    for cond in conditions:
        if cond not in CONDITION_TO_ADAPTER:
            print(f"[SKIP] Unknown condition: {cond}")
            continue
        try:
            train_adapter(cond, args.results_dir, args.adapters_dir)
        except RuntimeError as e:
            print(f"  [ERROR] {e}")
            print(f"  Skipping {cond} — fix diversity issue and re-run.")

    print("\nPhase 4 complete. Next: run phases/phase3_benchmark.py")


if __name__ == "__main__":
    main()
