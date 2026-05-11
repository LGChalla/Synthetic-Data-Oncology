"""
Phase 3 — TSTR Benchmark Construction and Evaluation
======================================================
Loads trained adapters from Phase 4, evaluates them on held-out synthetic
data and real MTSamples clinical notes under the train-on-synthetic,
test-on-real (TSTR) protocol.

Adapters evaluated (one per ablation condition):
  adapter_A_ungated      — trained on no-gate corpus
  adapter_B_schema       — trained on schema-only corpus
  adapter_C_schema_onto  — trained on schema+ontology corpus
  adapter_D_full_norag   — trained on full G(x) corpus, no RAG
  adapter_E_full_rag     — trained on full G(x) + RAG corpus

Test sets (increasing distributional distance from training):
  1. Synthetic held-out (n=~40)   — in-distribution
  2. MTSamples Lung (n=53)        — confirmed lung cancer notes
  3. MTSamples All-Cancer (n=283) — cross-tumour transfer

Primary metric: per-class T-stage accuracy with 95% bootstrap CI.
Aggregate accuracy is reported alongside the constant-classifier ceiling.

Usage:
  python phases/phase3_benchmark.py
  python phases/phase3_benchmark.py --results-dir /path/to/results --adapters-dir /path/to/adapters
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import bootstrap
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from core.generation    import load_hf_model, generate_one, parse_output
from core.logging_utils import load_jsonl
from core.schemas       import EXTRACTION_PROMPT

ADAPTER_ORDER = [
    "adapter_A_ungated",
    "adapter_B_schema",
    "adapter_C_schema_onto",
    "adapter_D_full_norag",
    "adapter_E_full_rag",
]


# ── MTSamples loader ──────────────────────────────────────────────────────────

def load_mtsamples(lung_path: str, all_path: str) -> dict:
    sets = {}
    for name, path in [("lung", lung_path), ("all_cancer", all_path)]:
        if os.path.exists(path):
            sets[name] = pd.read_csv(path)
            print(f"  MTSamples {name}: {len(sets[name])} records")
        else:
            print(f"  [SKIP] {path} not found — run preprocessing/mtsamples_prep.py first.")
    return sets


def load_synthetic_holdout(results_dir: str, condition: str = "full_rag") -> pd.DataFrame:
    """Uses the gated corpus from Phase 1 as in-distribution held-out test."""
    path = os.path.join(results_dir, f"phase1_{condition}.jsonl")
    recs = load_jsonl(path)
    admitted = [r for r in recs if r.get("gate_pass")]
    # 20% held-out
    n_holdout = max(int(len(admitted) * 0.2), 1)
    holdout   = admitted[-n_holdout:]
    rows = []
    for r in holdout:
        pj = r.get("parsed_json") or {}
        try:
            stg = pj.get("notes", [{}])[0].get("staging", {})
        except (IndexError, AttributeError):
            stg = {}
        rows.append({
            "free_text": pj.get("notes", [{}])[0].get("free_text", r.get("raw_output","")),
            "T_label":   str(stg.get("T","Unknown")).upper(),
            "N_label":   str(stg.get("N","Unknown")).upper(),
            "M_label":   str(stg.get("M","Unknown")).upper(),
            "source":    "synthetic",
        })
    return pd.DataFrame(rows)


# ── Adapter loading ───────────────────────────────────────────────────────────

def load_adapter(adapter_name: str, adapters_dir: str):
    from peft import PeftModel
    adapter_path = os.path.join(adapters_dir, adapter_name, "final_adapter")
    if not os.path.exists(adapter_path):
        print(f"  [SKIP] Adapter not found: {adapter_path}")
        return None, None
    base_model, tokenizer = load_hf_model(cfg.FINETUNE_MODEL)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"  Loaded adapter: {adapter_name}")
    return model, tokenizer


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_tnm(model, tokenizer, text: str, model_id: str) -> tuple:
    prompt = EXTRACTION_PROMPT.format(text=text[:2000])
    raw    = generate_one(model, tokenizer, prompt,
                          {"temperature": 0.1, "do_sample": False, "max_new_tokens": 64},
                          model_id=model_id)
    parsed, _ = parse_output(raw)
    if not isinstance(parsed, dict):
        return "Unknown", "Unknown", "Unknown"
    return (
        str(parsed.get("T", "Unknown")).upper(),
        str(parsed.get("N", "Unknown")).upper(),
        str(parsed.get("M", "Unknown")).upper(),
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def bootstrap_accuracy(y_true, y_pred, n=1000) -> tuple:
    if len(y_true) < 2:
        acc = accuracy_score(y_true, y_pred)
        return acc, acc, acc
    arr = np.array([(yt == yp) for yt, yp in zip(y_true, y_pred)], dtype=float)
    res = bootstrap((arr,), np.mean, n_resamples=n, confidence_level=0.95,
                    random_state=42, method="percentile")
    return (float(np.mean(arr)),
            float(res.confidence_interval.low),
            float(res.confidence_interval.high))


def per_class_accuracy(y_true, y_pred) -> dict:
    classes = sorted(set(y_true))
    out = {}
    for cls in classes:
        idxs = [i for i, y in enumerate(y_true) if y == cls]
        if not idxs:
            continue
        correct = sum(1 for i in idxs if y_pred[i] == cls)
        out[cls] = round(correct / len(idxs), 3)
    return out


def evaluate_on_set(model, tokenizer, df: pd.DataFrame,
                    label_col: str, model_id: str, set_name: str) -> dict:
    y_true, y_pred = [], []
    valid = df[df[label_col].notna() & (df[label_col] != "Unknown")].copy()
    print(f"    Evaluating on {set_name} (n={len(valid)})...")
    for _, row in valid.iterrows():
        t_pred, _, _ = extract_tnm(model, tokenizer, str(row.get("free_text", "")), model_id)
        y_true.append(str(row[label_col]).upper())
        y_pred.append(t_pred)

    if not y_true:
        return {}

    acc, lo, hi   = bootstrap_accuracy(y_true, y_pred)
    const_cls      = pd.Series(y_true).mode()[0]
    const_acc      = sum(1 for y in y_true if y == const_cls) / len(y_true)
    per_cls        = per_class_accuracy(y_true, y_pred)

    classes = sorted(set(y_true))
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro",
                        zero_division=0) if len(classes) > 1 else None

    return {
        "set":            set_name,
        "n":              len(y_true),
        "accuracy":       round(acc, 3),
        "ci_low":         round(lo, 3),
        "ci_high":        round(hi, 3),
        "constant_clf":   round(const_acc, 3),
        "macro_f1":       round(macro_f1, 3) if macro_f1 is not None else None,
        "per_class":      per_cls,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: TSTR Benchmark")
    parser.add_argument("--results-dir",  default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir", default=cfg.ADAPTERS_DIR)
    parser.add_argument("--export-dir",   default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    print("Loading test sets...")
    mt_sets  = load_mtsamples(cfg.MTSAMPLES_LUNG_CSV, cfg.MTSAMPLES_ALL_CSV)
    synth_df = load_synthetic_holdout(args.results_dir)
    print(f"  Synthetic held-out: {len(synth_df)} records")

    all_results = []
    for adapter_name in ADAPTER_ORDER:
        print(f"\n--- Adapter: {adapter_name} ---")
        model, tokenizer = load_adapter(adapter_name, args.adapters_dir)
        if model is None:
            continue

        for set_name, df, label_col in [
            ("synthetic_holdout", synth_df,                  "T_label"),
            ("mtsamples_lung",    mt_sets.get("lung",    pd.DataFrame()), "T_label"),
            ("mtsamples_all",     mt_sets.get("all_cancer", pd.DataFrame()), "T_label"),
        ]:
            if df.empty:
                continue
            res = evaluate_on_set(model, tokenizer, df, label_col,
                                  cfg.FINETUNE_MODEL, set_name)
            if res:
                res["adapter"] = adapter_name
                all_results.append(res)
                print(f"    {set_name}: acc={res['accuracy']:.3f} "
                      f"[{res['ci_low']:.3f},{res['ci_high']:.3f}] "
                      f"const={res['constant_clf']:.3f}")
                for cls, acc in res["per_class"].items():
                    print(f"      {cls}: {acc:.3f}")

        from core.generation import unload_model
        unload_model(model, tokenizer)

    # Export aggregate results
    if all_results:
        agg_rows = []
        for r in all_results:
            row = {k: v for k, v in r.items() if k != "per_class"}
            for cls, acc in r.get("per_class", {}).items():
                row[f"acc_{cls}"] = acc
            agg_rows.append(row)
        df_out = pd.DataFrame(agg_rows)
        path   = os.path.join(args.export_dir, "phase3_tstr_results.csv")
        df_out.to_csv(path, index=False)
        print(f"\nTSTR results saved -> {path}")

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
