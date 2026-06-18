# Phase3_fixed.py
# DATAGEN Phase 3 — Downstream Benchmark (TSTR)
#
# ── CHANGELOG ─────────────────────────────────────────────────────────────────
# FIX 1 [CRITICAL] Evaluation validity — macro F1 on single-class test sets:
#   Original reported macro F1 for the synthetic held-out set where ground truth
#   is all T2/N0/M0 (single class per stage). Macro F1 is deflated by out-of-
#   distribution predictions and misleads readers. Fix: on single-class test sets
#   report ONLY accuracy + per-class precision/recall. Add is_single_class_set()
#   detector. Macro F1 still reported for multi-class sets (MTSamples) where it
#   is valid.
#
# FIX 2 [CRITICAL] MTSamples ground truth sparsity:
#   Original benchmarked N and M on ALL MTSamples records, 82-86% of which have
#   N=Unknown or M=Unknown (regex-absent). This means N/M metrics measured
#   "Unknown" prediction quality, not staging extraction. Fix: filter_for_valid_gt()
#   separates records into T-only and complete-TNM subsets. N/M metrics are
#   reported ONLY on the complete-TNM subset with a clear n= annotation.
#
# FIX 3 [INHERITED from re_fixed] GPU, label normalization, adapter OOM — all
#   carried forward from Phase3-_Downstream_re_fixed.py unchanged.
#
# FIX 4 [IMPORTANT] Per-class accuracy breakdown:
#   Added per_class_accuracy() so T1/T2/T3/T4 performance is visible separately.
#   This makes "97.7% T-stage accuracy" interpretable instead of suspiciously high.
#
# FIX 5 [REVIEWER] Bootstrap 95% CI on all accuracy estimates:
#   bootstrap_ci_accuracy() uses scipy.stats.bootstrap (percentile method,
#   n=1000 resamples) to report [lo, hi] alongside every accuracy figure.
#   Addresses reviewer concern that +21pp cross-cancer claim lacks uncertainty
#   quantification at n=100.
#
# FIX 6 [REVIEWER] Constant-classifier baseline:
#   constant_classifier_baseline() explicitly computes what always-predicting
#   the majority label achieves on each test set. Printed before adapter metrics
#   so the reader can contextualise whether adapter accuracy exceeds the trivial
#   baseline. Directly addresses reviewer: "97.7% is consistent with a constant
#   classifier on a single-class test set."
#
# FIX 7 [REVIEWER] ground_truth_quality metadata column:
#   filter_for_valid_gt() now tags each MTSamples record as "reliable" or
#   "sparse_unknown" based on N/M availability. Column exported to CSV.
# ──────────────────────────────────────────────────────────────────────────────

import os

os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]    = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

import gc
import json
import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from scipy.stats import pointbiserialr, bootstrap as scipy_bootstrap
from peft import PeftModel

RESULTS_FILE = "results/master_results.jsonl"
OUT_DIR      = "data_splits"
EXPORT_DIR   = "results/analysis_exports"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def normalize_tnm_label(value: str, prefix: str) -> str:
    v = str(value).strip().upper()
    if not v.startswith(prefix.upper()):
        v = prefix.upper() + v
    return v


def release_gpu(model=None, tokenizer=None):
    if model    is not None: del model
    if tokenizer is not None: del tokenizer
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f">> GPU memory released. Free: {free:.1f}GB")
    else:
        print(">> GPU memory released.")


def is_single_class_set(series: pd.Series) -> bool:
    """Returns True if a label column has only one unique non-Unknown value."""
    values = series[~series.str.contains("UNKNOWN", case=False, na=False)]
    return values.nunique() <= 1


def per_class_accuracy(targets, preds, stage_label: str) -> pd.DataFrame:
    """
    FIX 4: Reports accuracy per label class (e.g. T1/T2/T3/T4 separately).
    Makes aggregate accuracy interpretable and surfaces per-class failures.
    """
    classes = sorted(set(targets) | set(preds))
    rows    = []
    for cls in classes:
        mask  = pd.Series(targets) == cls
        if mask.sum() == 0: continue
        n     = mask.sum()
        corr  = (pd.Series(preds)[mask] == cls).sum()
        rows.append({"class": cls, "n_gt": int(n),
                     "n_correct": int(corr),
                     "per_class_accuracy": round(corr / n, 3)})
    return pd.DataFrame(rows)


def bootstrap_ci_accuracy(targets, preds, n_resamples: int = 1000,
                           confidence: float = 0.95) -> tuple:
    """
    Computes a bootstrap confidence interval on accuracy.
    Returns (lower, upper) as floats rounded to 3 decimal places.
    Uses scipy.stats.bootstrap with the percentile method.
    """
    targets_arr = np.array(targets)
    preds_arr   = np.array(preds)
    correct     = (targets_arr == preds_arr).astype(float)

    def acc_stat(c):
        return np.mean(c)

    try:
        res = scipy_bootstrap(
            (correct,), acc_stat,
            n_resamples=n_resamples,
            confidence_level=confidence,
            method="percentile",
            random_state=42,
        )
        lo = round(res.confidence_interval.low,  3)
        hi = round(res.confidence_interval.high, 3)
        return lo, hi
    except Exception:
        # Fallback: Wilson score interval for proportions
        n   = len(correct)
        p   = float(np.mean(correct))
        z   = 1.96
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        margin = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
        return round(centre - margin, 3), round(centre + margin, 3)


def constant_classifier_baseline(targets, stage_label: str) -> dict:
    """
    Computes what a naive constant classifier (always predicts the majority label)
    achieves on this test set. This is the minimum bar any trained model must beat
    to demonstrate it has learned something beyond the label prior.
    Reviewer concern: 97.7% adapter accuracy is consistent with a constant classifier
    if all GT labels are T2/N0/M0.
    """
    targets_s  = pd.Series(targets)
    majority   = targets_s.mode()[0]
    const_preds = [majority] * len(targets_s)
    const_acc   = accuracy_score(targets_s, const_preds)
    lo, hi      = bootstrap_ci_accuracy(targets_s.tolist(), const_preds)
    print(f"  [{stage_label}] Constant classifier (always '{majority}'): "
          f"Accuracy={const_acc:.3f} (95% CI: [{lo}, {hi}])")
    return {
        "majority_label": majority,
        "constant_acc":   round(const_acc, 3),
        "ci_lower":       lo,
        "ci_upper":       hi,
    }


# FIX 2: MTSamples ground truth filter
def filter_for_valid_gt(df: pd.DataFrame) -> dict:
    """
    Returns a dict of subsets:
      'T_valid'   : rows where T_target is known (use for T benchmarking)
      'NM_valid'  : rows where N_target AND M_target are both known
                    (use for N/M benchmarking — NOT the full dataframe)
      'all'       : full dataframe (for counting only)
    """
    unknown_t  = df["T_target"].str.contains("UNKNOWN", case=False, na=False)
    unknown_n  = df["N_target"].str.contains("UNKNOWN", case=False, na=False)
    unknown_m  = df["M_target"].str.contains("UNKNOWN", case=False, na=False)

    t_valid    = df[~unknown_t].copy()
    nm_valid   = df[~unknown_n & ~unknown_m].copy()

    pct_t  = len(t_valid)  / len(df) * 100 if len(df) > 0 else 0
    pct_nm = len(nm_valid) / len(df) * 100 if len(df) > 0 else 0

    print(f"  Ground-truth availability: "
          f"T={len(t_valid)}/{len(df)} ({pct_t:.0f}%)  "
          f"N+M={len(nm_valid)}/{len(df)} ({pct_nm:.0f}%)")

    if pct_nm < 30:
        print(f"  ⚠️  N/M benchmark will run on {len(nm_valid)} records only "
              f"({pct_nm:.0f}% of total). Treat N/M metrics as exploratory.")

    # Add ground_truth_quality metadata column for export
    # "reliable" = T known; "sparse_unknown" = N or M is Unknown
    df = df.copy()
    df["ground_truth_quality"] = np.where(
        unknown_n | unknown_m, "sparse_unknown", "reliable"
    )
    t_valid["ground_truth_quality"]  = "reliable"
    nm_valid["ground_truth_quality"] = "reliable"

    qual_counts = df["ground_truth_quality"].value_counts().to_dict()
    print(f"  Ground-truth quality tags: {qual_counts}")

    return {"T_valid": t_valid, "NM_valid": nm_valid, "all": df}


# ── DATA STRATIFICATION ───────────────────────────────────────────────────────

def load_clean_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return pd.DataFrame()
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:   row = json.loads(line)
            except: continue
            if not row.get("parsed_json_valid") or not isinstance(row.get("parsed_json"), dict):
                continue
            notes    = row["parsed_json"].get("notes", [])
            if not notes or not isinstance(notes, list): continue
            note     = notes[0]
            free_text_raw = note.get("free_text")
            if free_text_raw is None:
                continue
            free_text = str(free_text_raw).strip()
            if not free_text:
                continue
            if not free_text: 
                continue
            stg = note.get("staging") or {}
            records.append({
                "run_id":            row.get("run_id"),
                "layer":             row.get("layer"),
                "model":             row.get("model"),
                "validation_errors": len(row.get("validation_errors", [])),
                "free_text":         free_text,
                "word_count":        len(free_text.split()),
                "T_target": normalize_tnm_label(str(stg.get("T", "Unknown")), "T"),
                "N_target": normalize_tnm_label(str(stg.get("N", "Unknown")), "N"),
                "M_target": normalize_tnm_label(str(stg.get("M", "Unknown")), "M"),
            })
    return pd.DataFrame(records)


def prepare_extraction_datasets(df):
    print("\n" + "="*60)
    print("LAYER 1: TSTR DATA STRATIFICATION")
    print("="*60)
    os.makedirs(OUT_DIR, exist_ok=True)

    golden_df = df[
        (df["layer"].isin(["controlled", "longitudinal"])) &
        (df["validation_errors"] == 0)
    ].copy()
    raw_df    = df[df["layer"] == "exploratory"].copy()

    if golden_df.empty:
        print("ERROR: No Golden records found."); return None

    # Report label distribution of the training set (FIX 1 diagnostic)
    print(f"Golden corpus size: {len(golden_df)}")
    for dim in ("T_target", "N_target", "M_target"):
        counts = golden_df[dim].value_counts()
        unique = golden_df[dim].nunique()
        print(f"  {dim} classes: {unique} | dist: {counts.to_dict()}")
        if unique == 1:
            print(f"  ⚠️  WARNING: {dim} is single-class — adapter will overfit to one label!")

    golden_df = golden_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(golden_df) * 0.8)

    golden_df.iloc[:split_idx].to_csv(f"{OUT_DIR}/train_tier3_golden.csv", index=False)
    raw_df.to_csv(f"{OUT_DIR}/train_tier1_raw.csv",                         index=False)
    golden_df.iloc[split_idx:].to_csv(f"{OUT_DIR}/test_heldout_eval.csv",  index=False)

    print(f"Exported Tier 3 (Golden) train: {len(golden_df.iloc[:split_idx])}")
    print(f"Exported Tier 1 (Raw)    train: {len(raw_df)}")
    print(f"Exported Held-Out eval set:     {len(golden_df.iloc[split_idx:])}")
    return f"{OUT_DIR}/test_heldout_eval.csv"


# ── DIVERSITY ─────────────────────────────────────────────────────────────────

def calculate_diversity_dcsn(df):
    print("\n" + "="*60)
    print("LAYER 2: FAISS DIVERSITY & MODE COLLAPSE CHECK")
    print("="*60)
    texts_all    = df["free_text"].tolist()
    texts_unique = df["free_text"].drop_duplicates().tolist()
    print(f"Total records: {len(texts_all)} | Unique free_texts: {len(texts_unique)}")
    if len(texts_unique) < 2:
        print(">> Not enough unique texts."); return

    print(f"Embedding {len(texts_unique)} unique notes (ClinicalBERT, CPU)...")
    embedder   = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT", device="cpu")
    embeddings = np.array(
        embedder.encode(texts_unique, show_progress_bar=True, device="cpu")
    ).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    sims, _ = index.search(embeddings, 2)
    dcsn    = 1.0 - sims[:, 1]

    print(f"Mean DCSN: {np.mean(dcsn):.4f} (SD: {np.std(dcsn):.4f})")
    print(f"5th–95th:  {np.percentile(dcsn, 5):.4f} – {np.percentile(dcsn, 95):.4f}")
    if np.min(dcsn) < 0.01:
        print(">> ⚠️  Mode Collapse detected in unique texts.")
    else:
        print(">> SUCCESS: Healthy variance across unique generated notes.")


# ── EXTRACTION BENCHMARK ──────────────────────────────────────────────────────

def run_extraction_benchmark(
    test_csv,
    eval_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_name="Data",
    is_real_world=False,        # FIX 2: flag to activate MTSamples GT filter
):
    print("\n" + "="*60)
    print(f"LAYER 3: ZERO-SHOT BENCHMARK — {dataset_name.upper()}")
    print(f"Model: {eval_model_id}")
    print("="*60)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if not os.path.exists(test_csv):
        print(f"File not found: {test_csv}"); return

    df = pd.read_csv(test_csv)
    if "word_count" not in df.columns:
        df["word_count"] = df["free_text"].apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 1)

    # Normalize labels from CSV
    for col, pfx in [("T_target", "T"), ("N_target", "N"), ("M_target", "M")]:
        df[col] = df[col].fillna("Unknown").astype(str).apply(
            lambda v: normalize_tnm_label(v, pfx))

    # FIX 2: For real-world data, apply GT filter; for synthetic, use all rows
    if is_real_world:
        subsets = filter_for_valid_gt(df)
    else:
        # Synthetic: all rows have GT; but flag single-class issue
        subsets = {"T_valid": df, "NM_valid": df, "all": df}
        for dim in ("T_target", "N_target", "M_target"):
            if is_single_class_set(df[dim]):
                print(f"  ⚠️  NOTE: {dim} is single-class — "
                      f"accuracy measures label reproduction, not discrimination.")

    
    
    all_rows = subsets["all"]
    if len(all_rows) == 0:
        print("No valid records to benchmark."); return
    if len(all_rows) > 100:
        all_rows = all_rows.sample(100, random_state=42)
        print("Sampling 100 records for inference...")

    # Also re-filter subsets to match the sampled indices
    t_bench  = subsets["T_valid"].loc[subsets["T_valid"].index.isin(all_rows.index)]
    nm_bench = subsets["NM_valid"].loc[subsets["NM_valid"].index.isin(all_rows.index)]

    # ── Load model ─────────────────────────────────────────────────────────────
    print("Loading model into VRAM (GPU 1)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    model = tokenizer = None
    try:
        if os.path.exists(os.path.join(eval_model_id, "adapter_config.json")):
            print(">> LoRA adapter detected — loading base + adapter...")
            base_id   = "meta-llama/Meta-Llama-3-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(base_id)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            base  = AutoModelForCausalLM.from_pretrained(
                base_id, quantization_config=bnb_config, device_map={"": 0})
            model = PeftModel.from_pretrained(
                base, eval_model_id, is_trainable=False, device_map={"": 0})
        else:
            tokenizer = AutoTokenizer.from_pretrained(eval_model_id)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                eval_model_id, quantization_config=bnb_config, device_map={"": 0})
        model.eval()
        print(f">> Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")
    except Exception as e:
        print(f">> FATAL: Could not load model: {e}")
        release_gpu(model, tokenizer); return

    # ── Inference ──────────────────────────────────────────────────────────────
    pred_T, pred_N, pred_M, success_binary = [], [], [], []
    failed = 0

    for _, row in all_rows.iterrows():
        prompt = (
            "You are a clinical data extractor. Read the clinical note and extract the TNM staging. "
            "Return a strictly formatted JSON object with keys 'T', 'N', and 'M'. "
            "Always use the full prefixed format: e.g. 'T2', 'N1', 'M0', 'T1C', 'NX'. "
            "If a value is not found in the note, use 'Unknown'.\n\n"
            f"NOTE: {row['free_text']}"
        )
        messages  = [
            {"role": "system", "content": "You are a helpful clinical JSON extractor."},
            {"role": "user",   "content": prompt},
        ]
        pt, pn, pm = "TUNKNOWN", "NUNKNOWN", "MUNKNOWN"
        try:
            encoded   = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True)
            input_ids = encoded["input_ids"].to("cuda")
            attn_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=input_ids, attention_mask=attn_mask,
                    max_new_tokens=100, temperature=0.1,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            output = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            s, e   = output.find("{"), output.rfind("}") + 1
            if s != -1 and e != -1:
                res = json.loads(output[s:e])
                pt  = normalize_tnm_label(str(res.get("T", "Unknown")), "T")
                pn  = normalize_tnm_label(str(res.get("N", "Unknown")), "N")
                pm  = normalize_tnm_label(str(res.get("M", "Unknown")), "M")
            else:
                failed += 1
                print(f"  [warn] no JSON in: {output[:80]}")
        except Exception as ex:
            failed += 1
            print(f"  [warn] inference error: {ex}")

        pred_T.append(pt); pred_N.append(pn); pred_M.append(pm)
        success_binary.append(int(
            pt == row["T_target"] and pn == row["N_target"] and pm == row["M_target"]))

    if failed > 0:
        print(f"\n>> {failed}/{len(all_rows)} records failed inference.")

    release_gpu(model, tokenizer)

    # ── Metrics ────────────────────────────────────────────────────────────────
    # Map predictions back to all_rows index for subset filtering
    all_rows = all_rows.reset_index(drop=True)
    all_rows["pred_T"] = pred_T
    all_rows["pred_N"] = pred_N
    all_rows["pred_M"] = pred_M
    all_rows["success"] = success_binary

    safe_name = dataset_name.replace(" ", "_").lower()
    metrics_rows = []

    def compute_stage_metrics(targets, preds, stage, n, single_class=False):
        acc          = accuracy_score(targets, preds)
        ci_lo, ci_hi = bootstrap_ci_accuracy(list(targets), list(preds))
        p, r, f1, _  = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0)
        wf1          = precision_recall_fscore_support(
            targets, preds, average="weighted", zero_division=0)[2]

        # Always print constant-classifier baseline for interpretive context
        cc = constant_classifier_baseline(list(targets), stage)

        # FIX 1: Do NOT report macro F1 for single-class sets — it is invalid.
        if single_class:
            print(f"  [{stage}] Accuracy={acc:.3f} (95% CI: [{ci_lo}, {ci_hi}]) (n={n}) "
                  f"[macro F1 omitted — single-class ground truth]")
            return {"stage": stage, "n": n, "accuracy": round(acc, 3),
                    "ci_lower": ci_lo, "ci_upper": ci_hi,
                    "constant_classifier_acc": cc["constant_acc"],
                    "constant_majority_label": cc["majority_label"],
                    "macro_precision": None, "macro_recall": None,
                    "macro_f1": None,
                    "weighted_f1": round(wf1, 3),
                    "note": "single_class_gt"}
        else:
            print(f"  [{stage}] Accuracy={acc:.3f} (95% CI: [{ci_lo}, {ci_hi}]) | "
                  f"Macro P={p:.3f} R={r:.3f} F1={f1:.3f} | "
                  f"Weighted F1={wf1:.3f} (n={n})")
            return {"stage": stage, "n": n, "accuracy": round(acc, 3),
                    "ci_lower": ci_lo, "ci_upper": ci_hi,
                    "constant_classifier_acc": cc["constant_acc"],
                    "constant_majority_label": cc["majority_label"],
                    "macro_precision": round(p, 3), "macro_recall": round(r, 3),
                    "macro_f1": round(f1, 3), "weighted_f1": round(wf1, 3),
                    "note": ""}

    print("\n--- Zero-Shot Extraction Metrics ---")

    # T-stage: use T_valid subset
    t_rows   = all_rows[all_rows.index.isin(t_bench.index)]
    t_sc     = is_single_class_set(t_rows["T_target"])
    metrics_rows.append(compute_stage_metrics(
        t_rows["T_target"], t_rows["pred_T"], "T", len(t_rows), t_sc))

    # Per-class T accuracy (FIX 4)
    t_pcacc = per_class_accuracy(t_rows["T_target"].tolist(), t_rows["pred_T"].tolist(), "T")
    t_pcacc.to_csv(f"{EXPORT_DIR}/{safe_name}_T_per_class_accuracy.csv", index=False)
    print(f"  T per-class accuracy:\n{t_pcacc.to_string(index=False)}")

    # N and M: use NM_valid subset — FIX 2
    nm_rows = all_rows[all_rows.index.isin(nm_bench.index)]
    for stage, col_t, col_p in [("N", "N_target", "pred_N"), ("M", "M_target", "pred_M")]:
        if len(nm_rows) == 0:
            print(f"  [{stage}] Skipped — no records with known {stage} ground truth.")
            metrics_rows.append({"stage": stage, "n": 0, "accuracy": None,
                                  "note": "no_valid_gt"})
            continue
        sc = is_single_class_set(nm_rows[col_t])
        metrics_rows.append(compute_stage_metrics(
            nm_rows[col_t], nm_rows[col_p], stage, len(nm_rows), sc))

    # Export metrics
    out_df = pd.DataFrame(metrics_rows)
    out_df.to_csv(f"{EXPORT_DIR}/table_6_{safe_name}_metrics.csv", index=False)
    print(f"\n>> Saved metrics → table_6_{safe_name}_metrics.csv")

    # Length robustness
    all_rows["extraction_success"] = success_binary
    print("\n--- Length-Robustness ---")
    if all_rows["extraction_success"].nunique() > 1 and all_rows["word_count"].nunique() > 1:
        corr, pv = pointbiserialr(all_rows["word_count"], all_rows["extraction_success"])
        print(f"  Note length vs success: r={corr:.3f}, p={pv:.4f}")
        if pv < 0.05 and corr < 0:
            print("  >> FINDING: Significant negative — model struggles with longer notes.")
        else:
            print("  >> No significant length degradation.")
    else:
        print("  >> Not enough variance to compute.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_file", default=None)
    parser.add_argument("--dataset_label",  default=None)
    args = parser.parse_args()

    if args.real_data_file and not args.dataset_label:
        parser.error("--dataset_label required when --real_data_file is set.")

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # All three adapters for the three-way ablation
    adapters = [
        ("adapters/tier1_raw/final_adapter",      "Adapter A"),
        ("adapters/tier3_scrambled/final_adapter", "Adapter A-prime"),
        ("adapters/tier3_golden/final_adapter",   "Adapter B"),
    ]
    adapters = [(p, n) for p, n in adapters if os.path.exists(p)]

    # Synthetic held-out (always runs)
    df = load_clean_data(RESULTS_FILE)
    if not df.empty:
        test_csv = prepare_extraction_datasets(df)
        calculate_diversity_dcsn(df)
        if test_csv:
            run_extraction_benchmark(
                test_csv, dataset_name="Synthetic Held-Out BASELINE",
                is_real_world=False)
            for adapter_path, adapter_name in adapters:
                run_extraction_benchmark(
                    test_csv, eval_model_id=adapter_path,
                    dataset_name=f"Synthetic Held-Out {adapter_name}",
                    is_real_world=False)

    # Real-world benchmark
    if args.real_data_file:
        if not os.path.exists(args.real_data_file):
            print(f"\n[ERROR] '{args.real_data_file}' not found. Run Pre-Phase3 first.")
            return
        run_extraction_benchmark(
            args.real_data_file,
            dataset_name=f"{args.dataset_label} BASELINE",
            is_real_world=True)
        for adapter_path, adapter_name in adapters:
            run_extraction_benchmark(
                args.real_data_file, eval_model_id=adapter_path,
                dataset_name=f"{args.dataset_label} {adapter_name}",
                is_real_world=True)


if __name__ == "__main__":
    main()
