#!/usr/bin/env python3
"""
STEP_6b_Phase3_benchmark.py
────────────────────────────
Runs the complete TSTR benchmark:
  • Synthetic held-out (in-distribution evaluation — NOT labelled as TSTR)
  • TCGA Lung       (genuine TSTR , TNM stages)
  • TCGA Cross-tumor (genuine TSTR, TNM stages)

For each dataset, evaluates:
  • Zero-shot Llama-3-8B-Instruct (baseline)
  • Adapter A (Tier 1 Raw — degenerate prior)
  • Adapter B (Tier 3 Golden — corrected corpus)

Run it three times (once per dataset), or combine into a shell script — see bottom.
"""

import os
os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]    = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

import gc, json, argparse, re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report
)
from scipy.stats import pointbiserialr, bootstrap as scipy_bootstrap
from peft import PeftModel

RESULTS_FILE  = "results/master_results.jsonl"
OUT_DIR       = "data_splits"
EXPORT_DIR    = "results/analysis_exports"
BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

ADAPTER_PATHS = {
    "Baseline":  None,                             # zero-shot, no adapter
    "Adapter_A": "adapters/tier1_raw/final_adapter",
    "Adapter_B": "adapters/tier3_golden/final_adapter",
}


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def normalise_tnm(value: str, prefix: str) -> str:
    v = str(value).strip().upper()
    if not v or v in ("UNKNOWN","NONE","NULL","NAN",""): return "UNKNOWN"
    p = prefix.upper()
    if v.startswith(p): v = v[len(p):]
    m = re.match(r"^(\d+|IS|X)", v)
    if m: return p + m.group(1)
    return p + v


# ─────────────────────────────────────────────────────────────────────────────
# METRICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(targets, preds, n=1000, conf=0.95):
    correct = (np.array(targets) == np.array(preds)).astype(float)
    try:
        res = scipy_bootstrap(
            (correct,), np.mean, n_resamples=n,
            confidence_level=conf, method="percentile", random_state=42,
        )
        return round(res.confidence_interval.low,3), round(res.confidence_interval.high,3)
    except Exception:
        # Wilson fallback
        p = float(np.mean(correct)); z = 1.96; nn = len(correct)
        denom = 1 + z**2/nn
        centre = (p + z**2/(2*nn)) / denom
        margin = (z * np.sqrt(p*(1-p)/nn + z**2/(4*nn**2))) / denom
        return round(centre-margin,3), round(centre+margin,3)


def constant_classifier_acc(targets):
    from collections import Counter
    majority = Counter(targets).most_common(1)[0][0]
    return accuracy_score(targets, [majority]*len(targets)), majority


def per_class_accuracy(targets, preds):
    rows = []
    for cls in sorted(set(targets)):
        mask = [t == cls for t in targets]
        n    = sum(mask)
        corr = sum(p == cls for t, p in zip(targets, preds) if t == cls)
        rows.append({"class":cls,"n_gt":n,"n_correct":corr,
                     "per_class_acc":round(corr/n,3) if n>0 else 0.0})
    return pd.DataFrame(rows)


def is_single_class(series):
    vals = [v for v in series if "UNKNOWN" not in str(v).upper()]
    return len(set(vals)) <= 1


def compute_metrics(targets, preds, stage, n, export_tag, single_class=False):
    acc          = accuracy_score(targets, preds)
    ci_lo, ci_hi = bootstrap_ci(list(targets), list(preds))
    cc_acc, cc_lbl = constant_classifier_acc(list(targets))
    p, r, f1, _  = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0)
    wf1          = precision_recall_fscore_support(
        targets, preds, average="weighted", zero_division=0)[2]

    print(f"\n  [{stage}] n={n}  Acc={acc:.3f}  95%CI=[{ci_lo},{ci_hi}]  "
          f"ConstClf={cc_acc:.3f}('{cc_lbl}')")
    if single_class:
        print(f"         Macro F1 OMITTED — single-class ground truth")
    else:
        print(f"         MacroP={p:.3f}  MacroR={r:.3f}  MacroF1={f1:.3f}  WtdF1={wf1:.3f}")

    # Per-class breakdown
    pc = per_class_accuracy(list(targets), list(preds))
    print(f"         Per-class:\n{pc.to_string(index=False)}")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    pc.to_csv(f"{EXPORT_DIR}/{export_tag}_{stage}_per_class.csv", index=False)

    return {
        "stage": stage, "n": n,
        "accuracy": round(acc,3), "ci_lower": ci_lo, "ci_upper": ci_hi,
        "constant_clf_acc": round(cc_acc,3), "constant_majority_label": cc_lbl,
        "macro_precision": None if single_class else round(p,3),
        "macro_recall":    None if single_class else round(r,3),
        "macro_f1":        None if single_class else round(f1,3),
        "weighted_f1":     round(wf1,3),
        "note": "single_class_gt" if single_class else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPU HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def release_gpu(model=None, tokenizer=None):
    if model:    del model
    if tokenizer: del tokenizer
    gc.collect(); gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free = torch.cuda.mem_get_info()[0]/1e9
        print(f">> GPU freed. Free: {free:.1f}GB")


def load_model(adapter_path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb, device_map={"":0})
    if adapter_path and os.path.exists(
            os.path.join(adapter_path,"adapter_config.json")):
        model = PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False, device_map={"":0})
        print(f">> Adapter loaded: {adapter_path}")
    else:
        model = base
        print(">> Zero-shot baseline (no adapter)")
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_synthetic_test():
    path = f"{OUT_DIR}/test_heldout_eval.csv"
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run Step 4 first."); return None
    df = pd.read_csv(path)
    for col, pfx in [("T_target","T"),("N_target","N"),("M_target","M")]:
        df[col] = df[col].fillna("Unknown").astype(str).apply(
            lambda v: normalise_tnm(v, pfx))
    return df


def load_mtsamples_test(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run Step 6a first."); return None
    df = pd.read_csv(path)
    for col, pfx in [("T_target","T"),("N_target","N"),("M_target","M")]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).apply(
                lambda v: normalise_tnm(v, pfx))
    # For MTSamples: only evaluate on records where T is explicitly known
    t_valid = df[~df["T_target"].str.contains("UNKNOWN", case=False, na=False)].copy()
    print(f"  MTSamples: {len(df)} total, {len(t_valid)} with T-stage ground truth")
    # Sample to 100 if larger
    if len(t_valid) > 100:
        t_valid = t_valid.sample(100, random_state=42).reset_index(drop=True)
        print(f"  Sampled to 100 for inference.")
    return t_valid


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(df, model, tokenizer):
    pred_T, pred_N, pred_M = [], [], []
    failed = 0
    for _, row in df.iterrows():
        prompt = (
            "You are a clinical data extractor. Read the clinical note and extract "
            "the TNM staging. Return a strictly formatted JSON object with keys "
            "'T', 'N', and 'M'. Always use the full prefixed format: e.g. 'T2', 'N1', 'M0'. "
            "If a value is not found in the note, use 'Unknown'.\n\n"
            f"NOTE: {row['free_text']}"
        )
        messages = [
            {"role":"system","content":"You are a helpful clinical JSON extractor."},
            {"role":"user",  "content":prompt},
        ]
        pt, pn, pm = "TUNKNOWN","NUNKNOWN","MUNKNOWN"
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
            output = tokenizer.decode(out_ids[0][input_ids.shape[1]:],
                                      skip_special_tokens=True)
            s, e = output.find("{"), output.rfind("}")+1
            if s != -1 and e > s:
                res = json.loads(output[s:e])
                pt  = normalise_tnm(str(res.get("T","Unknown")), "T")
                pn  = normalise_tnm(str(res.get("N","Unknown")), "N")
                pm  = normalise_tnm(str(res.get("M","Unknown")), "M")
            else:
                failed += 1
        except Exception as ex:
            failed += 1
            print(f"  [warn] {ex}")
        pred_T.append(pt); pred_N.append(pn); pred_M.append(pm)

    if failed > 0:
        print(f"\n  {failed}/{len(df)} records failed inference.")
    return pred_T, pred_N, pred_M


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(df, adapter_name, adapter_path,
                  dataset_name, t_only=False):
    """
    t_only=True  → only evaluate T-stage (used for MTSamples where N/M GT is sparse)
    t_only=False → evaluate all three stages (used for synthetic held-out)
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {dataset_name.upper()} — {adapter_name}")
    print(f"{'='*60}")

    model, tokenizer = load_model(adapter_path)
    pred_T, pred_N, pred_M = run_inference(df, model, tokenizer)
    release_gpu(model, tokenizer)

    safe_tag = f"{dataset_name.replace(' ','_').lower()}_{adapter_name.lower()}"
    metrics  = []

    # T-stage (always evaluated)
    sc_T = is_single_class(df["T_target"])
    metrics.append(compute_metrics(
        df["T_target"].tolist(), pred_T, "T", len(df), safe_tag, sc_T))

    if not t_only:
        # N and M (synthetic held-out only)
        sc_N = is_single_class(df["N_target"])
        sc_M = is_single_class(df["M_target"])
        metrics.append(compute_metrics(
            df["N_target"].tolist(), pred_N, "N", len(df), safe_tag, sc_N))
        metrics.append(compute_metrics(
            df["M_target"].tolist(), pred_M, "M", len(df), safe_tag, sc_M))

    # Length robustness
    if "word_count" in df.columns and df["word_count"].nunique() > 1:
        success = [int(t==g) for t,g in zip(pred_T, df["T_target"].tolist())]
        if len(set(success)) > 1:
            corr, pv = pointbiserialr(df["word_count"].tolist(), success)
            print(f"\n  Length robustness: r={corr:.3f}, p={pv:.4f} "
                  f"({'significant' if pv<0.05 else 'not significant'})")

    out = pd.DataFrame(metrics)
    out.to_csv(f"{EXPORT_DIR}/metrics_{safe_tag}.csv", index=False)
    print(f"\n>> Saved → metrics_{safe_tag}.csv")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data_file", default=None,
                        help="Path to MTSamples gold CSV (from Step 6a)")
    parser.add_argument("--dataset_label",  default=None,
                        help="Label for the real-world dataset, e.g. 'MTSamples Lung'")
    args = parser.parse_args()

    if args.real_data_file and not args.dataset_label:
        parser.error("--dataset_label required when --real_data_file is set.")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    all_results = []

    # ── 1. Synthetic held-out (in-distribution) ────────────────────────────
    synth_df = load_synthetic_test()
    if synth_df is not None:
        print(f"\n{'='*60}")
        print("SYNTHETIC HELD-OUT EVALUATION (in-distribution, NOT TSTR)")
        print(f"n={len(synth_df)}")
        print(f"{'='*60}")
        for name, path in ADAPTER_PATHS.items():
            if path and not os.path.exists(
                    os.path.join(path, "adapter_config.json")):
                print(f"\n[SKIP] {name}: adapter not found at {path}")
                continue
            res = run_benchmark(synth_df, name, path,
                                f"Synthetic_HeldOut", t_only=False)
            res["adapter"] = name
            res["dataset"] = "Synthetic Held-Out"
            all_results.append(res)

    # ── 2. Real-world TSTR (MTSamples) ────────────────────────────────────
    if args.real_data_file:
        mt_df = load_mtsamples_test(args.real_data_file)
        if mt_df is not None:
            label = args.dataset_label
            print(f"\n{'='*60}")
            print(f"REAL-WORLD TSTR: {label.upper()} (T-stage only)")
            print(f"n={len(mt_df)}")
            print(f"{'='*60}")
            for name, path in ADAPTER_PATHS.items():
                if path and not os.path.exists(
                        os.path.join(path,"adapter_config.json")):
                    print(f"\n[SKIP] {name}: adapter not found at {path}")
                    continue
                res = run_benchmark(mt_df, name, path,
                                    label.replace(" ","_"), t_only=True)
                res["adapter"] = name
                res["dataset"] = label
                all_results.append(res)

    # ── Summary ────────────────────────────────────────────────────────────
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(f"{EXPORT_DIR}/all_benchmark_results.csv", index=False)
        print(f"\n>> All results saved → {EXPORT_DIR}/all_benchmark_results.csv")

    print("\n" + "="*60)
    print("Step 6b complete.")
    print("Update paper with these three result sets:")
    print("  1. Synthetic Held-Out       → in-distribution evaluation")
    print("  2. MTSamples Lung T-stage   → genuine TSTR (real-world)")
    print("  3. MTSamples All-Cancer T   → genuine TSTR (cross-cancer)")
    print("="*60)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: run all three benchmarks in sequence
# Save as run_all_benchmarks.sh and execute with: bash run_all_benchmarks.sh
# ─────────────────────────────────────────────────────────────────────────────
SHELL_SCRIPT = """#!/bin/bash
# Run all three TSTR benchmarks in sequence on GPU 1
set -e
CUDA_VISIBLE_DEVICES=1 python STEP_6b_Phase3_benchmark.py

CUDA_VISIBLE_DEVICES=1 python STEP_6b_Phase3_benchmark.py \\
    --real_data_file mtsamples_lung_gold.csv \\
    --dataset_label "MTSamples Lung"

CUDA_VISIBLE_DEVICES=1 python STEP_6b_Phase3_benchmark.py \\
    --real_data_file mtsamples_all_cancer_gold.csv \\
    --dataset_label "MTSamples All-Cancer"

echo "All benchmarks complete."
"""

if __name__ == "__main__":
    with open("run_all_benchmarks.sh", "w") as f:
        f.write(SHELL_SCRIPT)
    print("Shell script written: run_all_benchmarks.sh")
