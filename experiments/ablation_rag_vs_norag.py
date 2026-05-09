"""
experiments/ablation_rag_vs_norag.py
Ablation: RAG-Grounded vs Non-RAG Generation

Compares records generated with MedCPT RAG grounding against those
generated without retrieval context. The paper reports an 18.4% SNOMED
density advantage for RAG; this script formalises and extends that
comparison across multiple quality dimensions.

Condition assignment uses the 'prompt_type' field logged by phase1_datagen.py:
  - RAG records:    prompt_type contains 'rag'
  - Non-RAG records: all others in the controlled layer

If your run predates the prompt_type field, pass --infer-from-run-id to
split by run order (first N records = non-RAG, rest = RAG).

Metrics:
  - SNOMED CT density (terms / 100 words)
  - Unique SNOMED concept count
  - Schema compliance rate
  - Label diversity (T / N / M entropy)
  - Retrieval concentration: unique PMIDs per N records (if logged)
  - Statistical test: Mann-Whitney U on SNOMED density distributions
"""

import os
import json
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, entropy as shannon_entropy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.bioportal import snomed_density

RESULTS_FILE = "results/master_results.jsonl"
EXPORT_DIR   = "results/analysis_exports"


def load_records(filepath: str) -> list:
    if not os.path.exists(filepath):
        print(f"[ERROR] {filepath} not found. Run phase1_datagen.py first.")
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON on line {i}: {e}")
    return records


def assign_rag_condition(records: list, infer_from_run_id: bool = False,
                         norag_cutoff: int = 192) -> list:
    """
    Tags each record with rag_condition = 'RAG' | 'No-RAG'.

    Primary method: reads 'rag_grounded' boolean field (set by updated phase1).
    Fallback: checks if prompt_type contains 'rag'.
    Inference mode: first norag_cutoff controlled records = No-RAG.
    """
    controlled = [r for r in records if r.get("layer") == "controlled"]
    for i, r in enumerate(controlled):
        if "rag_grounded" in r:
            r["_rag_condition"] = "RAG" if r["rag_grounded"] else "No-RAG"
        elif "rag" in str(r.get("prompt_type", "")).lower():
            r["_rag_condition"] = "RAG"
        elif infer_from_run_id:
            r["_rag_condition"] = "No-RAG" if i < norag_cutoff else "RAG"
        else:
            r["_rag_condition"] = "RAG"  # default assumption for newer runs
    return controlled


def compute_entropy(labels: list) -> float:
    if not labels:
        return 0.0
    counts = pd.Series(labels).value_counts()
    return float(shannon_entropy(counts)) if len(counts) > 1 else 0.0


def extract_tnm(record: dict) -> tuple:
    try:
        pj  = record.get("parsed_json") or {}
        stg = pj.get("notes", [{}])[0].get("staging", {})
        return (
            str(stg.get("T", "Unknown")).upper(),
            str(stg.get("N", "Unknown")).upper(),
            str(stg.get("M", "Unknown")).upper(),
        )
    except (IndexError, AttributeError, TypeError):
        return ("Unknown", "Unknown", "Unknown")


def analyze_condition(records: list, condition_label: str) -> dict:
    n_total  = len(records)
    n_valid  = sum(1 for r in records if r.get("parsed_json_valid"))

    snomed_vals   = [
        snomed_density(r.get("raw_output", ""), r.get("snomed_codes") or [])
        for r in records
    ]
    unique_concepts = len({
        a.get("snomed_id", "")
        for r in records
        for a in (r.get("snomed_codes") or [])
        if a.get("snomed_id")
    })

    valid_records = [r for r in records if r.get("parsed_json_valid")]
    t_labels = [extract_tnm(r)[0] for r in valid_records]
    n_labels = [extract_tnm(r)[1] for r in valid_records]
    m_labels = [extract_tnm(r)[2] for r in valid_records]

    return {
        "condition":              condition_label,
        "n_records":              n_total,
        "schema_compliance":      round(n_valid / n_total, 3) if n_total else 0.0,
        "avg_snomed_per_100w":    round(float(np.mean(snomed_vals)), 2) if snomed_vals else 0.0,
        "median_snomed_per_100w": round(float(np.median(snomed_vals)), 2) if snomed_vals else 0.0,
        "unique_snomed_concepts": unique_concepts,
        "T_entropy":              round(compute_entropy(t_labels), 3),
        "N_entropy":              round(compute_entropy(n_labels), 3),
        "M_entropy":              round(compute_entropy(m_labels), 3),
        "_snomed_vals":           snomed_vals,  # kept for statistical test; dropped on export
    }


def run_ablation(records: list, infer: bool = False, norag_cutoff: int = 192) -> pd.DataFrame:
    controlled = assign_rag_condition(records, infer_from_run_id=infer,
                                      norag_cutoff=norag_cutoff)

    rag_records   = [r for r in controlled if r.get("_rag_condition") == "RAG"]
    norag_records = [r for r in controlled if r.get("_rag_condition") == "No-RAG"]

    if not rag_records or not norag_records:
        print("[WARN] Could not split records into RAG / No-RAG conditions.")
        print("       Re-run with --infer-from-run-id or ensure 'rag_grounded' field is logged.")
        return pd.DataFrame()

    rag_stats   = analyze_condition(rag_records,   "RAG-Grounded")
    norag_stats = analyze_condition(norag_records, "No-RAG")

    # Mann-Whitney U test on SNOMED density distributions
    stat, p = mannwhitneyu(
        rag_stats["_snomed_vals"], norag_stats["_snomed_vals"],
        alternative="greater",
    )

    print("\n" + "=" * 70)
    print("ABLATION: RAG-GROUNDED vs NON-RAG GENERATION (Controlled Layer)")
    print("=" * 70)
    metrics = [
        ("n_records",              "Records"),
        ("schema_compliance",      "Schema compliance"),
        ("avg_snomed_per_100w",    "Avg SNOMED density (/100w)"),
        ("median_snomed_per_100w", "Median SNOMED density (/100w)"),
        ("unique_snomed_concepts", "Unique SNOMED concepts"),
        ("T_entropy",              "T-stage entropy"),
        ("N_entropy",              "N-stage entropy"),
        ("M_entropy",              "M-stage entropy"),
    ]
    header = f"  {'Metric':<35} {'No-RAG':>12} {'RAG':>12} {'Δ':>10}"
    print(header)
    print("  " + "-" * 73)
    for key, label in metrics:
        nr = norag_stats.get(key, "—")
        rg = rag_stats.get(key, "—")
        try:
            delta = f"+{rg - nr:.3f}" if rg > nr else f"{rg - nr:.3f}"
        except TypeError:
            delta = "—"
        print(f"  {label:<35} {str(nr):>12} {str(rg):>12} {delta:>10}")

    density_lift = (
        (rag_stats["avg_snomed_per_100w"] - norag_stats["avg_snomed_per_100w"])
        / max(norag_stats["avg_snomed_per_100w"], 1e-9) * 100
    )
    print(f"\n  SNOMED density lift (RAG over No-RAG): {density_lift:+.1f}%")
    print(f"  Mann-Whitney U (one-sided, RAG > No-RAG): U={stat:.0f}, p={p:.4f} "
          f"{'*significant*' if p < 0.05 else '(not significant)'}")
    print("=" * 70)

    # Clean up internal field before export
    for d in (rag_stats, norag_stats):
        d.pop("_snomed_vals", None)

    out_df = pd.DataFrame([norag_stats, rag_stats])
    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: RAG-grounded vs Non-RAG generation"
    )
    parser.add_argument(
        "--infer-from-run-id", action="store_true",
        help="Split by run order: first --norag-cutoff records = No-RAG, rest = RAG."
    )
    parser.add_argument(
        "--norag-cutoff", type=int, default=192,
        help="Number of non-RAG records (default: 192, matching paper's run structure)."
    )
    args = parser.parse_args()

    os.makedirs(EXPORT_DIR, exist_ok=True)
    records = load_records(RESULTS_FILE)
    if not records:
        return

    print(f"Loaded {len(records)} records from {RESULTS_FILE}")

    df = run_ablation(records, infer=args.infer_from_run_id,
                      norag_cutoff=args.norag_cutoff)
    if not df.empty:
        path = f"{EXPORT_DIR}/ablation_rag_vs_norag.csv"
        df.to_csv(path, index=False)
        print(f"\nExported → {path}")


if __name__ == "__main__":
    main()
