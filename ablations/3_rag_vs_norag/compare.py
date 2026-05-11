"""
Ablation 3 — RAG vs No-RAG | Comparison Analysis
==================================================
Produces two tables:
  1. Pooled — overall RAG vs No-RAG SNOMED density comparison
  2. Per-model — density lift broken down by model

Mann-Whitney U test on SNOMED density distributions (one-sided: RAG > No-RAG).

Outputs:
  results/analysis/ablation3_rag_vs_norag.csv
  results/analysis/ablation3_rag_vs_norag_by_model.csv
  results/analysis/ablation3_rag_vs_norag_detail.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from core.logging_utils import load_jsonl
from core.tnm_grid      import compute_entropy, ENTROPY_FLOORS

MODEL_SHORT = {
    "gpt-4o":                      "GPT-4o",
    "Llama-3.3-70B-Instruct":      "Llama-3.3-70B",
    "Meta-Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "ClinicalCamel-70B":           "ClinicalCamel-70B",
}


def model_label(model_id: str) -> str:
    short = model_id.split("/")[-1]
    return MODEL_SHORT.get(short, short)


def condition_stats(recs: list, label: str) -> dict:
    n         = len(recs)
    n_valid   = sum(1 for r in recs if r.get("parsed_json_valid"))
    n_gate    = sum(1 for r in recs if r.get("gate_pass"))
    densities = [r.get("snomed_density", 0.0) for r in recs]
    unique_ids = {
        a.get("snomed_id", "")
        for r in recs
        for a in (r.get("snomed_codes") or [])
        if a.get("snomed_id")
    }
    valid = [r for r in recs if r.get("parsed_json_valid")]
    t_lbl = [r.get("T", "Unknown") for r in valid]
    n_lbl = [r.get("N", "Unknown") for r in valid]
    m_lbl = [r.get("M", "Unknown") for r in valid]
    return {
        "condition":              label,
        "n_records":              n,
        "schema_compliance":      round(n_valid / n, 3) if n else 0,
        "gate_pass_rate":         round(n_gate  / n, 3) if n else 0,
        "avg_snomed_per_100w":    round(float(np.mean(densities)), 2) if densities else 0,
        "median_snomed_per_100w": round(float(np.median(densities)), 2) if densities else 0,
        "unique_snomed_concepts": len(unique_ids),
        "T_entropy":              round(compute_entropy(t_lbl), 3),
        "N_entropy":              round(compute_entropy(n_lbl), 3),
        "M_entropy":              round(compute_entropy(m_lbl), 3),
        "_densities":             densities,
    }


def by_model_stats(norag_recs: list, rag_recs: list) -> pd.DataFrame:
    models = sorted({r.get("model", "Unknown")
                     for r in norag_recs + rag_recs})
    rows = []
    for m in models:
        nr = [r for r in norag_recs if r.get("model") == m]
        rg = [r for r in rag_recs   if r.get("model") == m]
        if not nr or not rg:
            continue
        nr_d = [r.get("snomed_density", 0.0) for r in nr]
        rg_d = [r.get("snomed_density", 0.0) for r in rg]
        lift  = (np.mean(rg_d) - np.mean(nr_d)) / max(np.mean(nr_d), 1e-9) * 100
        try:
            stat, p = mannwhitneyu(rg_d, nr_d, alternative="greater")
            sig = p < 0.05
        except Exception:
            stat, p, sig = 0, 1, False
        rows.append({
            "model":               model_label(m),
            "norag_n":             len(nr),
            "rag_n":               len(rg),
            "norag_avg_density":   round(float(np.mean(nr_d)), 2),
            "rag_avg_density":     round(float(np.mean(rg_d)), 2),
            "density_lift_pct":    round(lift, 1),
            "mann_whitney_U":      round(stat, 0),
            "p_value":             round(p, 4),
            "significant":         sig,
            "norag_schema":        round(sum(1 for r in nr if r.get("parsed_json_valid")) / max(len(nr),1), 3),
            "rag_schema":          round(sum(1 for r in rg if r.get("parsed_json_valid")) / max(len(rg),1), 3),
            "norag_gate":          round(sum(1 for r in nr if r.get("gate_pass")) / max(len(nr),1), 3),
            "rag_gate":            round(sum(1 for r in rg if r.get("gate_pass")) / max(len(rg),1), 3),
        })
    return pd.DataFrame(rows)


def print_pooled(norag: dict, rag: dict, stat: float, p: float, lift: float):
    METRICS = [
        ("n_records",              "Records"),
        ("schema_compliance",      "Schema compliance"),
        ("gate_pass_rate",         "Gate pass rate"),
        ("avg_snomed_per_100w",    "Avg SNOMED density (/100w)"),
        ("median_snomed_per_100w", "Median SNOMED density (/100w)"),
        ("unique_snomed_concepts", "Unique SNOMED concepts"),
        ("T_entropy",              f"T entropy  (floor {ENTROPY_FLOORS['T']})"),
        ("N_entropy",              f"N entropy  (floor {ENTROPY_FLOORS['N']})"),
        ("M_entropy",              f"M entropy  (floor {ENTROPY_FLOORS['M']})"),
    ]
    print("\n" + "=" * 78)
    print("ABLATION 3: RAG vs NO-RAG  |  Pooled (all models)")
    print("=" * 78)
    print(f"  {'Metric':<38} {'No-RAG (D)':>14} {'RAG (E)':>14} {'Delta':>10}")
    print("  " + "-" * 80)
    for key, label in METRICS:
        nr = norag.get(key, "-")
        rg = rag.get(key, "-")
        try:
            delta = f"+{rg-nr:.3f}" if rg >= nr else f"{rg-nr:.3f}"
        except TypeError:
            delta = "-"
        print(f"  {label:<38} {str(nr):>14} {str(rg):>14} {delta:>10}")
    print(f"\n  SNOMED density lift: {lift:+.1f}%")
    print(f"  Mann-Whitney U (RAG > No-RAG): U={stat:.0f}, p={p:.4f} "
          f"{'*significant*' if p < 0.05 else '(not significant)'}")
    print("=" * 78)


def print_by_model(df: pd.DataFrame):
    print("\n" + "=" * 78)
    print("ABLATION 3: RAG vs NO-RAG  |  Per-Model Density Lift")
    print("=" * 78)
    print(f"  {'Model':<22} {'No-RAG':>10} {'RAG':>10} {'Lift %':>8} {'p-value':>10} {'Sig':>5}")
    print("  " + "-" * 70)
    for _, row in df.iterrows():
        sig = "*" if row["significant"] else ""
        print(f"  {row['model']:<22} {row['norag_avg_density']:>10.2f} "
              f"{row['rag_avg_density']:>10.2f} {row['density_lift_pct']:>7.1f}% "
              f"{row['p_value']:>10.4f} {sig:>5}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Ablation 3: RAG vs No-RAG comparison")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--export-dir",  default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    norag_path = os.path.join(args.results_dir, "phase1_full_norag.jsonl")
    rag_path   = os.path.join(args.results_dir, "phase1_full_rag.jsonl")

    for path, name in [(norag_path, "run_norag.py"), (rag_path, "run_rag.py")]:
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — run {name} first")
            return

    norag_recs = load_jsonl(norag_path)
    rag_recs   = load_jsonl(rag_path)

    norag = condition_stats(norag_recs, "No-RAG (Adapter D)")
    rag   = condition_stats(rag_recs,   "RAG (Adapter E)")

    stat, p = mannwhitneyu(rag["_densities"], norag["_densities"], alternative="greater")
    lift    = (
        (rag["avg_snomed_per_100w"] - norag["avg_snomed_per_100w"])
        / max(norag["avg_snomed_per_100w"], 1e-9) * 100
    )
    print_pooled(norag, rag, stat, p, lift)

    model_df = by_model_stats(norag_recs, rag_recs)
    if not model_df.empty:
        print_by_model(model_df)

    for d in (norag, rag):
        d.pop("_densities", None)

    pd.DataFrame([norag, rag]).to_csv(
        os.path.join(args.export_dir, "ablation3_rag_vs_norag.csv"), index=False)
    model_df.to_csv(
        os.path.join(args.export_dir, "ablation3_rag_vs_norag_by_model.csv"), index=False)

    # Per-record detail (matched by run_index and model)
    detail = []
    norag_by_key = {(r.get("model",""), r.get("run_index", -1)): r for r in norag_recs}
    rag_by_key   = {(r.get("model",""), r.get("run_index", -1)): r for r in rag_recs}
    all_keys     = sorted(set(norag_by_key) & set(rag_by_key))
    for key in all_keys:
        nr = norag_by_key[key]
        rg = rag_by_key[key]
        detail.append({
            "model":         model_label(nr.get("model", "")),
            "run_index":     nr.get("run_index"),
            "T_target":      nr.get("T_target"),
            "N_target":      nr.get("N_target"),
            "M_target":      nr.get("M_target"),
            "norag_density": nr.get("snomed_density"),
            "rag_density":   rg.get("snomed_density"),
            "density_delta": round((rg.get("snomed_density") or 0)
                                   - (nr.get("snomed_density") or 0), 2),
            "norag_gate":    nr.get("gate_pass"),
            "rag_gate":      rg.get("gate_pass"),
            "rag_retriever": rg.get("rag_retriever"),
        })
    pd.DataFrame(detail).to_csv(
        os.path.join(args.export_dir, "ablation3_rag_vs_norag_detail.csv"), index=False)

    print(f"\nSaved -> {args.export_dir}")


if __name__ == "__main__":
    main()
