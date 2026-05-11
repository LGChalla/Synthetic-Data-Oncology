"""
Ablation 1 — Gate vs No-Gate | Comparison Analysis
====================================================
Produces two tables:
  1. Pooled (all models combined) — overall gate effect
  2. Per-model breakdown — shows which models drive the rejection story

Run AFTER both generation scripts have completed for all models.

Outputs:
  results/analysis/ablation1_gate_vs_nogate.csv
  results/analysis/ablation1_gate_vs_nogate_by_model.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config as cfg
from core.logging_utils import load_jsonl
from core.tnm_grid      import compute_entropy, ENTROPY_FLOORS

CONDITIONS = {
    "ungated":    "Adapter A — Ungated",
    "full_norag": "Adapter D — Full Gate",
}

MODEL_SHORT = {
    "gpt-4o":                    "GPT-4o",
    "Llama-3.3-70B-Instruct":    "Llama-3.3-70B",
    "Meta-Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "ClinicalCamel-70B":         "ClinicalCamel-70B",
}


def model_label(model_id: str) -> str:
    short = model_id.split("/")[-1]
    return MODEL_SHORT.get(short, short)


def summarize(recs: list, label: str) -> dict:
    admitted   = [r for r in recs if r.get("admitted")]
    n_total    = len(recs)
    n_admitted = len(admitted)
    densities  = [r.get("snomed_density", 0.0) for r in recs]
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
        "condition":           label,
        "total_generated":     n_total,
        "corpus_yield":        round(n_admitted / n_total, 3) if n_total else 0,
        "schema_compliance":   round(sum(1 for r in recs if r.get("gate_schema"))   / max(n_total, 1), 3),
        "ontology_coverage":   round(sum(1 for r in recs if r.get("gate_ontology")) / max(n_total, 1), 3),
        "logic_pass_rate":     round(sum(1 for r in recs if r.get("gate_logic"))    / max(n_total, 1), 3),
        "avg_snomed_per_100w": round(float(np.mean(densities)), 2) if densities else 0,
        "unique_snomed":       len(unique_ids),
        "T_entropy":           round(compute_entropy(t_lbl), 3),
        "N_entropy":           round(compute_entropy(n_lbl), 3),
        "M_entropy":           round(compute_entropy(m_lbl), 3),
        "T_floor_pass":        compute_entropy(t_lbl) >= ENTROPY_FLOORS["T"],
        "N_floor_pass":        compute_entropy(n_lbl) >= ENTROPY_FLOORS["N"],
        "M_floor_pass":        compute_entropy(m_lbl) >= ENTROPY_FLOORS["M"],
    }


def by_model_rows(recs: list, cond_label: str) -> list:
    models = sorted({r.get("model", "Unknown") for r in recs})
    rows = []
    for m in models:
        m_recs = [r for r in recs if r.get("model") == m]
        row = summarize(m_recs, cond_label)
        row["model"] = model_label(m)
        rows.append(row)
    return rows


METRICS = [
    ("total_generated",     "Records generated"),
    ("corpus_yield",        "Corpus yield"),
    ("schema_compliance",   "Schema compliance  (C1)"),
    ("ontology_coverage",   "Ontology coverage  (C2)"),
    ("logic_pass_rate",     "AJCC logic pass    (C3)"),
    ("avg_snomed_per_100w", "SNOMED density (/100w)"),
    ("unique_snomed",       "Unique SNOMED concepts"),
    ("T_entropy",           f"T entropy  (floor {ENTROPY_FLOORS['T']})"),
    ("N_entropy",           f"N entropy  (floor {ENTROPY_FLOORS['N']})"),
    ("M_entropy",           f"M entropy  (floor {ENTROPY_FLOORS['M']})"),
    ("T_floor_pass",        "T diversity PASS"),
    ("N_floor_pass",        "N diversity PASS"),
    ("M_floor_pass",        "M diversity PASS"),
]


def print_pooled(rows: list):
    by_cond = {r["condition"]: r for r in rows}
    labels  = list(by_cond.keys())
    print("\n" + "=" * 80)
    print("ABLATION 1: GATE vs NO-GATE  |  Pooled (all models)")
    print("=" * 80)
    print(f"  {'Metric':<38}" + "".join(f" {l[:18]:>18}" for l in labels))
    print("  " + "-" * 78)
    for key, label in METRICS:
        line = f"  {label:<38}"
        for lbl in labels:
            line += f" {str(by_cond[lbl].get(key, '-')):>18}"
        print(line)
    print("=" * 80)


def print_by_model(all_model_rows: dict):
    print("\n" + "=" * 80)
    print("ABLATION 1: GATE vs NO-GATE  |  Per-Model Breakdown")
    print("=" * 80)
    for cond_label, rows in all_model_rows.items():
        if not rows:
            continue
        models = [r["model"] for r in rows]
        print(f"\n  {cond_label}")
        print(f"  {'Metric':<38}" + "".join(f" {m[:16]:>16}" for m in models))
        print("  " + "-" * (38 + 17 * len(models)))
        for key, label in METRICS:
            line = f"  {label:<38}"
            for row in rows:
                line += f" {str(row.get(key, '-')):>16}"
            print(line)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Ablation 1: Gate vs No-Gate comparison")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--export-dir",  default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    pooled_rows    = []
    all_model_rows = {}

    for cond, label in CONDITIONS.items():
        path = os.path.join(args.results_dir, f"phase1_{cond}.jsonl")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        recs = load_jsonl(path)
        pooled_rows.append(summarize(recs, label))
        all_model_rows[label] = by_model_rows(recs, label)

    if not pooled_rows:
        print("No data found.")
        return

    print_pooled(pooled_rows)
    print_by_model(all_model_rows)

    pd.DataFrame(pooled_rows).to_csv(
        os.path.join(args.export_dir, "ablation1_gate_vs_nogate.csv"), index=False)

    model_rows_flat = [
        {**row, "condition": cond}
        for cond, rows in all_model_rows.items()
        for row in rows
    ]
    pd.DataFrame(model_rows_flat).to_csv(
        os.path.join(args.export_dir, "ablation1_gate_vs_nogate_by_model.csv"), index=False)

    print(f"\nSaved -> {args.export_dir}")


if __name__ == "__main__":
    main()
