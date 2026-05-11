"""
Ablation 2 — Gate Decomposition | Comparison Analysis
======================================================
Produces two tables:
  1. Pooled — sequential gate tightening across all models combined
  2. Per-model — shows which model drives rejection at each checkpoint

The per-model table is the key result: ClinicalCamel's low C1 compliance
means the schema gate is doing real work when that model is included.

Outputs:
  results/analysis/ablation2_gate_decomposition.csv
  results/analysis/ablation2_gate_decomposition_by_model.csv
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

CONDITIONS = [
    ("schema_only", "B — Schema (C1)"),
    ("schema_onto", "C — Schema + Ontology (C2)"),
    ("full_norag",  "D — Full G(x) (C3)"),
]

MODEL_SHORT = {
    "gpt-4o":                      "GPT-4o",
    "Llama-3.3-70B-Instruct":      "Llama-3.3-70B",
    "Meta-Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "ClinicalCamel-70B":           "ClinicalCamel-70B",
}


def model_label(model_id: str) -> str:
    short = model_id.split("/")[-1]
    return MODEL_SHORT.get(short, short)


def checkpoint_stats(recs: list, label: str, n_prev: int) -> dict:
    admitted   = [r for r in recs if r.get("admitted")]
    n_total    = len(recs)
    n_admitted = len(admitted)
    densities  = [r.get("snomed_density", 0.0) for r in admitted]
    unique_ids = {
        a.get("snomed_id", "")
        for r in admitted
        for a in (r.get("snomed_codes") or [])
        if a.get("snomed_id")
    }
    valid = [r for r in admitted if r.get("parsed_json_valid")]
    t_lbl = [r.get("T", "Unknown") for r in valid]
    n_lbl = [r.get("N", "Unknown") for r in valid]
    m_lbl = [r.get("M", "Unknown") for r in valid]
    return {
        "checkpoint":          label,
        "total_generated":     n_total,
        "n_admitted":          n_admitted,
        "corpus_yield":        round(n_admitted / n_total, 3) if n_total else 0,
        "rejected_by_gate":    (n_prev - n_admitted) if n_prev is not None else "-",
        "schema_pass_rate":    round(sum(1 for r in recs if r.get("gate_schema"))   / max(n_total, 1), 3),
        "onto_pass_rate":      round(sum(1 for r in recs if r.get("gate_ontology")) / max(n_total, 1), 3),
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


def by_model_stats(recs: list, label: str) -> list:
    models = sorted({r.get("model", "Unknown") for r in recs})
    rows = []
    for m in models:
        m_recs = [r for r in recs if r.get("model") == m]
        row = checkpoint_stats(m_recs, label, None)
        row["model"] = model_label(m)
        rows.append(row)
    return rows


def print_pooled(rows: list):
    print("\n" + "=" * 88)
    print("ABLATION 2: GATE DECOMPOSITION  |  Pooled (all models)")
    print("=" * 88)
    print(f"\n  {'Checkpoint':<32} {'Total':>7} {'Admitted':>9} {'Yield':>7} "
          f"{'Rejected':>9} {'Schema':>7} {'Onto':>7} {'Logic':>7} {'SNOMED/100w':>12}")
    print("  " + "-" * 86)
    for r in rows:
        print(f"  {r['checkpoint']:<32} {r['total_generated']:>7} {r['n_admitted']:>9} "
              f"{r['corpus_yield']:>7.1%} {str(r['rejected_by_gate']):>9} "
              f"{r['schema_pass_rate']:>7.3f} {r['onto_pass_rate']:>7.3f} "
              f"{r['logic_pass_rate']:>7.3f} {r['avg_snomed_per_100w']:>12.2f}")

    print(f"\n  Entropy  (floors: T/N >= {ENTROPY_FLOORS['T']}, M >= {ENTROPY_FLOORS['M']})")
    print(f"  {'Checkpoint':<32} {'T':>8} {'':>5} {'N':>8} {'':>5} {'M':>8} {'':>5}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['checkpoint']:<32} "
              f"{r['T_entropy']:>8.3f} {'PASS' if r['T_floor_pass'] else 'FAIL':>5} "
              f"{r['N_entropy']:>8.3f} {'PASS' if r['N_floor_pass'] else 'FAIL':>5} "
              f"{r['M_entropy']:>8.3f} {'PASS' if r['M_floor_pass'] else 'FAIL':>5}")
    print("=" * 88)


def print_by_model(all_rows: list):
    print("\n" + "=" * 88)
    print("ABLATION 2: GATE DECOMPOSITION  |  Per-Model Schema Compliance")
    print("=" * 88)
    checkpoints = sorted({r["checkpoint"] for r in all_rows})
    models      = sorted({r["model"]      for r in all_rows})

    print(f"\n  {'Checkpoint':<32}" + "".join(f" {m[:18]:>18}" for m in models))
    print("  " + "-" * (32 + 19 * len(models)))

    for metric_key, metric_label in [
        ("schema_pass_rate", "Schema compliance (C1)"),
        ("onto_pass_rate",   "Ontology coverage (C2)"),
        ("logic_pass_rate",  "AJCC logic pass   (C3)"),
        ("corpus_yield",     "Corpus yield"),
        ("n_admitted",       "Records admitted"),
    ]:
        print(f"\n  {metric_label}")
        for cp in checkpoints:
            cp_rows = {r["model"]: r for r in all_rows if r["checkpoint"] == cp}
            line = f"    {cp:<30}"
            for m in models:
                val = cp_rows.get(m, {}).get(metric_key, "-")
                line += f" {str(val):>18}"
            print(line)
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description="Ablation 2: Gate decomposition comparison")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--export-dir",  default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)

    pooled_rows, by_model_rows_all = [], []
    prev_admitted = None

    for cond, label in CONDITIONS:
        path = os.path.join(args.results_dir, f"phase1_{cond}.jsonl")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found — run run_{cond}.py first")
            continue
        recs = load_jsonl(path)

        row = checkpoint_stats(recs, label, prev_admitted)
        prev_admitted = row["n_admitted"]
        pooled_rows.append(row)

        for model_row in by_model_stats(recs, label):
            by_model_rows_all.append(model_row)

    if not pooled_rows:
        print("No data found.")
        return

    print_pooled(pooled_rows)
    if by_model_rows_all:
        print_by_model(by_model_rows_all)

    pd.DataFrame(pooled_rows).to_csv(
        os.path.join(args.export_dir, "ablation2_gate_decomposition.csv"), index=False)
    pd.DataFrame(by_model_rows_all).to_csv(
        os.path.join(args.export_dir, "ablation2_gate_decomposition_by_model.csv"), index=False)

    print(f"\nSaved -> {args.export_dir}")


if __name__ == "__main__":
    main()
