"""
Phase 2 — Statistical Quality Audit
=====================================
Loads one or more Phase 1 JSONL files and reports quality across all
ablation conditions side by side. This is the primary diagnostic phase —
it tells you what each gate component and RAG grounding actually buys.

Modules:
  1. Schema compliance     — JSON validity rate per condition
  2. Gate decomposition    — C0/C1/C2/C3 pass rates per condition
  3. Diversity audit       — Shannon entropy per T/N/M per condition
  4. SNOMED density        — length-normalised ontology coverage per condition
  5. AJCC violations       — logic error breakdown per condition
  6. Cross-condition table — all metrics in one comparative table

Outputs:
  results/analysis/phase2_schema_compliance.csv
  results/analysis/phase2_gate_decomposition.csv
  results/analysis/phase2_diversity.csv
  results/analysis/phase2_snomed_density.csv
  results/analysis/phase2_ajcc_violations.csv
  results/analysis/phase2_summary_table.csv

Usage:
  python phases/phase2_audit.py
  python phases/phase2_audit.py --results-dir /path/to/results
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy as shannon_entropy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from core.logging_utils import load_jsonl
from core.tnm_grid      import compute_entropy, ENTROPY_FLOORS

VALID_CONDITIONS = ["ungated","schema_only","schema_onto","full_norag","full_rag"]


def load_all(results_dir: str) -> dict:
    """Loads all Phase 1 JSONL files that exist."""
    data = {}
    for cond in VALID_CONDITIONS:
        path = os.path.join(results_dir, f"phase1_{cond}.jsonl")
        if os.path.exists(path):
            recs = load_jsonl(path)
            data[cond] = recs
            print(f"  Loaded {len(recs):>4} records  [{cond}]")
        else:
            print(f"  [SKIP] {path} not found")
    return data


# ── Analysis modules ──────────────────────────────────────────────────────────

def schema_compliance(data: dict) -> pd.DataFrame:
    rows = []
    for cond, recs in data.items():
        n_total = len(recs)
        n_valid = sum(1 for r in recs if r.get("parsed_json_valid"))
        rows.append({
            "condition":        cond,
            "total_generated":  n_total,
            "schema_valid":     n_valid,
            "compliance_rate":  round(n_valid / n_total, 3) if n_total else 0,
        })
    return pd.DataFrame(rows)


def gate_decomposition(data: dict) -> pd.DataFrame:
    rows = []
    for cond, recs in data.items():
        n = len(recs)
        rows.append({
            "condition": cond,
            "total":     n,
            "C1_schema":      sum(1 for r in recs if r.get("gate_schema")),
            "C2_onto":        sum(1 for r in recs if r.get("gate_ontology") and r.get("gate_schema")),
            "C3_logic":       sum(1 for r in recs if r.get("gate_logic") and r.get("gate_schema") and r.get("gate_ontology")),
            "C3_rate":        round(sum(1 for r in recs if r.get("gate_pass")) / n, 3) if n else 0,
        })
    df = pd.DataFrame(rows)
    for col in ("C1_schema","C2_onto","C3_logic"):
        df[f"{col}_rate"] = (df[col] / df["total"].replace(0, 1)).round(3)
    return df


def diversity_audit(data: dict) -> pd.DataFrame:
    rows = []
    for cond, recs in data.items():
        valid = [r for r in recs if r.get("parsed_json_valid")]
        t_labels = [r.get("T","Unknown") for r in valid]
        n_labels = [r.get("N","Unknown") for r in valid]
        m_labels = [r.get("M","Unknown") for r in valid]
        t_ent = compute_entropy(t_labels)
        n_ent = compute_entropy(n_labels)
        m_ent = compute_entropy(m_labels)
        rows.append({
            "condition":   cond,
            "n_valid":     len(valid),
            "T_entropy":   round(t_ent, 3),
            "N_entropy":   round(n_ent, 3),
            "M_entropy":   round(m_ent, 3),
            "T_pass":      t_ent >= ENTROPY_FLOORS["T"],
            "N_pass":      n_ent >= ENTROPY_FLOORS["N"],
            "M_pass":      m_ent >= ENTROPY_FLOORS["M"],
            "all_pass":    all([t_ent >= ENTROPY_FLOORS["T"],
                                n_ent >= ENTROPY_FLOORS["N"],
                                m_ent >= ENTROPY_FLOORS["M"]]),
        })
    return pd.DataFrame(rows)


def snomed_density_table(data: dict) -> pd.DataFrame:
    rows = []
    for cond, recs in data.items():
        densities = [r.get("snomed_density", 0.0) for r in recs]
        unique_ids = {
            a.get("snomed_id","")
            for r in recs
            for a in (r.get("snomed_codes") or [])
            if a.get("snomed_id")
        }
        rows.append({
            "condition":              cond,
            "avg_snomed_per_100w":    round(float(np.mean(densities)), 2) if densities else 0,
            "median_snomed_per_100w": round(float(np.median(densities)), 2) if densities else 0,
            "unique_snomed_concepts": len(unique_ids),
            "rag_grounded_pct":       round(
                sum(1 for r in recs if r.get("rag_grounded")) / max(len(recs),1) * 100, 1),
        })
    return pd.DataFrame(rows)


def ajcc_violations_table(data: dict) -> pd.DataFrame:
    rows = []
    for cond, recs in data.items():
        schema_valid = [r for r in recs if r.get("gate_schema")]
        all_vio      = [v for r in schema_valid for v in (r.get("ajcc_violations") or [])]
        vio_counts   = pd.Series(all_vio).value_counts().to_dict() if all_vio else {}
        rows.append({
            "condition":        cond,
            "schema_valid":     len(schema_valid),
            "n_violations":     len(all_vio),
            "violation_rate":   round(len(all_vio) / max(len(schema_valid),1), 3),
            "M1_no_stageIV":    sum(1 for v in all_vio if "M1_without" in v),
            "other_violations": len(all_vio) - sum(1 for v in all_vio if "M1_without" in v),
        })
    return pd.DataFrame(rows)


def summary_table(schema_df, gate_df, div_df, snomed_df, vio_df) -> pd.DataFrame:
    df = schema_df[["condition","compliance_rate"]].merge(
         gate_df[["condition","C3_rate"]], on="condition").merge(
         div_df[["condition","T_entropy","N_entropy","M_entropy","all_pass"]], on="condition").merge(
         snomed_df[["condition","avg_snomed_per_100w","unique_snomed_concepts"]], on="condition").merge(
         vio_df[["condition","violation_rate"]], on="condition")
    df.columns = [
        "Condition","Schema compliance","Gate pass rate (C3)",
        "T entropy","N entropy","M entropy","Diversity all pass",
        "Avg SNOMED /100w","Unique SNOMED concepts","AJCC violation rate",
    ]
    return df


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 100)
    print("PHASE 2 — CROSS-CONDITION QUALITY SUMMARY")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Quality Audit")
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--export-dir",  default=os.path.join(cfg.RESULTS_DIR, "analysis"))
    args = parser.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)
    print("Loading Phase 1 records...")
    data = load_all(args.results_dir)
    if not data:
        print("[ERROR] No Phase 1 data found. Run phases/phase1_generate.py first.")
        return

    print("\nRunning quality analysis...")
    schema_df = schema_compliance(data)
    gate_df   = gate_decomposition(data)
    div_df    = diversity_audit(data)
    snomed_df = snomed_density_table(data)
    vio_df    = ajcc_violations_table(data)
    summ_df   = summary_table(schema_df, gate_df, div_df, snomed_df, vio_df)

    print_summary(summ_df)

    exports = {
        "phase2_schema_compliance.csv":  schema_df,
        "phase2_gate_decomposition.csv": gate_df,
        "phase2_diversity.csv":          div_df,
        "phase2_snomed_density.csv":     snomed_df,
        "phase2_ajcc_violations.csv":    vio_df,
        "phase2_summary_table.csv":      summ_df,
    }
    for fname, df in exports.items():
        path = os.path.join(args.export_dir, fname)
        df.to_csv(path, index=False)
        print(f"  Saved -> {path}")

    print(f"\nPhase 2 complete. Next: run phases/phase3_benchmark.py")


if __name__ == "__main__":
    main()
