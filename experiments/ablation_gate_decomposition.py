"""
experiments/ablation_gate_decomposition.py
Ablation: Gate Decomposition — Schema / Ontology / Logic

Decomposes the full gate G(x) into its three component constraints,
showing which layer of validation catches what, and how many records
each gate condition admits.

Three checkpoints applied in sequence:
  C1 — Schema only:            rigid JSON field completeness
  C2 — Schema + Ontology:      C1 AND at least one valid SNOMED CT term
  C3 — Schema + Ontology + Logic (full G(x)):  C2 AND no AJCC violations

For each checkpoint, reports:
  - Admission rate (records passing / total generated)
  - Incremental rejection count (caught by this gate, not the previous)
  - SNOMED density of admitted records
  - Label entropy (T / N / M)
  - Distribution of AJCC logic violations by type

This lets you see whether the pipeline's quality is bottlenecked by
schema compliance, ontology coverage, or clinical-logic correctness.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy

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


def passes_schema(record: dict) -> bool:
    return bool(record.get("parsed_json_valid"))


def passes_ontology(record: dict) -> bool:
    """At least one SNOMED CT term annotated in the raw output."""
    codes = record.get("snomed_codes")
    return isinstance(codes, list) and len(codes) > 0


def get_logic_violations(record: dict) -> list:
    """Returns list of AJCC clinical-logic violations."""
    violations = []
    pj = record.get("parsed_json")
    if not pj or "notes" not in pj:
        return ["missing_notes_key"]
    for idx, note in enumerate(pj.get("notes", [])):
        stg   = note.get("staging", {}) if isinstance(note, dict) else {}
        m     = str(stg.get("M", "")).upper()
        group = str(stg.get("stage_group", "")).upper()
        if "M1" in m and "IV" not in group:
            violations.append(f"M1_without_StageIV (visit {idx})")
    return violations


def passes_logic(record: dict) -> bool:
    return len(get_logic_violations(record)) == 0


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


def compute_entropy(labels: list) -> float:
    if not labels:
        return 0.0
    counts = pd.Series(labels).value_counts()
    return float(shannon_entropy(counts)) if len(counts) > 1 else 0.0


def summarize_subset(subset: list, total: int, label: str) -> dict:
    n = len(subset)
    snomed_vals = [
        snomed_density(r.get("raw_output", ""), r.get("snomed_codes") or [])
        for r in subset
    ]
    t_labels = [extract_tnm(r)[0] for r in subset if passes_schema(r)]
    n_labels = [extract_tnm(r)[1] for r in subset if passes_schema(r)]
    m_labels = [extract_tnm(r)[2] for r in subset if passes_schema(r)]

    return {
        "condition":             label,
        "n_admitted":            n,
        "admission_rate":        round(n / total, 3) if total else 0.0,
        "avg_snomed_per_100w":   round(float(np.mean(snomed_vals)), 2) if snomed_vals else 0.0,
        "T_entropy":             round(compute_entropy(t_labels), 3),
        "N_entropy":             round(compute_entropy(n_labels), 3),
        "M_entropy":             round(compute_entropy(m_labels), 3),
    }


def run_decomposition(records: list, layer_filter: str = "controlled") -> pd.DataFrame:
    subset = [r for r in records if r.get("layer") == layer_filter]
    if not subset:
        print(f"[WARN] No records for layer='{layer_filter}'.")
        return pd.DataFrame()

    total = len(subset)

    c1 = [r for r in subset if passes_schema(r)]
    c2 = [r for r in c1 if passes_ontology(r)]
    c3 = [r for r in c2 if passes_logic(r)]

    rows = [
        summarize_subset(subset, total, "C0 — All generated"),
        summarize_subset(c1,     total, "C1 — Schema only"),
        summarize_subset(c2,     total, "C1+C2 — Schema + Ontology"),
        summarize_subset(c3,     total, "C1+C2+C3 — Full G(x)"),
    ]

    # Incremental rejection analysis
    print(f"\n  Layer: {layer_filter}  |  Total generated: {total}")
    print(f"  {'Condition':<35} {'Admitted':>9} {'Rate':>8} {'Rejected by this gate':>22}")
    print("  " + "-" * 78)

    prev = total
    for i, row in enumerate(rows):
        n          = row["n_admitted"]
        rate       = row["admission_rate"]
        rejected   = prev - n if i > 0 else 0
        prev       = n
        print(f"  {row['condition']:<35} {n:>9} {rate:>8.1%} {rejected:>22}")

    # Logic violation breakdown
    all_violations = []
    for r in c1:
        all_violations.extend(get_logic_violations(r))
    if all_violations:
        vdf = pd.Series(all_violations).value_counts()
        print(f"\n  AJCC violation breakdown (across schema-valid records):")
        for vtype, count in vdf.items():
            print(f"    {vtype}: {count}")

    return pd.DataFrame(rows)


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    records = load_records(RESULTS_FILE)
    if not records:
        return

    print(f"\nLoaded {len(records)} records.")
    print("=" * 80)
    print("ABLATION: GATE DECOMPOSITION — Schema → Ontology → Clinical Logic")
    print("=" * 80)

    all_dfs = []
    for layer in ("controlled", "longitudinal"):
        df = run_decomposition(records, layer_filter=layer)
        if not df.empty:
            df["layer"] = layer
            all_dfs.append(df)

    if all_dfs:
        out = pd.concat(all_dfs, ignore_index=True)
        path = f"{EXPORT_DIR}/ablation_gate_decomposition.csv"
        out.to_csv(path, index=False)
        print(f"\nExported → {path}")


if __name__ == "__main__":
    main()
