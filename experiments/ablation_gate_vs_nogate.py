"""
experiments/ablation_gate_vs_nogate.py
Ablation: Full Gate G(x) vs No Gate

Compares records generated under the full neuro-symbolic validation gate
against records admitted without any gate. Quantifies what the gate actually
removes and at what cost in yield.

Two conditions:
  - GATED:    Records must pass schema + ontology + AJCC logic (G(x) = 1)
  - UNGATED:  All generated records admitted regardless of validity

Metrics reported:
  - Schema compliance rate (JSON validity %)
  - SNOMED CT term density (terms / 100 words)
  - AJCC logic violation rate (M1 without Stage IV, etc.)
  - Label diversity (Shannon entropy per T / N / M dimension)
  - Corpus yield (records admitted per generation run)
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from scipy.stats import entropy as shannon_entropy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.bioportal import snomed_density, annotate_snomed

RESULTS_FILE = "results/master_results.jsonl"
EXPORT_DIR   = "results/analysis_exports"

AJCC_FLOORS = {"T": 1.109, "N": 1.109, "M": 0.554}


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


def check_ajcc_logic(parsed_json: dict) -> list:
    """Returns list of clinical-logic violations found in the record."""
    errors = []
    if not parsed_json or "notes" not in parsed_json:
        return ["Missing 'notes' key"]
    for idx, note in enumerate(parsed_json.get("notes", [])):
        stg   = note.get("staging", {}) if isinstance(note, dict) else {}
        m     = str(stg.get("M", "")).upper()
        group = str(stg.get("stage_group", "")).upper()
        if "M1" in m and "IV" not in group:
            errors.append(f"Visit {idx}: M1 without Stage IV (got '{group}')")
    return errors


def extract_tnm(parsed_json: dict) -> tuple:
    """Returns (T, N, M) strings from the first note, or ('Unknown', ...) on failure."""
    try:
        stg = parsed_json.get("notes", [{}])[0].get("staging", {})
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


def run_ablation(records: list, layer_filter: str = "controlled") -> dict:
    """
    Partitions records into GATED and UNGATED conditions and computes
    all comparison metrics.
    """
    layer_records = [r for r in records if r.get("layer") == layer_filter]
    if not layer_records:
        print(f"[WARN] No records found for layer='{layer_filter}'.")
        return {}

    results = {}
    for condition, subset in [
        ("UNGATED", layer_records),
        ("GATED",   [r for r in layer_records if r.get("parsed_json_valid")
                     and not check_ajcc_logic(r.get("parsed_json") or {})]),
    ]:
        n_total     = len(layer_records)
        n_admitted  = len(subset)
        yield_rate  = n_admitted / n_total if n_total else 0.0

        valid_count = sum(1 for r in subset if r.get("parsed_json_valid"))
        schema_rate = valid_count / n_admitted if n_admitted else 0.0

        logic_errors = sum(
            1 for r in subset
            if check_ajcc_logic(r.get("parsed_json") or {})
        )
        logic_error_rate = logic_errors / n_admitted if n_admitted else 0.0

        snomed_densities = []
        for r in subset:
            raw  = r.get("raw_output", "")
            anns = r.get("snomed_codes") or []
            snomed_densities.append(snomed_density(raw, anns))
        avg_snomed = float(np.mean(snomed_densities)) if snomed_densities else 0.0

        t_labels = [extract_tnm(r.get("parsed_json") or {})[0] for r in subset
                    if r.get("parsed_json_valid")]
        n_labels = [extract_tnm(r.get("parsed_json") or {})[1] for r in subset
                    if r.get("parsed_json_valid")]
        m_labels = [extract_tnm(r.get("parsed_json") or {})[2] for r in subset
                    if r.get("parsed_json_valid")]

        results[condition] = {
            "n_total_generated": n_total,
            "n_admitted":        n_admitted,
            "yield_rate":        round(yield_rate, 3),
            "schema_rate":       round(schema_rate, 3),
            "logic_error_rate":  round(logic_error_rate, 3),
            "avg_snomed_per_100w": round(avg_snomed, 2),
            "T_entropy":         round(compute_entropy(t_labels), 3),
            "N_entropy":         round(compute_entropy(n_labels), 3),
            "M_entropy":         round(compute_entropy(m_labels), 3),
        }

    return results


def print_comparison(results: dict):
    if not results:
        return
    print("\n" + "=" * 65)
    print("ABLATION: GATE vs NO-GATE")
    print("=" * 65)
    metrics = [
        ("n_admitted",           "Records admitted"),
        ("yield_rate",           "Corpus yield rate"),
        ("schema_rate",          "Schema compliance"),
        ("logic_error_rate",     "AJCC logic error rate"),
        ("avg_snomed_per_100w",  "SNOMED density (/100w)"),
        ("T_entropy",            "T-stage entropy (floor 1.109)"),
        ("N_entropy",            "N-stage entropy (floor 1.109)"),
        ("M_entropy",            "M-stage entropy (floor 0.554)"),
    ]
    header = f"{'Metric':<35} {'UNGATED':>12} {'GATED':>12}"
    print(header)
    print("-" * 65)
    for key, label in metrics:
        ug = results.get("UNGATED", {}).get(key, "—")
        g  = results.get("GATED",   {}).get(key, "—")
        print(f"  {label:<33} {str(ug):>12} {str(g):>12}")
    print("=" * 65)


def main():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    records = load_records(RESULTS_FILE)
    if not records:
        return

    print(f"Loaded {len(records)} total records from {RESULTS_FILE}")

    for layer in ("controlled", "longitudinal"):
        print(f"\n--- Layer: {layer} ---")
        results = run_ablation(records, layer_filter=layer)
        print_comparison(results)
        if results:
            out_path = f"{EXPORT_DIR}/ablation_gate_vs_nogate_{layer}.csv"
            pd.DataFrame(results).T.to_csv(out_path)
            print(f"Exported → {out_path}")


if __name__ == "__main__":
    main()
