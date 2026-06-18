# pipeline/phase2_analysis.py
# DATAGEN Phase 2 — Statistical Quality Analysis
#
# ── CHANGELOG ─────────────────────────────────────────────────────────────────
# FIX 1 [CRITICAL] Label uniformity auditing (analyze_label_diversity).
# FIX 2 [IMPORTANT] Per-class TNM breakdown in exploratory analysis.
# FIX 3 [MINOR] CSV export for downstream graph generation.
# FIX 4 [CRITICAL] Robust TNM normalization — identical to Phase 3 / Phase 4.
#   Without this, raw labels like 'T3', bare '3', 'T2a', 'pT1' are counted as
#   SEPARATE classes, fragmenting the distribution and inflating entropy
#   (e.g. T-entropy 1.99 across 10 phantom classes instead of 1.31 across 4).
#   normalize_tnm_label() collapses them to the canonical T1-T4 / N0-N3 / M0-M1
#   space before any diversity statistic is computed.
# ──────────────────────────────────────────────────────────────────────────────
import os
import re
import json
import pandas as pd
import numpy as np
from scipy.stats import entropy, chisquare, chi2_contingency

RESULTS_FILE = "results/master_results.jsonl"
EXPORT_DIR   = "results/analysis_exports"

DIVERSITY_ENTROPY_FLOOR = {"T": 1.11, "N": 1.11, "M": 0.55}


def normalize_tnm_label(value: str, prefix: str) -> str:
    """
    FIX 4: Robust TNM normalization — identical to Phase 3 / Phase 4.
      pT1 -> T1, cN2 -> N2, T2a -> T2, bare 3 -> T3, blank -> <PREFIX>UNKNOWN.
    """
    v = str(value).strip().upper()
    if v in ("", "NONE", "NULL", "NAN", "UNKNOWN"):
        return prefix.upper() + "UNKNOWN"
    v = re.sub(r"^[PC]", "", v)
    if v.startswith(prefix.upper()):
        v = v[len(prefix):]
    m = re.match(r"(IS|X|[0-4])", v)
    if not m:
        return prefix.upper() + "UNKNOWN"
    return prefix.upper() + m.group(1)


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] {filepath} not found. Run phase1_datagen.py first.")
        return pd.DataFrame()
    data, skipped = [], 0
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    skipped += 1
                    print(f"[WARN] Skipping malformed JSON on line {i}: {e}")
    if skipped:
        print(f"[WARN] {skipped} record(s) skipped due to parse errors.")
    return pd.DataFrame(data)


def flatten_controlled_data(df):
    """Extract normalized T/N/M labels from controlled + longitudinal valid records."""
    records = []
    for _, row in df[df["layer"].isin(["controlled", "longitudinal"])].iterrows():
        if not row.get("parsed_json_valid") or not isinstance(row.get("parsed_json"), dict):
            continue
        if len(row.get("validation_errors", [])) > 0:
            continue
        notes = row["parsed_json"].get("notes", [])
        if not notes: continue
        stg = notes[0].get("staging", {}) if isinstance(notes[0], dict) else {}
        records.append({
            "model": row.get("model", "Unknown"),
            "layer": row.get("layer", "Unknown"),
            "T":     normalize_tnm_label(stg.get("T", "Unknown"), "T"),
            "N":     normalize_tnm_label(stg.get("N", "Unknown"), "N"),
            "M":     normalize_tnm_label(stg.get("M", "Unknown"), "M"),
            "stage_group": str(stg.get("stage_group", "Unknown")).upper(),
        })
    return pd.DataFrame(records)


def flatten_exploratory_data(df):
    records = []
    for _, row in df[df["layer"] == "exploratory"].iterrows():
        if not row.get("parsed_json_valid") or not isinstance(row.get("parsed_json"), dict):
            continue
        notes = row["parsed_json"].get("notes", [])
        if not isinstance(notes, list) or not notes: continue
        note = notes[0]
        if not isinstance(note, dict): continue
        stg  = note.get("staging", {}) if isinstance(note.get("staging"), dict) else {}
        hist = note.get("histology", {}) if isinstance(note.get("histology"), dict) else {}
        demo = note.get("demographics", {}) if isinstance(note.get("demographics"), dict) else {}
        records.append({
            "model":       row.get("model", "Unknown"),
            "stage_group": str(stg.get("stage_group", "Unknown")).upper(),
            "T":           normalize_tnm_label(stg.get("T", "Unknown"), "T"),
            "N":           normalize_tnm_label(stg.get("N", "Unknown"), "N"),
            "M":           normalize_tnm_label(stg.get("M", "Unknown"), "M"),
            "histology":   str(hist.get("name", "Unknown")),
            "sex":         str(demo.get("sex", "Unknown")),
        })
    return pd.DataFrame(records)


def calculate_diversity_score(series):
    counts = series.value_counts()
    return entropy(counts) if len(counts) > 1 else 0.0


def calculate_cramers_v(contingency_table):
    chi2  = chi2_contingency(contingency_table)[0]
    n     = contingency_table.sum().sum()
    r, k  = contingency_table.shape
    if n == 0 or min(r-1, k-1) == 0: return 0.0
    return np.sqrt(chi2 / (n * min(r-1, k-1)))


# ── ANALYSIS FUNCTIONS ────────────────────────────────────────────────────────

def analyze_json_validity(df):
    print("\n" + "="*60)
    print("GLOBAL JSON VALIDITY")
    print("="*60)
    summary = df.groupby(["model", "layer"])["parsed_json_valid"].agg(
        Total="count", Valid="sum"
    )
    summary["Validity_Rate (%)"] = (summary["Valid"] / summary["Total"] * 100).round(1)
    print(summary.to_string())
    summary.to_csv(f"{EXPORT_DIR}/table_1_global_validity.csv")


def analyze_exploratory(df):
    print("\n" + "="*60)
    print("LAYER 1: EXPLORATORY — STAGING BIAS (stage_group + T/N/M)")
    print("="*60)
    flat_df = flatten_exploratory_data(df)
    if flat_df.empty:
        print("No exploratory data found."); return
    stats_records = []
    for model in flat_df["model"].unique():
        print(f"\n--- Model: {model} ---")
        mdf = flat_df[flat_df["model"] == model]
        sg_counts = mdf["stage_group"].value_counts()
        expected  = np.array([sg_counts.sum() / len(sg_counts)] * len(sg_counts))
        stat, p   = chisquare(f_obs=sg_counts.values, f_exp=expected)
        sg_ent    = calculate_diversity_score(mdf["stage_group"])
        print(f"  stage_group entropy: {sg_ent:.3f} | χ² p={p:.4f} "
              f"{'(*Bias*)' if p < 0.05 else '(No Bias)'}")
        for dim in ("T", "N", "M"):
            ent = calculate_diversity_score(mdf[dim])
            print(f"  {dim} entropy: {ent:.3f} | dist: {mdf[dim].value_counts().to_dict()}")
        stats_records.append({
            "model": model,
            "stage_group_entropy": round(sg_ent, 3),
            "stage_bias_pval":     round(p, 4),
            "T_entropy":           round(calculate_diversity_score(mdf["T"]), 3),
            "N_entropy":           round(calculate_diversity_score(mdf["N"]), 3),
            "M_entropy":           round(calculate_diversity_score(mdf["M"]), 3),
        })
    out = pd.DataFrame(stats_records)
    out.to_csv(f"{EXPORT_DIR}/table_2_exploratory_bias.csv", index=False)
    print(f"\nExported → table_2_exploratory_bias.csv")


def analyze_label_diversity(df):
    print("\n" + "="*60)
    print("GOLDEN CORPUS DIVERSITY AUDIT (Controlled + Longitudinal)")
    print("="*60)
    flat_df = flatten_controlled_data(df)
    if flat_df.empty:
        print("No golden records found — run Phase 1 first."); return
    audit_records = []
    for model in flat_df["model"].unique():
        print(f"\n--- Model: {model} (n={len(flat_df[flat_df['model']==model])}) ---")
        mdf = flat_df[flat_df["model"] == model]
        model_row = {"model": model}
        for dim in ("T", "N", "M", "stage_group"):
            counts  = mdf[dim].value_counts()
            ent     = entropy(counts) if len(counts) > 1 else 0.0
            floor   = DIVERSITY_ENTROPY_FLOOR.get(dim, 0.0)
            status  = "PASS" if ent >= floor else "FAIL ⚠️"
            print(f"  [{dim}] entropy={ent:.3f}  floor={floor:.3f}  {status}")
            print(f"       dist: {counts.to_dict()}")
            model_row[f"{dim}_entropy"] = round(ent, 3)
            model_row[f"{dim}_status"]  = status
        audit_records.append(model_row)
    out = pd.DataFrame(audit_records)
    out.to_csv(f"{EXPORT_DIR}/table_label_diversity_audit.csv", index=False)
    print(f"\nExported → table_label_diversity_audit.csv")

    print("\n--- Overall Corpus (all models) ---")
    for dim in ("T", "N", "M"):
        counts = flat_df[dim].value_counts()
        ent    = entropy(counts) if len(counts) > 1 else 0.0
        floor  = DIVERSITY_ENTROPY_FLOOR.get(dim, 0.0)
        status = "PASS" if ent >= floor else "FAIL ⚠️  — label collapse risk"
        print(f"  [{dim}] entropy={ent:.3f}  {status}  | {counts.to_dict()}")


def analyze_controlled(df):
    print("\n" + "="*60)
    print("LAYER 2: CONTROLLED — ABLATION (Cramer's V)")
    print("="*60)
    ctrl_df = df[df["layer"] == "controlled"].copy()
    if ctrl_df.empty: return
    ctrl_df["temperature"] = ctrl_df["params"].apply(
        lambda x: x.get("temperature", "N/A") if isinstance(x, dict) else "N/A")
    ctrl_df["chat_tpl"]    = ctrl_df["params"].apply(
        lambda x: x.get("use_chat_template", True) if isinstance(x, dict) else True)
    ctrl_df["strict_json"] = ctrl_df["params"].apply(
        lambda x: x.get("strict_json", True) if isinstance(x, dict) else True)
    summary = ctrl_df.groupby(["model", "temperature", "chat_tpl", "strict_json"])[
        "parsed_json_valid"
    ].agg(Total_Runs="count", Valid_Count="sum")
    summary["Compliance_Rate (%)"] = (summary["Valid_Count"] / summary["Total_Runs"] * 100).round(1)
    print(summary.to_string())
    summary.to_csv(f"{EXPORT_DIR}/table_3_controlled_descriptive.csv")
    print("\n--- Strict JSON Stopper Effect Size ---")
    effect_records = []
    for model in ctrl_df["model"].unique():
        mdf         = ctrl_df[ctrl_df["model"] == model]
        contingency = pd.crosstab(mdf["strict_json"], mdf["parsed_json_valid"])
        if contingency.size == 4:
            chi2, p, _, _ = chi2_contingency(contingency)
            cv            = calculate_cramers_v(contingency)
            print(f"  {model}: p={p:.4f} | Cramer's V={cv:.3f} "
                  f"({'Negligible' if cv < 0.1 else 'Moderate+'})")
            effect_records.append({"model": model, "p_value": p, "cramers_v": cv})
    pd.DataFrame(effect_records).to_csv(f"{EXPORT_DIR}/table_3_ablation_effect_size.csv", index=False)


def analyze_longitudinal(df):
    print("\n" + "="*60)
    print("LAYER 3: LONGITUDINAL — CLINICAL LOGIC ERRORS")
    print("="*60)
    long_df = df[df["layer"] == "longitudinal"]
    if long_df.empty: return
    records = []
    for model in long_df["model"].unique():
        mdf         = long_df[long_df["model"] == model]
        total       = len(mdf)
        error_runs  = mdf[mdf["validation_errors"].apply(
            lambda x: isinstance(x, list) and len(x) > 0)]
        n_errors    = len(error_runs)
        error_rate  = (n_errors / total) * 100 if total > 0 else 0
        print(f"  {model}: {total} timelines | {n_errors} errors ({error_rate:.1f}%)")
        records.append({"model": model, "total_timelines": total,
                        "errors": n_errors, "error_rate": round(error_rate, 1)})
    pd.DataFrame(records).to_csv(f"{EXPORT_DIR}/table_4_longitudinal_errors.csv", index=False)


def analyze_snomed_coverage(df):
    print("\n" + "="*60)
    print("ONTOLOGY: LENGTH-NORMALIZED SNOMED DENSITY")
    print("="*60)
    df["snomed_count"]    = df["snomed_codes"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["word_count"]      = df["raw_output"].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 1)
    df["word_count"]      = df["word_count"].replace(0, 1)
    df["snomed_per_100w"] = (df["snomed_count"] / df["word_count"]) * 100
    summary = df.groupby(["model", "layer"]).agg(
        Avg_Raw_Terms  =("snomed_count",    "mean"),
        Avg_Word_Count =("word_count",      "mean"),
        SNOMED_per_100w=("snomed_per_100w", "mean"),
    ).round(2)
    print(summary.to_string())
    summary.to_csv(f"{EXPORT_DIR}/table_5_snomed_normalized.csv")

    # RAG vs non-RAG SNOMED density (uses rag_used flag from the real corpus)
    if "rag_used" in df.columns:
        print("\n--- SNOMED density by RAG condition ---")
        # include zero-word/zero-density failed generations as density 0
        rag_summary = df.groupby("rag_used")["snomed_per_100w"].agg(
            mean="mean", median="median", n="count").round(3)
        print(rag_summary.to_string())
        rag_summary.to_csv(f"{EXPORT_DIR}/table_5b_snomed_by_rag.csv")


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"[ERROR] {RESULTS_FILE} not found. Run pipeline/phase1_datagen.py first.")
        return
    os.makedirs(EXPORT_DIR, exist_ok=True)
    df = load_data(RESULTS_FILE)
    if df.empty:
        print("No data found. Run Phase 1 first.")
        return
    analyze_json_validity(df)
    analyze_exploratory(df)
    analyze_label_diversity(df)
    analyze_controlled(df)
    analyze_longitudinal(df)
    analyze_snomed_coverage(df)
    print(f"\nAnalysis complete. Exports in ./{EXPORT_DIR}/")


if __name__ == "__main__":
    main()
