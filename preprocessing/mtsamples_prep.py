#!/usr/bin/env python3
"""
STEP_6a_Pre_Phase3_MTSamples.py
────────────────────────────────
Downloads MTSamples from HuggingFace, extracts explicit TNM annotations,
and writes the benchmark CSV that Step 6b uses for real-world TSTR evaluation.

Run TWICE:
    python STEP_6a_Pre_Phase3_MTSamples.py --mode lung
    python STEP_6a_Pre_Phase3_MTSamples.py --mode all_cancer

Outputs:
    mtsamples_lung_gold.csv        (~72 records, T-stage present)
    mtsamples_all_cancer_gold.csv  (~100-300 records, T-stage present)

IMPORTANT: N and M ground truth is absent in 82-86% of records (physicians
write staging in prose, not explicit TNM notation). The downstream benchmark
(Step 6b) evaluates T-stage ONLY for MTSamples. This is the honest approach.
"""

import re, argparse
import pandas as pd
from datasets import load_dataset


# ── Cancer type keywords ──────────────────────────────────────────────────────
CANCER_TYPES = {
    "lung":       r"lung|pulmonary|broncho|bronchogenic",
    "breast":     r"breast|mammary|mastectomy",
    "colorectal": r"colon|colorectal|rectal|rectum",
    "bladder":    r"bladder|urothelial|cystectomy",
    "prostate":   r"prostate|prostatic",
    "cervical":   r"cervix|cervical|vulv",
    "head_neck":  r"larynx|pharynx|oral|tongue|thyroid|parotid|head.*neck|neck.*head",
    "skin":       r"melanoma|skin|squamous.*skin|basal.*cell",
    "kidney":     r"renal|kidney|nephrectomy",
    "other":      r"carcinoma|cancer|tumor|tumour|malignant|neoplasm|sarcoma|lymphoma",
}

def infer_cancer_type(text: str) -> str:
    if not isinstance(text, str): return "unknown"
    t = text.lower()
    for label, pattern in CANCER_TYPES.items():
        if re.search(pattern, t): return label
    return "unknown"


# ── TNM regex ─────────────────────────────────────────────────────────────────
TNM_COMPACT    = re.compile(
    r'(?i)\b[cpyrma]?\s*T\s*([0-4](?:[a-c])?|is|x)\s*[,\s;/:-]*'
    r'[cpyrma]?\s*N\s*([0-3](?:[a-c])?|x)\s*[,\s;/:-]*'
    r'[cpyrma]?\s*M\s*([0-1](?:[a-c])?|x)\b'
)
TNM_SEPARATE_T = re.compile(r'(?i)\b[cpyrma]?\s*T\s*([0-4](?:[a-c])?|is|x)\b')
TNM_SEPARATE_N = re.compile(r'(?i)\b[cpyrma]?\s*N\s*([0-3](?:[a-c])?|x)\b')
TNM_SEPARATE_M = re.compile(r'(?i)\b[cpyrma]?\s*M\s*([0-1](?:[a-c])?|x)\b')


def norm_tok(prefix: str, value: str) -> str:
    return f"{prefix}{value.upper().replace(' ','')}"


def extract_tnm(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"T_target":"Unknown","N_target":"Unknown","M_target":"Unknown",
                "has_complete_tnm":0,"has_partial_tnm":0}
    clean = " ".join(text.split())
    m     = TNM_COMPACT.search(clean)
    if m:
        return {"T_target": norm_tok("T",m.group(1)),
                "N_target": norm_tok("N",m.group(2)),
                "M_target": norm_tok("M",m.group(3)),
                "has_complete_tnm":1, "has_partial_tnm":1}
    t_m = TNM_SEPARATE_T.search(clean)
    n_m = TNM_SEPARATE_N.search(clean)
    mm  = TNM_SEPARATE_M.search(clean)
    t   = norm_tok("T", t_m.group(1)) if t_m else "Unknown"
    n   = norm_tok("N", n_m.group(1)) if n_m else "Unknown"
    mv  = norm_tok("M", mm.group(1))  if mm  else "Unknown"
    return {"T_target": t, "N_target": n, "M_target": mv,
            "has_complete_tnm": int(t!="Unknown" and n!="Unknown" and mv!="Unknown"),
            "has_partial_tnm":  int(t!="Unknown")}


def load_mtsamples(mode: str) -> pd.DataFrame:
    print("Downloading MTSamples from Hugging Face …")
    ds = load_dataset("harishnair04/mtsamples", split="train")
    df = pd.DataFrame(ds)

    df = df[df["medical_specialty"].str.contains(
        r"Oncology|Radiology|Pulmonary|Thoracic|Surgery|Pathology",
        case=False, na=False
    )].copy()
    print(f"After specialty filter: {len(df)} notes")

    if mode == "lung":
        df = df[df["transcription"].str.contains(
            r"lung|pulmonary|adenocarcinoma|carcinoma|bronchogenic|nodule|mass|metastasis",
            case=False, na=False
        )].copy()
        print(f"After lung content filter: {len(df)} notes")

    df = df.rename(columns={"transcription":"free_text"})
    df["cancer_type"] = df["free_text"].apply(infer_cancer_type)
    return df[["medical_specialty","cancer_type","free_text"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["lung","all_cancer"], required=True,
        help="lung = lung-specific notes | all_cancer = all oncology specialties"
    )
    args = parser.parse_args()

    out_file = (
        "mtsamples_lung_gold.csv"
        if args.mode == "lung"
        else "mtsamples_all_cancer_gold.csv"
    )

    df       = load_mtsamples(args.mode)
    extracted = df["free_text"].apply(extract_tnm).apply(pd.Series)
    out       = pd.concat([df, extracted], axis=1)

    complete = int(out["has_complete_tnm"].sum())
    partial  = int(out["has_partial_tnm"].sum())
    print(f"\nNotes with complete TNM (T+N+M): {complete}")
    print(f"Notes with T-stage present:      {partial}")

    gold = out[out["has_partial_tnm"] == 1].copy()

    print(f"\nCancer type breakdown ({len(gold)} total):")
    print(gold["cancer_type"].value_counts().to_string())

    print(f"\nComplete TNM by cancer type:")
    print(gold.groupby("cancer_type")["has_complete_tnm"]
              .agg(["sum","count"])
              .rename(columns={"sum":"complete","count":"total"})
              .to_string())

    gold.to_csv(out_file, index=False)
    print(f"\n>> Saved {len(gold)} records to '{out_file}'")
    print(f"   Complete TNM : {complete} / {len(gold)}")
    print(f"   T-only (N/M=Unknown): {len(gold)-complete} / {len(gold)}")
    print(f"\n   ⚠️  NOTE: N/M ground truth absent in ~{100-round(complete/len(gold)*100)}%")
    print(   "   of records. The Step 6b benchmark evaluates T-stage ONLY for this dataset.")
    print(f"\n>> Next: python STEP_6b_Phase3_benchmark.py --real_data_file {out_file} "
          f"--dataset_label \"{args.mode.replace('_',' ').title()}\"")


if __name__ == "__main__":
    main()
