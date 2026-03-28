# Pre-Phase3_prepare_mtsamples_FIXED.py
#
# RUN ORDER:
#   Step 1 (lung-only):       python Pre-Phase3_prepare_mtsamples_FIXED.py --mode lung
#   Step 2 (all cancer):      python Pre-Phase3_prepare_mtsamples_FIXED.py --mode all_cancer
#
# Each mode writes a SEPARATE output file so results never overwrite each other:
#   --mode lung        →  mtsamples_lung_gold.csv          (~72 records, T-present)
#   --mode all_cancer  →  mtsamples_all_cancer_gold.csv    (~150-300 records, T-present)
#
# WHAT CHANGED FROM ORIGINAL:
#   - Relaxed filter: keeps any note where T is found (not requiring all three of T/N/M)
#     Missing N or M filled as "Unknown" — honest and evaluable downstream
#   - Added --mode flag so lung and cross-cancer runs are completely separate
#   - Added cancer_type column for stratification in the report
#   - Regex patterns and normalization logic unchanged

import re
import argparse
import pandas as pd
from datasets import load_dataset


# ── Cancer type keywords for stratification ───────────────────────────────────
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
    if not isinstance(text, str):
        return "unknown"
    t = text.lower()
    for label, pattern in CANCER_TYPES.items():
        if re.search(pattern, t):
            return label
    return "unknown"


def load_mtsamples(mode: str) -> pd.DataFrame:
    print("Downloading MTSamples from Hugging Face...")
    dataset = load_dataset("harishnair04/mtsamples", split="train")
    df = pd.DataFrame(dataset)

    # Step 1 — specialty filter (always applied)
    df = df[df["medical_specialty"].str.contains(
        r"Oncology|Radiology|Pulmonary|Thoracic|Surgery|Pathology",
        case=False, na=False
    )].copy()
    print(f"After specialty filter: {len(df)} notes")

    # Step 2 — content filter (lung mode only)
    if mode == "lung":
        df = df[df["transcription"].str.contains(
            r"lung|pulmonary|adenocarcinoma|carcinoma|bronchogenic|nodule|mass|metastasis",
            case=False, na=False
        )].copy()
        print(f"After lung content filter: {len(df)} notes")
    else:
        print("No content filter applied (all-cancer mode)")

    df = df.rename(columns={"transcription": "free_text"})
    df["cancer_type"] = df["free_text"].apply(infer_cancer_type)
    return df[["medical_specialty", "cancer_type", "free_text"]].reset_index(drop=True)


# ── Regex patterns ────────────────────────────────────────────────────────────
TNM_COMPACT    = re.compile(r'(?i)\b[cpyrma]?\s*T\s*([0-4](?:[a-c])?|is|x)\s*[,\s;/:-]*[cpyrma]?\s*N\s*([0-3](?:[a-c])?|x)\s*[,\s;/:-]*[cpyrma]?\s*M\s*([0-1](?:[a-c])?|x)\b')
TNM_SEPARATE_T = re.compile(r'(?i)\b[cpyrma]?\s*T\s*([0-4](?:[a-c])?|is|x)\b')
TNM_SEPARATE_N = re.compile(r'(?i)\b[cpyrma]?\s*N\s*([0-3](?:[a-c])?|x)\b')
TNM_SEPARATE_M = re.compile(r'(?i)\b[cpyrma]?\s*M\s*([0-1](?:[a-c])?|x)\b')

def normalize_token(prefix: str, value: str) -> str:
    return f"{prefix}{value.upper().replace(' ', '')}"

def extract_explicit_tnm(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"T_target": "Unknown", "N_target": "Unknown", "M_target": "Unknown",
                "has_complete_tnm": 0, "has_partial_tnm": 0}

    clean = " ".join(text.split())

    m = TNM_COMPACT.search(clean)
    if m:
        return {"T_target": normalize_token("T", m.group(1)),
                "N_target": normalize_token("N", m.group(2)),
                "M_target": normalize_token("M", m.group(3)),
                "has_complete_tnm": 1, "has_partial_tnm": 1}

    t_m = TNM_SEPARATE_T.search(clean)
    n_m = TNM_SEPARATE_N.search(clean)
    mm  = TNM_SEPARATE_M.search(clean)

    t  = normalize_token("T", t_m.group(1)) if t_m else "Unknown"
    n  = normalize_token("N", n_m.group(1)) if n_m else "Unknown"
    mv = normalize_token("M", mm.group(1))  if mm  else "Unknown"

    return {"T_target": t, "N_target": n, "M_target": mv,
            "has_complete_tnm": int(t != "Unknown" and n != "Unknown" and mv != "Unknown"),
            "has_partial_tnm":  int(t != "Unknown")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["lung", "all_cancer"],
        required=True,
        help="lung = lung-specific notes only | all_cancer = all oncology specialties"
    )
    args = parser.parse_args()

    # Output filename is mode-specific — runs never overwrite each other
    out_file = "mtsamples_lung_gold.csv" if args.mode == "lung" else "mtsamples_all_cancer_gold.csv"

    df = load_mtsamples(args.mode)

    extracted = df["free_text"].apply(extract_explicit_tnm).apply(pd.Series)
    out = pd.concat([df, extracted], axis=1)

    complete_count = int(out["has_complete_tnm"].sum())
    partial_count  = int(out["has_partial_tnm"].sum())
    print(f"\nNotes with complete TNM (T+N+M all found): {complete_count}")
    print(f"Notes with at least T-stage present:        {partial_count}")

    gold = out[out["has_partial_tnm"] == 1].copy()

    print(f"\nCancer type breakdown ({len(gold)} total):")
    print(gold["cancer_type"].value_counts().to_string())
    print(f"\nComplete TNM by cancer type:")
    print(gold.groupby("cancer_type")["has_complete_tnm"].agg(["sum","count"])
               .rename(columns={"sum":"complete","count":"total"}).to_string())

    gold.to_csv(out_file, index=False)

    print(f"\n>> SUCCESS: Saved {len(gold)} records to '{out_file}'")
    print(f"   Complete TNM: {complete_count} / {len(gold)}")
    print(f"   T-only (N/M=Unknown): {len(gold) - complete_count} / {len(gold)}")
    print(f"\n>> Next: pass --real_data_file {out_file} to Phase3-_Downstream_re_FIXED.py")


if __name__ == "__main__":
    main()
