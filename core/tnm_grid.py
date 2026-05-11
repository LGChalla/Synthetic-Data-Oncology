"""core/tnm_grid.py — TNM staging grid, AJCC logic, diversity gates."""

import itertools
import random
import pandas as pd
from scipy.stats import entropy as shannon_entropy

T_VALUES = ["T1", "T2", "T3", "T4"]
N_VALUES = ["N0", "N1", "N2", "N3"]
M_VALUES = ["M0", "M1"]
TNM_GRID = list(itertools.product(T_VALUES, N_VALUES, M_VALUES))  # 32 cells

ENTROPY_FLOORS = {"T": 1.109, "N": 1.109, "M": 0.554}

_STAGE_MAP = {
    ("T1","N0"):"IA",  ("T2","N0"):"IB",  ("T1","N1"):"IIA", ("T2","N1"):"IIB",
    ("T3","N0"):"IIB", ("T3","N1"):"IIIA",("T4","N0"):"IIIA",("T4","N1"):"IIIA",
    ("T1","N2"):"IIIA",("T2","N2"):"IIIA",("T3","N2"):"IIIB",("T4","N2"):"IIIB",
    ("T1","N3"):"IIIB",("T2","N3"):"IIIB",("T3","N3"):"IIIC",("T4","N3"):"IIIC",
}

def infer_stage_group(t: str, n: str, m: str) -> str:
    if m == "M1":
        return "IV"
    return _STAGE_MAP.get((t, n), "Unknown")


def build_case(t: str, n: str, m: str) -> str:
    size  = {"1":"1.5 cm","2":"3 cm","3":"6 cm","4":"8 cm (chest wall invasion)"}
    n_dsc = {
        "0":"no lymph node involvement",
        "1":"ipsilateral peribronchial nodes",
        "2":"ipsilateral mediastinal nodes",
        "3":"contralateral or supraclavicular nodes",
    }
    m_dsc = {"0":"no distant metastasis","1":"distant metastasis present (liver)"}
    age   = random.randint(45, 80)
    sex   = random.choice(["Male","Female"])
    hist  = random.choice(["Adenocarcinoma","Squamous Cell Carcinoma",
                           "Large Cell Carcinoma","Small Cell Lung Cancer"])
    race  = random.choice(["White","Black or African American","Asian",
                           "Hispanic or Latino","Other"])
    stage = infer_stage_group(t, n, m)
    return (
        f"Patient: {age}yo {sex}, {race}. Lung {hist}. "
        f"Primary tumor {size[t[1:]]} ({t}), {n_dsc[n[1:]]} ({n}), "
        f"{m_dsc[m[1:]]} ({m}). Clinical Stage {stage}."
    )


def get_tnm_cell(run_index: int) -> tuple:
    """Round-robin assignment of TNM cells across ablation runs."""
    return TNM_GRID[run_index % len(TNM_GRID)]


def compute_entropy(labels: list) -> float:
    if not labels:
        return 0.0
    counts = pd.Series(labels).value_counts()
    return float(shannon_entropy(counts)) if len(counts) > 1 else 0.0


def audit_diversity(records: list) -> dict:
    """
    Computes Shannon entropy per TNM dimension over valid records.
    Returns per-dim entropy, floor status, and missing/scarce cells.
    """
    valid = [r for r in records if r.get("parsed_json_valid")]
    t_labels = [r.get("T","Unknown") for r in valid]
    n_labels = [r.get("N","Unknown") for r in valid]
    m_labels = [r.get("M","Unknown") for r in valid]

    report = {"n_valid": len(valid)}
    for dim, labels, expected in [
        ("T", t_labels, T_VALUES),
        ("N", n_labels, N_VALUES),
        ("M", m_labels, M_VALUES),
    ]:
        ent     = compute_entropy(labels)
        floor   = ENTROPY_FLOORS[dim]
        counts  = pd.Series(labels).value_counts().to_dict()
        missing = [c for c in expected if counts.get(c, 0) == 0]
        scarce  = [c for c in expected
                   if 0 < counts.get(c, 0) < len(labels) / (len(expected) * 2)]
        report[dim] = {
            "entropy":  round(ent, 3),
            "floor":    floor,
            "pass":     ent >= floor,
            "dist":     counts,
            "missing":  missing,
            "scarce":   scarce,
        }

    report["all_pass"] = all(report[d]["pass"] for d in ("T","N","M"))
    return report


def print_diversity_report(report: dict, label: str = ""):
    prefix = f"[{label}] " if label else ""
    print(f"\n  {prefix}Diversity audit — n_valid={report['n_valid']}")
    for dim in ("T","N","M"):
        d = report[dim]
        status = "PASS" if d["pass"] else "FAIL"
        print(f"    [{dim}] entropy={d['entropy']:.3f}  floor={d['floor']}  {status}")
        if d.get("missing"):
            print(f"         Missing cells: {d['missing']}")
        if d.get("scarce"):
            print(f"         Scarce cells:  {d['scarce']}")
