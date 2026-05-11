"""core/logging_utils.py — Record logging, checkpointing, and result assembly."""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from core import bioportal as bp
from core.gate import GateResult


def make_record(
    model_id:    str,
    condition:   str,
    run_index:   int,
    t_target:    str,
    n_target:    str,
    m_target:    str,
    raw_output:  str,
    parsed_json,
    snomed_codes: list,
    gate_result:  GateResult,
    rag_grounded: bool,
    rag_retriever: str,
    params:       dict,
) -> dict:
    stg = {}
    if isinstance(parsed_json, dict) and "notes" in parsed_json:
        try:
            stg = parsed_json["notes"][0].get("staging", {})
        except (IndexError, AttributeError):
            pass

    from core.gate import check_schema
    return {
        "run_id":              str(uuid.uuid4()),
        "timestamp":           datetime.now().isoformat(),
        "model":               model_id,
        "condition":           condition,
        "run_index":           run_index,
        "T_target":            t_target,
        "N_target":            n_target,
        "M_target":            m_target,
        "T":                   str(stg.get("T", "Unknown")).upper(),
        "N":                   str(stg.get("N", "Unknown")).upper(),
        "M":                   str(stg.get("M", "Unknown")).upper(),
        "stage_group":         str(stg.get("stage_group", "Unknown")).upper(),
        "free_text":           (parsed_json or {}).get("notes", [{}])[0].get("free_text", "")
                               if isinstance(parsed_json, dict) else "",
        "raw_output":          raw_output,
        "parsed_json_valid":   check_schema(parsed_json),
        "parsed_json":         parsed_json,
        "snomed_codes":        snomed_codes,
        "snomed_density":      round(bp.density(raw_output, snomed_codes), 2),
        "gate_schema":         gate_result.schema,
        "gate_ontology":       gate_result.ontology,
        "gate_logic":          gate_result.logic,
        "gate_pass":           gate_result.full,
        "gate_label":          gate_result.label(),
        "ajcc_violations":     gate_result.violations,
        "rag_grounded":        rag_grounded,
        "rag_retriever":       rag_retriever,
        "params":              params,
    }


def append_jsonl(record: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    records, skipped = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"  [WARN] Line {i}: {e}")
    if skipped:
        print(f"  [WARN] {skipped} line(s) skipped.")
    return records


# ── Checkpointing ─────────────────────────────────────────────────────────────

def load_completed(checkpoint_path: str) -> set:
    if not os.path.exists(checkpoint_path):
        return set()
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def mark_completed(run_key: str, checkpoint_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(run_key + "\n")
