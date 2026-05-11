"""
Phase 1 — Ablation-Aware Synthetic Data Generation
=====================================================
Generates synthetic lung cancer staging records under a specified ablation
condition. Each condition varies which gate components are enforced and
whether RAG grounding is used.

Ablation conditions:
  ungated       — no gate; all records admitted
  schema_only   — JSON schema compliance required
  schema_onto   — schema + SNOMED CT ontology required
  full_norag    — full G(x) gate, no RAG context
  full_rag      — full G(x) gate + MedCPT RAG grounding

For the gate_decomposition ablation, all five conditions run on the
same generated pool and are labelled post-generation.

Usage (standalone):
  python phases/phase1_generate.py --condition full_rag --runs 64
  python phases/phase1_generate.py --condition ungated  --model gpt-4o --runs 32
"""

import argparse
import gc
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from core.generation   import load_hf_model, generate_one, parse_output, unload_model
from core.gate         import evaluate as gate_evaluate
from core.bioportal    import annotate as snomed_annotate
from core.medcpt       import build_retriever, keyword_context
from core.tnm_grid     import get_tnm_cell, build_case, audit_diversity, print_diversity_report
from core.schemas      import RIGID_V3
from core.logging_utils import make_record, append_jsonl, load_completed, mark_completed

VALID_CONDITIONS = ["ungated", "schema_only", "schema_onto", "full_norag", "full_rag"]


def get_paths(condition: str, results_dir: str) -> dict:
    return {
        "jsonl":      os.path.join(results_dir, f"phase1_{condition}.jsonl"),
        "checkpoint": os.path.join(results_dir, f"phase1_{condition}.checkpoint"),
    }


def should_admit(gate_result, condition: str) -> bool:
    if condition == "ungated":      return True
    if condition == "schema_only":  return gate_result.schema
    if condition == "schema_onto":  return gate_result.schema and gate_result.ontology
    if condition in ("full_norag", "full_rag"):
        return gate_result.full
    return False


def run(
    condition:   str,
    model_id:    str,
    n_runs:      int,
    results_dir: str,
    faiss_index: str = "",
    faiss_texts: str = "",
):
    """
    Main generation loop for one ablation condition.
    Returns list of all generated records (admitted and rejected).
    """
    assert condition in VALID_CONDITIONS, f"Unknown condition: {condition}"
    use_rag = condition == "full_rag"
    paths   = get_paths(condition, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"PHASE 1 — Generation  |  condition={condition}")
    print(f"  Model : {model_id}")
    print(f"  Runs  : {n_runs}")
    print(f"  RAG   : {use_rag}")
    print(f"  Output: {paths['jsonl']}")
    print(f"{'='*65}")

    # Load model
    is_gpt = "gpt" in model_id.lower()
    if not is_gpt:
        model, tokenizer = load_hf_model(model_id)
    else:
        model = tokenizer = None

    # Load retriever
    retriever = None
    if use_rag:
        retriever = build_retriever(faiss_index, faiss_texts)
        if retriever is None or not retriever.available:
            print("  [WARN] FAISS unavailable — using keyword context for RAG condition.")

    completed = load_completed(paths["checkpoint"])
    if completed:
        print(f"  Checkpoint: {len(completed)} run(s) already done — skipping.")

    records, admitted, rejected = [], 0, 0

    for i in range(n_runs):
        run_key = f"{condition}::{i}"
        if run_key in completed:
            print(f"  [SKIP] Run {i+1}/{n_runs} already completed.")
            continue

        t, n, m = get_tnm_cell(i)
        case    = build_case(t, n, m)
        prompt  = (
            f"SCHEMA:\n{json.dumps(RIGID_V3, indent=2)}\n\n"
            f"CASE:\n{case}\n\n"
            "Map this case to the schema exactly. All fields are mandatory."
        )

        # RAG context
        rag_ctx = ""
        if use_rag:
            query   = f"lung cancer {t} {n} {m} AJCC staging prognosis treatment SNOMED"
            rag_ctx = (retriever.retrieve(query)
                       if retriever and retriever.available
                       else keyword_context(t, n, m))

        retriever_label = ("medcpt" if (use_rag and retriever and retriever.available)
                           else "keyword" if use_rag else "none")

        print(f"  [{i+1:>3}/{n_runs}] {t}/{n}/{m}", end="  ", flush=True)
        raw = generate_one(model, tokenizer, prompt, cfg.GEN_PARAMS,
                           rag_context=rag_ctx, use_json_stop=True, model_id=model_id)

        parsed, _      = parse_output(raw)
        snomed_codes   = snomed_annotate(raw, cfg.BIOPORTAL_KEY)
        gate_result    = gate_evaluate(parsed, snomed_codes)
        admit          = should_admit(gate_result, condition)

        status_str = (f"schema={gate_result.schema} onto={gate_result.ontology} "
                      f"logic={gate_result.logic} admit={admit}")
        print(status_str)

        rec = make_record(
            model_id=model_id, condition=condition, run_index=i,
            t_target=t, n_target=n, m_target=m,
            raw_output=raw, parsed_json=parsed, snomed_codes=snomed_codes,
            gate_result=gate_result, rag_grounded=use_rag,
            rag_retriever=retriever_label, params={**cfg.GEN_PARAMS, "run_index": i},
        )
        rec["admitted"] = admit
        records.append(rec)
        append_jsonl(rec, paths["jsonl"])
        mark_completed(run_key, paths["checkpoint"])

        if admit: admitted += 1
        else:     rejected += 1

    print(f"\n  Generation complete: {admitted} admitted / {rejected} rejected")

    # Diversity audit on admitted records
    admitted_recs = [r for r in records if r.get("admitted")]
    if admitted_recs:
        report = audit_diversity(admitted_recs)
        print_diversity_report(report, label=condition)
        if not report["all_pass"]:
            print("  [WARN] Diversity gate FAILED — consider running more records "
                  "for under-represented TNM cells before proceeding to Phase 4.")

    # Clean up
    if not is_gpt:
        unload_model(model, tokenizer)

    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Ablation-Aware Generation")
    parser.add_argument("--condition",   default="full_norag", choices=VALID_CONDITIONS)
    parser.add_argument("--model",       default=cfg.GENERATOR_MODEL)
    parser.add_argument("--runs",        type=int, default=cfg.GATE_ABLATION_RUNS)
    parser.add_argument("--results-dir", default=cfg.RESULTS_DIR)
    parser.add_argument("--faiss-index", default=cfg.FAISS_INDEX_PATH)
    parser.add_argument("--faiss-texts", default=cfg.FAISS_TEXTS_PATH)
    args = parser.parse_args()

    run(
        condition=args.condition, model_id=args.model,
        n_runs=args.runs, results_dir=args.results_dir,
        faiss_index=args.faiss_index, faiss_texts=args.faiss_texts,
    )


if __name__ == "__main__":
    main()
