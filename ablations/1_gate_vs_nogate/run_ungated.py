"""
Ablation 1 — Gate vs No-Gate
Condition: UNGATED  (Adapter A)
===============================
Generates synthetic lung cancer records with NO validation gate.
Every record is admitted regardless of schema compliance, ontology
coverage, or clinical-logic correctness.

This is the lower bound — a corpus that represents what you get
if you trust the model's output unconditionally.

Trains:  adapter_A_ungated
Outputs: results/phase1_ungated.jsonl
         adapters/adapter_A_ungated/final_adapter/

Usage:
  python ablations/1_gate_vs_nogate/run_ungated.py
  python ablations/1_gate_vs_nogate/run_ungated.py --runs 64 --model meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config as cfg
from core.generation    import load_hf_model, generate_one, parse_output, unload_model
from core.gate          import evaluate as gate_evaluate
from core.bioportal     import annotate as snomed_annotate
from core.tnm_grid      import get_tnm_cell, build_case, audit_diversity, print_diversity_report
from core.schemas       import RIGID_V3
from core.logging_utils import make_record, append_jsonl, load_completed, mark_completed
from phases.phase4_finetune import train_adapter

CONDITION = "ungated"


def generate(model_id: str, n_runs: int, results_dir: str) -> list:
    jsonl_path  = os.path.join(results_dir, f"phase1_{CONDITION}.jsonl")
    ckpt_path   = os.path.join(results_dir, f"phase1_{CONDITION}.checkpoint")
    os.makedirs(results_dir, exist_ok=True)

    model, tokenizer = load_hf_model(model_id)
    completed        = load_completed(ckpt_path)
    if completed:
        print(f"  Checkpoint: {len(completed)} run(s) already done.")

    records = []
    for i in range(n_runs):
        run_key = f"{CONDITION}::{model_id.split(chr(47))[-1]}::{i}"
        if run_key in completed:
            print(f"  [SKIP] {i+1}/{n_runs}")
            continue

        t, n, m  = get_tnm_cell(i)
        case     = build_case(t, n, m)
        prompt   = (
            f"SCHEMA:\n{json.dumps(RIGID_V3, indent=2)}\n\n"
            f"CASE:\n{case}\n\n"
            "Map this case to the schema exactly. All fields are mandatory."
        )

        print(f"  [{i+1:>3}/{n_runs}] {t}/{n}/{m}", end="  ", flush=True)
        raw          = generate_one(model, tokenizer, prompt, cfg.GEN_PARAMS,
                                    use_json_stop=True, model_id=model_id)
        parsed, _    = parse_output(raw)
        snomed_codes = snomed_annotate(raw, cfg.BIOPORTAL_KEY)
        gate_result  = gate_evaluate(parsed, snomed_codes)

        print(f"schema={gate_result.schema} onto={gate_result.ontology} logic={gate_result.logic}")

        rec             = make_record(
            model_id=model_id, condition=CONDITION, run_index=i,
            t_target=t, n_target=n, m_target=m,
            raw_output=raw, parsed_json=parsed, snomed_codes=snomed_codes,
            gate_result=gate_result, rag_grounded=False, rag_retriever="none",
            params={**cfg.GEN_PARAMS, "run_index": i},
        )
        rec["admitted"] = True   # ungated: always admit
        records.append(rec)
        append_jsonl(rec, jsonl_path)
        mark_completed(run_key, ckpt_path)

    unload_model(model, tokenizer)
    print(f"\n  Admitted: {len(records)} (all records — no gate)")

    report = audit_diversity(records)
    print_diversity_report(report, label=CONDITION)
    return records


def main():
    parser = argparse.ArgumentParser(description="Ablation 1 — Ungated generation")
    parser.add_argument("--model",         default=cfg.GENERATOR_MODEL)
    parser.add_argument("--runs",          type=int, default=cfg.GATE_ABLATION_RUNS)
    parser.add_argument("--results-dir",   default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir",  default=cfg.ADAPTERS_DIR)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*65)
    print(f"ABLATION 1 — UNGATED  |  Adapter A")
    print(f"  Model : {args.model}")
    print(f"  Runs  : {args.runs}")
    print("="*65)

    generate(args.model, args.runs, args.results_dir)

    if not args.skip_training:
        print("\n[Phase 4] Training adapter_A_ungated...")
        train_adapter(CONDITION, args.results_dir, args.adapters_dir)

    print(f"\nDone. Results -> {args.results_dir}")
    print("Next: run run_gated.py, then compare.py")


if __name__ == "__main__":
    main()
