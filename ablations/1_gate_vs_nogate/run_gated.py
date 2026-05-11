"""
Ablation 1 — Gate vs No-Gate
Condition: FULL GATE, NO RAG  (Adapter D)
==========================================
Generates synthetic lung cancer records under the full neuro-symbolic
validation gate G(x) = schema ∧ ontology ∧ AJCC logic, without RAG.

Only records passing all three gate components are admitted to the corpus.
Records that fail are logged but excluded from training.

Trains:  adapter_D_full_norag
Outputs: results/phase1_full_norag.jsonl
         adapters/adapter_D_full_norag/final_adapter/

Usage:
  python ablations/1_gate_vs_nogate/run_gated.py
  python ablations/1_gate_vs_nogate/run_gated.py --runs 64 --model meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config as cfg
from core.generation    import load_hf_model, generate_one, parse_output, unload_model
from core.gate          import evaluate as gate_evaluate
from core.bioportal     import annotate as snomed_annotate
from core.tnm_grid      import get_tnm_cell, build_case, audit_diversity, print_diversity_report
from core.schemas       import RIGID_V3
from core.logging_utils import make_record, append_jsonl, load_completed, mark_completed
from phases.phase4_finetune import train_adapter

CONDITION = "full_norag"


def generate(model_id: str, n_runs: int, results_dir: str) -> list:
    jsonl_path = os.path.join(results_dir, f"phase1_{CONDITION}.jsonl")
    ckpt_path  = os.path.join(results_dir, f"phase1_{CONDITION}.checkpoint")
    os.makedirs(results_dir, exist_ok=True)

    model, tokenizer = load_hf_model(model_id)
    completed        = load_completed(ckpt_path)
    if completed:
        print(f"  Checkpoint: {len(completed)} run(s) already done.")

    records, admitted, rejected = [], 0, 0

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
        admit        = gate_result.full

        print(f"schema={gate_result.schema} onto={gate_result.ontology} "
              f"logic={gate_result.logic} admit={admit}")

        rec             = make_record(
            model_id=model_id, condition=CONDITION, run_index=i,
            t_target=t, n_target=n, m_target=m,
            raw_output=raw, parsed_json=parsed, snomed_codes=snomed_codes,
            gate_result=gate_result, rag_grounded=False, rag_retriever="none",
            params={**cfg.GEN_PARAMS, "run_index": i},
        )
        rec["admitted"] = admit
        records.append(rec)
        append_jsonl(rec, jsonl_path)
        mark_completed(run_key, ckpt_path)

        if admit: admitted += 1
        else:     rejected += 1

    unload_model(model, tokenizer)
    print(f"\n  Admitted: {admitted}  |  Rejected: {rejected}")

    admitted_recs = [r for r in records if r.get("admitted")]
    if admitted_recs:
        report = audit_diversity(admitted_recs)
        print_diversity_report(report, label=CONDITION)
        if not report["all_pass"]:
            print("  [WARN] Diversity FAILED — generate more records before training.")

    return records


def main():
    parser = argparse.ArgumentParser(description="Ablation 1 — Full gate, no RAG")
    parser.add_argument("--model",         default=cfg.GENERATOR_MODEL)
    parser.add_argument("--runs",          type=int, default=cfg.GATE_ABLATION_RUNS)
    parser.add_argument("--results-dir",   default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir",  default=cfg.ADAPTERS_DIR)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*65)
    print(f"ABLATION 1 — FULL GATE (No RAG)  |  Adapter D")
    print(f"  Model : {args.model}")
    print(f"  Runs  : {args.runs}")
    print("="*65)

    generate(args.model, args.runs, args.results_dir)

    if not args.skip_training:
        print("\n[Phase 4] Training adapter_D_full_norag...")
        train_adapter(CONDITION, args.results_dir, args.adapters_dir)

    print(f"\nDone. Results -> {args.results_dir}")
    print("Next: run compare.py")


if __name__ == "__main__":
    main()
