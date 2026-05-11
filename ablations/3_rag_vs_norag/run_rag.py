"""
Ablation 3 — RAG vs No-RAG
Condition: FULL GATE + MedCPT RAG  (Adapter E)
===============================================
Generates records under the full G(x) gate WITH MedCPT retrieval.
Before each generation call, the MedCPT Query Encoder retrieves the
k most semantically relevant PubMed abstracts for the target TNM cell
and injects them into the prompt as grounding context.

If the FAISS index is unavailable, keyword_context() is used as
a fallback — the RAG vocabulary enrichment effect will be understated
but the experiment still runs and produces a valid result.

Trains:  adapter_E_full_rag
Outputs: results/phase1_full_rag.jsonl
         adapters/adapter_E_full_rag/final_adapter/

Usage:
  python ablations/3_rag_vs_norag/run_rag.py
  python ablations/3_rag_vs_norag/run_rag.py --runs 32
  python ablations/3_rag_vs_norag/run_rag.py --faiss-index /path/to/index.faiss --faiss-texts /path/to/abstracts.txt
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
from core.medcpt        import build_retriever, keyword_context
from core.tnm_grid      import get_tnm_cell, build_case, audit_diversity, print_diversity_report
from core.schemas       import RIGID_V3
from core.logging_utils import make_record, append_jsonl, load_completed, mark_completed
from phases.phase4_finetune import train_adapter

CONDITION = "full_rag"


def generate(model_id: str, n_runs: int, results_dir: str,
             faiss_index: str, faiss_texts: str) -> list:
    jsonl_path = os.path.join(results_dir, f"phase1_{CONDITION}.jsonl")
    ckpt_path  = os.path.join(results_dir, f"phase1_{CONDITION}.checkpoint")
    os.makedirs(results_dir, exist_ok=True)

    retriever = build_retriever(faiss_index, faiss_texts)
    if retriever and retriever.available:
        print("  MedCPT retriever ready.")
    else:
        print("  [INFO] MedCPT unavailable — using keyword_context() fallback.")

    model, tokenizer = load_hf_model(model_id)
    completed        = load_completed(ckpt_path)
    records, admitted, rejected = [], 0, 0

    for i in range(n_runs):
        run_key = f"{CONDITION}::{model_id.split(chr(47))[-1]}::{i}"
        if run_key in completed:
            print(f"  [SKIP] {i+1}/{n_runs}")
            continue

        t, n, m = get_tnm_cell(i)
        case    = build_case(t, n, m)
        prompt  = (
            f"SCHEMA:\n{json.dumps(RIGID_V3, indent=2)}\n\n"
            f"CASE:\n{case}\n\n"
            "Map this case to the schema exactly. All fields are mandatory. "
            "Use the retrieved clinical context above to ground your vocabulary."
        )

        # Retrieve context for this TNM cell
        query   = f"lung cancer {t} {n} {m} AJCC 8th edition staging treatment SNOMED"
        rag_ctx = (retriever.retrieve(query)
                   if retriever and retriever.available
                   else keyword_context(t, n, m))
        retriever_label = "medcpt" if (retriever and retriever.available) else "keyword"

        print(f"  [{i+1:>3}/{n_runs}] {t}/{n}/{m}  rag={retriever_label}", end="  ", flush=True)
        raw          = generate_one(model, tokenizer, prompt, cfg.GEN_PARAMS,
                                    rag_context=rag_ctx, use_json_stop=True,
                                    model_id=model_id)
        parsed, _    = parse_output(raw)
        snomed_codes = snomed_annotate(raw, cfg.BIOPORTAL_KEY)
        gate_result  = gate_evaluate(parsed, snomed_codes)
        admit        = gate_result.full

        print(f"gate={admit}  snomed={len(snomed_codes)}")

        rec             = make_record(
            model_id=model_id, condition=CONDITION, run_index=i,
            t_target=t, n_target=n, m_target=m,
            raw_output=raw, parsed_json=parsed, snomed_codes=snomed_codes,
            gate_result=gate_result, rag_grounded=True,
            rag_retriever=retriever_label,
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
            print("  [WARN] Diversity gate FAILED — generate more records before training.")
    return records


def main():
    parser = argparse.ArgumentParser(description="Ablation 3 — Full gate + RAG (Adapter E)")
    parser.add_argument("--model",         default=cfg.GENERATOR_MODEL)
    parser.add_argument("--runs",          type=int, default=cfg.RAG_ABLATION_RUNS)
    parser.add_argument("--faiss-index",   default=cfg.FAISS_INDEX_PATH)
    parser.add_argument("--faiss-texts",   default=cfg.FAISS_TEXTS_PATH)
    parser.add_argument("--results-dir",   default=cfg.RESULTS_DIR)
    parser.add_argument("--adapters-dir",  default=cfg.ADAPTERS_DIR)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("ABLATION 3 — FULL GATE + RAG  |  Adapter E")
    print(f"  Model      : {args.model}")
    print(f"  Runs       : {args.runs}")
    print(f"  FAISS index: {args.faiss_index or 'not set (keyword fallback)'}")
    print("="*65)

    generate(args.model, args.runs, args.results_dir,
             args.faiss_index, args.faiss_texts)

    if not args.skip_training:
        print("\n[Phase 4] Training adapter_E_full_rag...")
        try:
            train_adapter(CONDITION, args.results_dir, args.adapters_dir)
        except RuntimeError as e:
            print(f"  [ERROR] {e}")

    print(f"\nDone. Next: run compare.py")


if __name__ == "__main__":
    main()
