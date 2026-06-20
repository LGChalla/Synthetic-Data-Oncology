# Synthetic Data for Lung Cancer Staging — Full Pipeline with Ablation Studies

*What happens when the data you need to build a clinical AI system is exactly the data you can't have?*

Lung cancer kills more people than any other cancer. Survival swings from 80% at Stage I to under 20% at Stage IV — and three in four patients are diagnosed at the late stages. The models that could change that need structured, annotated clinical notes. Those notes are locked behind privacy law, annotation bottlenecks, and institutional walls that take years to navigate.

This project builds the data instead. Not as a shortcut — as a principled engineering commitment.

This repository contains the complete pipeline: the original four-phase generation and evaluation framework, plus the full ablation study infrastructure that isolates exactly which parts of the pipeline do the work.

---

## The Original Pipeline

Four phases. Each one gates the next.

```
[Phase 1] Generate          →  RAG-grounded LLM synthesis across 639 runs
              ↓ G(x) gate: JSON schema + SNOMED CT + AJCC logic
[Phase 2] Audit             →  Shannon entropy per T / N / M — catches silent label collapse
              ↓ entropy gate: H_T,N ≥ 1.109 · H_M ≥ 0.554
[Phase 3] Benchmark         →  TSTR on real TCGA pathology reports (extracted TNM stages)
[Phase 4] Fine-tune         →  QLoRA adapters on 162 certified records
```

No record reaches fine-tuning without passing both gates.

---

## What This Found

**The label-collapse failure.** All 128 ablation rows were seeded from a single patient — "65yo Male, T2N0M0." Every record passed schema validation, SNOMED coverage, and AJCC logic. The corpus looked clean. The adapter predicted T2N0M0 for everything. The failure was distributional, not structural, and nothing in the existing literature catches it.

**The fix:** a 32-cell TNM grid ({T1–T4} × {N0–N3} × {M0, M1}), round-robin seeded, with entropy gates enforced before training.

**The result on real clinical notes:**

| Stage | Zero-Shot Baseline | Adapter B |
|-------|--------------------|-----------|
| T3    | 62.5%              | **100.0%** |
| T4    | 0.0%               | **54.5%** |

On 283 multi-cancer notes: +41.0 pp on T3, +54.5 pp on T4.

**The deeper finding:** label diversity is necessary but not sufficient. Adapter A′ — trained on the same 162 records with labels permuted — achieves comparable aggregate accuracy through frequency-matching while collapsing on T3 by 33 percentage points. The operative variable is the semantic correspondence between what a note describes and what label it carries.

---

## The Ablation Studies

Three independent studies, each running the full pipeline under controlled conditions:

| Study | Question | Adapters |
|-------|----------|---------|
| Gate vs No-Gate | Does the validation gate actually improve corpus quality? | A (ungated) vs D (full gate) |
| Gate Decomposition | Which gate component does the most work? | B (schema) vs C (schema+ontology) vs D (full G(x)) |
| RAG vs No-RAG | Does MedCPT retrieval enrich the corpus meaningfully? | D (no RAG) vs E (RAG) |

Each study runs through Phase 1 (generation), Phase 2 (audit), and Phase 4 (training) independently — then Phase 3 evaluates all adapters together on TCGA samples.

---

## Repository

```
pipeline/               Original Phase 1–4 (reference implementation)
  phase1_datagen.py
  phase2_analysis.py
  phase3_benchmark.py
  phase4_finetuning.py

ablation_phases/        Ablation-aware Phase 1–4 (condition-parameterized)
  phase1_generate.py    accepts --condition ungated/schema_only/schema_onto/full_norag/full_rag
  phase2_audit.py       cross-condition quality comparison
  phase3_benchmark.py   all five adapters on TCGA data
  phase4_finetune.py    trains one adapter per condition

ablations/              Three independent ablation studies
  1_gate_vs_nogate/
    run_ungated.py      generates + trains Adapter A
    run_gated.py        generates + trains Adapter D
    compare.py          side-by-side quality table + per-model breakdown
  2_gate_decomposition/
    run_schema_only.py  generates + trains Adapter B (C1)
    run_schema_onto.py  generates + trains Adapter C (C2)
    run_full_gate.py    generates + trains Adapter D (C3)
    compare.py          sequential gate decomposition table
  3_rag_vs_norag/
    run_norag.py        generates + trains Adapter D
    run_rag.py          generates + trains Adapter E (MedCPT RAG)
    compare.py          SNOMED density comparison + Mann-Whitney U

core/                   Shared utilities
  gate.py               G(x) gate components (schema, ontology, logic)
  generation.py         Model loading and generation (4-bit NF4, GPT-4o)
  schemas.py            rigid.v3 and longitudinal.v1 schema definitions
  tnm_grid.py           32-cell TNM grid, diversity audit, entropy gates
  bioportal.py          SNOMED CT annotation
  medcpt.py             MedCPT RAG retrieval over FAISS PubMed index
  logging_utils.py      JSONL logging and checkpointing

preprocessing/          TCGA pathology reports extraction and real-world TSTR setup
experiments/            Prompt engineering · Model comparison · Longitudinal generation
utils/                  BioPortal post-processing
data/                   Ablation design CSVs (128-cell full factorial · 11-point OFAT)
config.py               All paths, model IDs, and hyperparameters
```

---

## Running the Ablation Studies

Each ablation study runs independently. Run all three models per condition for meaningful gate rejection rates.

```bash
# Ablation 1 — Gate vs No-Gate
python ablations/1_gate_vs_nogate/run_ungated.py --model meta-llama/Llama-3.3-70B-Instruct --runs 128
python ablations/1_gate_vs_nogate/run_ungated.py --model wanglab/ClinicalCamel-70B          --runs 128
python ablations/1_gate_vs_nogate/run_ungated.py --model gpt-4o                             --runs 128

python ablations/1_gate_vs_nogate/run_gated.py   --model meta-llama/Llama-3.3-70B-Instruct --runs 128
python ablations/1_gate_vs_nogate/run_gated.py   --model wanglab/ClinicalCamel-70B          --runs 128
python ablations/1_gate_vs_nogate/run_gated.py   --model gpt-4o                             --runs 128

python ablations/1_gate_vs_nogate/compare.py

# Ablation 2 — Gate Decomposition (same pattern, three conditions)
# Ablation 3 — RAG vs No-RAG (add --faiss-index for MedCPT)
```

---

## What Is Next

- Colorectal and breast cancer — TNM grids and staging rules exist; the architecture transfers
- Demographic stratification — 91.5% of generated records are Male; a demographic grid is the fix
- MedCPT Cross-Encoder reranking — the retrieval collapse to a single paper (80.5% concentration) needs addressing
- Multi-institutional evaluation — MIMIC-III and eICU to test whether synthetic-to-real transfer holds at scale

---

*Laxmigayathri Challa · PhD, Information Science (Data Science) · University of North Texas*
*Paper: IEEE BIBM 2026*
*All generated data is synthetic and not intended for clinical use, YET.*
