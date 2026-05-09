# Synthetic Data for Lung Cancer Staging

*What happens when the data you need to build a clinical AI system is exactly the data you can't have?*

Lung cancer kills more people than any other cancer. Survival swings from 80% at Stage I to under 20% at Stage IV — and three in four patients are diagnosed at the late stages. The models that could change that need structured, annotated clinical notes. Those notes are locked behind privacy law, annotation bottlenecks, and institutional walls that take years to navigate.

This project builds the data instead. Not as a shortcut — as a principled engineering commitment.

---

## The Pipeline

Four phases. Each one gates the next.

```
[Phase 1] Generate          →  RAG-grounded LLM synthesis across 639 runs
              ↓ G(x) gate: JSON schema + SNOMED CT + AJCC logic
[Phase 2] Audit             →  Shannon entropy per T / N / M — catches silent label collapse
              ↓ entropy gate: H_T,N ≥ 1.109 · H_M ≥ 0.554
[Phase 3] Benchmark         →  TSTR on real MTSamples clinical notes
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

## Repository

```
pipeline/           Four-phase pipeline (datagen → analysis → benchmark → finetune)
preprocessing/      MTSamples extraction and real-world TSTR setup
experiments/        Ablations: Gate vs No-Gate · Gate Decomposition · RAG vs No-RAG
                    Prompt engineering · Model comparison · Longitudinal generation
utils/              BioPortal SNOMED CT annotation
data/               Ablation design CSVs (128-cell full factorial · 11-point OFAT)
```

**Run order:** `phase1_datagen.py` → `phase2_analysis.py` → `mtsamples_prep.py` → `phase3_benchmark.py` → `phase4_finetuning.py`

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
