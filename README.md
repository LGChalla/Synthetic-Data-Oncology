# Synthetic Data Generation and Quality Assurance for Lung Cancer Staging Using LLMs

Laxmigayathri Challa · PhD Student, Information Science (Data Science)  
University of North Texas · [Paper (IEEE BIBM 2026)](https://github.com/LGChalla/Synthetic-Data-Oncology)

---

## Why This Exists

Clinical AI for cancer staging has a data problem that is not algorithmic — it is epistemic. The records that would let us train accurate staging models are precisely the records we cannot reach. Real-world clinical notes are locked behind HIPAA and GDPR. Even when access is granted, structured TNM annotations are rarely embedded in free text, and manual annotation requires oncology expertise at a scale that is practically unattainable.

Lung cancer makes that cost concrete. It is the leading cause of cancer mortality worldwide — 2.48 million new cases and 1.8 million deaths in 2022 alone, with 124,730 projected US deaths in 2025. Five-year survival ranges from roughly 80% at AJCC Stage I to under 20% at Stage IV, and three in four patients are diagnosed at Stage III or IV. Staging is decided largely by the moment a tumour is recorded in unstructured clinical text, and staging errors are not symmetric. A false downstage — assigning T1 or T2 to a patient with T4 disease — can mean under-treatment, delayed systemic therapy, and inappropriate surgical planning, all with direct consequences for whether a patient survives.

Synthetic data is the response to that bottleneck, but only if it is built with enough rigour to be trusted. This project is about what that rigour actually requires in practice — and what silently breaks when you assume it is already in place.

---

## What Was Built and What Was Found

The pipeline has four phases. Phase 1 generates synthetic lung cancer staging notes using three LLMs — GPT-4o, Llama-3.3-70B-Instruct, and ClinicalCamel-70B — grounded through MedCPT retrieval over a FAISS index of ~2 million PubMed abstracts. Every candidate record passes through a binary neuro-symbolic gate enforcing rigid JSON schema completeness, SNOMED CT ontology coverage via BioPortal, and AJCC 8th Edition clinical-logic consistency. Records that fail the gate do not enter the corpus at all. Hallucination stops being a qualitative concern and becomes an auditable engineering property.

Phase 2 runs statistical quality analysis, including the diversity audit that turned out to matter most. Phase 3 stratifies the validated corpus and constructs the TSTR benchmark test sets — synthetic held-out, MTSamples lung cancer (n=53), and MTSamples all-cancer (n=283). Phase 4 trains QLoRA adapters on Llama-3-8B-Instruct using the certified corpus.

**The most important thing this pipeline found was a failure it was not designed to find.**

During development, all 128 controlled-layer ablation rows were seeded from a single patient profile — "65-year-old Male, T2 N0 M0, Adenocarcinoma." Every generated record passed schema validation, SNOMED coverage checks, and AJCC logic. The corpus looked fine. The downstream adapter trained on it predicted T2 N0 M0 for every input, because that was the only label it had ever seen. The failure was not structural — it was distributional, and nothing in the existing pipeline caught it.

The fix is a 32-cell TNM grid ({T1–T4} × {N0–N3} × {M0, M1}) assigned round-robin across ablation rows, with Shannon entropy gates enforced per label dimension (H_T, H_N ≥ 1.109; H_M ≥ 0.554, derived as 80% of theoretical maximum entropy). Any failure at the gate blocks pipeline progression and triggers targeted supplementary generation for under-represented cells.

A 128-cell full-factorial decoding ablation confirmed that schema compliance does not respond meaningfully to temperature, top-p, top-k, or stopping criteria (maximum Cramér's V = 0.106 across all factors). Hyperparameter search for compliance improvement is unproductive. Model selection and API-level format enforcement are the only effective levers.

On real clinical notes under a train-on-synthetic, test-on-real (TSTR) protocol, Adapter B — trained on 162 diversity-certified golden records — recovers T4 accuracy from 0% to 54.5% and T3 from 62.5% to 100% on 53 confirmed lung cancer notes. On 283 multi-cancer notes, T3 improves by +41.0 pp and T4 by +54.5 pp. The zero-shot baseline identifies T4 in 0% of cases on both real-world sets.

**The deeper finding came from the three-way ablation.**

Adapter A′ was trained on the same 162 schema-valid, diversity-certified records as Adapter B, with one change: the TNM label triples were permuted across records under a fixed random seed, destroying the correspondence between each note's clinical content and its label while preserving the marginal label distribution and Shannon entropy exactly. Adapter A′ achieves superficially reasonable aggregate accuracy by matching the training-corpus frequency distribution — it scores well on the most common classes — but collapses on T3 by 33 percentage points on the all-cancer set, where correct classification demands content-driven discrimination. Label diversity is necessary but not sufficient. The operative variable is the semantic correspondence between what a note describes and what label it carries.

MedCPT RAG grounding adds a measurable 18.4% increase in SNOMED CT vocabulary density over non-RAG runs, confirming that retrieval is doing real work. It also surfaced a failure mode worth flagging: in the initial single-chunk run, 80.5% of all retrieval calls returned the same paper. Retrieval collapse is not visible in vocabulary metrics alone — it requires direct inspection of retrieval logs.

---

## Repository Structure

```
Synthetic-Data-Oncology/
│
├── pipeline/
│   ├── phase1_datagen.py          # RAG-grounded generation across 3 LLMs, 3 schema layers
│   ├── phase2_analysis.py         # Schema validity, entropy audit, SNOMED density, ablation effects
│   ├── phase3_benchmark.py        # TSTR benchmark construction and evaluation (bootstrap CIs)
│   └── phase4_finetuning.py       # QLoRA adapter training (Adapter A, A′, B)
│
├── preprocessing/
│   ├── mtsamples_prep.py          # MTSamples TNM extraction, lung + all-cancer filters
│   └── realworld_benchmark.py     # Real-world TSTR evaluation across all three adapters
│
├── experiments/
│   ├── prompt_engineering.py      # Multi-factor prompt design across generation modes
│   ├── model_comparison.py        # ClinicalCamel vs Llama across CSV parameter grids
│   └── longitudinal_datagen.py    # Multi-visit timeline generation with temporal consistency
│
├── utils/
│   └── bioportal.py               # SNOMED CT annotation and BioPortal post-processing
│
├── data/
│   ├── full_factorial.csv         # 128-cell full-factorial decoding ablation design
│   └── one_factor_at_a_time.csv   # 11-point one-factor-at-a-time temperature sweep
│
├── requirements.txt
└── README.md
```

---

## What Is Next

The pipeline works for non-small cell lung cancer. The results on the 283-note all-cancer set suggest that the clinical language of advanced local invasion (T3, T4) transfers across tumour types, but early-stage language (T1) does not — it is more tumour-specific. That asymmetry points directly at what needs to be built next.

Expanding to colorectal and breast cancer is the immediate priority, where TNM grids and staging rules are well-standardised and the transfer question can be tested systematically. Each new cancer type requires its own TNM grid, histology stratification, and staging logic — the architecture is in place, the scope is not.

The generated corpus has a demographic bias that needs to be addressed before any of this scales: approximately 91.5% of records are Male, concentrated in a narrow age band. A demographic stratification grid analogous to the TNM grid is the natural fix — one that enforces representation across sex, age, race, and histology subtype rather than leaving those distributions to the model's unconstrained prior.

Scaling the golden corpus to ≥500 certified records per TNM cell would provide the statistical power needed for per-cell model comparisons. Applying MedCPT Cross-Encoder reranking would address the retrieval concentration problem surfaced in this work. Evaluating on MIMIC-III and eICU would test whether the synthetic-to-real transfer demonstrated here holds against multi-institutional EHR notes with validated ground truth — which the MTSamples evaluation, relying on regex-extracted labels, cannot fully provide.

---

*All generated data is synthetic and not intended for clinical use.*
