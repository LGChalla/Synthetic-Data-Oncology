# Synthetic-Data-Oncology

A research-driven framework for generating, validating, and evaluating synthetic clinical oncology data using structured prompting and constraint-based verification.

## Overview

This repository contains experiments and pipelines developed as part of ongoing doctoral research focused on improving reliability and interpretability in AI-generated clinical data.

The project explores how large language models (LLMs) can be used to generate structured oncology notes (e.g., lung cancer staging summaries) while minimizing hallucinations and ensuring schema-level consistency.

The core objective is to move from creative text generation to **verifiable, constraint-based engineering** for clinical data synthesis.


## Research Motivation

Clinical AI systems often suffer from:
- Hallucinated medical facts
- Incomplete staging information
- Schema inconsistencies
- Lack of ontology alignment

This repository implements structured pipelines that:

1. Generate synthetic oncology notes
2. Enforce JSON schema constraints
3. Validate staging completeness
4. Perform ontology-aligned verification
5. Quantify hallucination and completeness metrics

The broader research goal is to support reliable synthetic data generation for:
- Model training
- Simulation experiments
- Evaluation of clinical NLP systems
- Early-stage oncology modeling


## Project Components

### 1️⃣ Structured Prompting Framework
- Template-based generation
- Zero-shot and few-shot experiments
- Constraint-based output formatting

### 2️⃣ JSON Schema Validation
- Strict schema enforcement
- Field-level completeness checks
- Structured output auditing

### 3️⃣ Ontology Alignment
- Mapping clinical terms to structured categories
- Evaluating coverage and terminology consistency

### 4️⃣ Evaluation Metrics
- JSON validity
- Schema completeness
- Ontology coverage
- Hallucination detection signals


## Technologies Used

- Python
- Hugging Face Transformers
- OpenAI / LLM APIs
- JSON Schema validation
- NumPy / Pandas
- Matplotlib (for evaluation visualization)


## Example Workflow

1. Provide structured prompt template
2. Generate oncology note using LLM
3. Validate output against JSON schema
4. Score completeness and coverage
5. Log evaluation metrics for comparison


## Research Context

This work is part of a broader dissertation effort aimed at:

- Improving reliability in clinical text generation
- Modeling structured oncology data pipelines
- Supporting trustworthy healthcare AI
- Reducing hallucinations in high-stakes domains

The results contribute to ongoing research on constraint-based LLM engineering for clinical applications.


## Repository Structure
Synthetic-Data-Oncology/
│
├── New multi input-tuned prompts.py
│   → Prompt engineering experiments with multi-factor structured inputs (This was the initial set-up to understand prompt engineering)
│
├── Prompt+factors for camel and llama new.py
│   → Comparative experiments across ClinicalCamel and LLaMA models (Within this, the two csv. files need to be used to understand the influence of a variety of combination of parameters)
│
├── longitudinal synthetic Data.py
│   → Synthetic longitudinal oncology note generation and evaluation
│
├── optimized bioportal post processing.py
│   → Ontology-aligned post-processing and concept verification (BioPortal integration)
│
├── Full_Factorial_Combinations.csv
│   → Full-factorial experimental design outputs
│
├── One-Factor-at-a-Time_Experiment.csv
│   → Controlled ablation study results
│
└── README.md

## Future Directions

- Integration with ontology APIs (e.g., SNOMED CT)
- Comparative benchmarking across model families
- Expansion to multimodal oncology data
- Automated hallucination taxonomy integration


## Disclaimer

This repository is for research and experimental purposes only.  
Generated content is synthetic and not intended for clinical decision-making.

## Author

Laxmigayathri Challa  
PhD Student – Information Science (Data Science Concentration)  
University of North Texas
