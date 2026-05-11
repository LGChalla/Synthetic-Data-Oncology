"""core/schemas.py — Schema definitions and prompt templates."""

RIGID_V3 = {
    "$schema_version": "rigid.v3",
    "notes": [{
        "staging":      {"prefix": "", "T": "", "N": "", "M": "", "stage_group": ""},
        "histology":    {"name": "", "icdo3": "", "snomed": ""},
        "molecular":    {"drivers": [{"gene": "", "result": "",
                                      "variant": None, "snomed_qualifier": ""}]},
        "demographics": {"age": "", "sex": "", "race/ethnicity": ""},
        "imaging":      [{"modality": "", "finding": "", "snomed": ""}],
        "treatment":    {"intent": "", "modalities": [{"type": "", "snomed": ""}]},
        "equity":       {"factors": [{"issue": "", "snomed": ""}]},
        "free_text":    "",
    }],
}

TIMELINE_V1 = {
    "$schema_version": "longitudinal.v1",
    "patient_id": "SYNTH-001",
    "notes": [{
        "encounter_type":  "Initial / Follow-up / Relapse",
        "time_offset":     "Month X",
        "staging":         {"prefix": "", "T": "", "N": "", "M": "", "stage_group": ""},
        "clinical_status": "Progression / Response / Stable",
        "treatment":       {"intent": "", "modalities": [{"type": "", "snomed": ""}]},
        "free_text":       "",
    }],
}

SYSTEM_PROMPT = (
    "You are an expert oncologist. "
    "Return ONLY a valid JSON object that strictly conforms to the provided schema. "
    "No prose, markdown, or code fences. "
    "All fields are mandatory — no empty strings. "
    "Follow AJCC 8th Edition strictly."
)

EXTRACTION_PROMPT = (
    "You are a clinical data extractor. Read the clinical note and extract the TNM staging. "
    "Return a strictly formatted JSON object with keys 'T', 'N', and 'M'. "
    "Use the full prefixed format: e.g. 'T2', 'N1', 'M0'. "
    "If a value is not found, use 'Unknown'.\n\nNOTE: {text}"
)
