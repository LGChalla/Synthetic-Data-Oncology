"""
utils/bioportal.py
BioPortal SNOMED CT annotation utility.

Standalone helper used by phase1_datagen.py and the ablation scripts.
Set BIOPORTAL_API_KEY in your environment before use.
"""

import os
import re
import requests
from typing import List, Dict

BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY", "")
_BASE_URL = "https://data.bioontology.org/annotator"
_LUNG_CANCER_PATTERN = re.compile(
    r"\b(lung|pulmon|bronch|adenocarcinoma|squamous|neoplasm|staging|carcinoma|metastasis)\b",
    re.IGNORECASE,
)


def annotate_snomed(text: str, filter_lung_cancer: bool = False) -> List[Dict]:
    """
    Calls BioPortal Annotator and returns SNOMED CT concept matches.

    Args:
        text: Clinical text to annotate.
        filter_lung_cancer: If True, only returns concepts whose prefLabel
                            matches lung-cancer-relevant keywords.

    Returns:
        List of dicts with keys: snomed_id, prefLabel, matched_text.
    """
    if not BIOPORTAL_API_KEY:
        print("[BioPortal] BIOPORTAL_API_KEY not set — returning empty.")
        return []
    if not text or not text.strip():
        return []

    params = {
        "text":            text,
        "ontologies":      "SNOMEDCT",
        "longest_only":    "true",
        "whole_word_only": "true",
        "exclude_numbers": "true",
    }
    headers = {"Authorization": f"apikey token={BIOPORTAL_API_KEY}"}

    try:
        for attempt in range(3):
            r = requests.get(_BASE_URL, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                break
            if attempt == 2:
                r.raise_for_status()

        results: List[Dict] = []
        seen = set()
        for ann in r.json():
            clz        = ann.get("annotatedClass", {})
            pref_label = clz.get("prefLabel", "Unknown")
            iri        = clz.get("@id", "")
            snomed_id  = iri.rsplit("/", 1)[-1] if iri else ""
            matched    = ann.get("annotations", [{}])[0].get("text", "")
            key        = f"{snomed_id}_{matched}"
            if key in seen:
                continue
            seen.add(key)
            if filter_lung_cancer and not _LUNG_CANCER_PATTERN.search(pref_label):
                continue
            results.append({
                "snomed_id":    snomed_id,
                "prefLabel":    pref_label,
                "matched_text": matched,
            })
        return results

    except requests.RequestException as e:
        print(f"[BioPortal] Request error: {e}")
        return []
    except (KeyError, ValueError) as e:
        print(f"[BioPortal] Parse error: {e}")
        return []


def inject_ontology_block(note_text: str) -> str:
    """
    Annotates note_text with SNOMED CT codes and appends a grounding block.
    Used by the neuro-symbolic post-processing layer.
    """
    annotations = annotate_snomed(note_text, filter_lung_cancer=True)
    if not annotations:
        return note_text
    block = "\n\n[Ontology Grounding]\n" + "\n".join(
        f"- {a['prefLabel']}: {a['snomed_id']}" for a in annotations
    )
    return note_text + block


def snomed_density(text: str, annotations: List[Dict]) -> float:
    """Returns length-normalised SNOMED term count (terms per 100 words)."""
    word_count = max(len(text.split()), 1)
    return (len(annotations) / word_count) * 100
