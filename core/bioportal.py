"""core/bioportal.py — BioPortal SNOMED CT annotation."""

import os
import requests
from typing import List, Dict

_BASE_URL = "https://data.bioontology.org/annotator"


def annotate(text: str, api_key: str = "") -> List[Dict]:
    key = api_key or os.getenv("BIOPORTAL_API_KEY", "")
    if not key or not text.strip():
        return []

    params  = {
        "text": text, "ontologies": "SNOMEDCT",
        "longest_only": "true", "whole_word_only": "true", "exclude_numbers": "true",
    }
    headers = {"Authorization": f"apikey token={key}"}

    try:
        for attempt in range(3):
            r = requests.get(_BASE_URL, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                break
        r.raise_for_status()
        out, seen = [], set()
        for ann in r.json():
            clz   = ann.get("annotatedClass", {})
            label = clz.get("prefLabel", "Unknown")
            iri   = clz.get("@id", "")
            sid   = iri.rsplit("/", 1)[-1] if iri else ""
            match = ann.get("annotations", [{}])[0].get("text", "")
            key_  = f"{sid}_{match}"
            if key_ not in seen:
                seen.add(key_)
                out.append({"snomed_id": sid, "prefLabel": label, "matched_text": match})
        return out
    except requests.RequestException as e:
        print(f"  [BioPortal] {e}")
        return []
    except (KeyError, ValueError) as e:
        print(f"  [BioPortal parse] {e}")
        return []


def density(text: str, codes: List[Dict]) -> float:
    """Terms per 100 words."""
    return (len(codes) / max(len(text.split()), 1)) * 100
