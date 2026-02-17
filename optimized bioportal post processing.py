import os, re, json, uuid, csv, torch, requests, gc
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# 1. ENHANCED CONFIGURATION
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY") 
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN or not BIOPORTAL_API_KEY:
    print("CRITICAL: Ensure HF_TOKEN and BIOPORTAL_API_KEY are set in your environment.")

login(HF_TOKEN)
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_LOG = f"postprocessed_results_{session_time}.txt"

# =========================
# 2. THE NEURO-SYMBOLIC GUARDRAIL (BioPortal)
# =========================
def annotate_bioportal_snomed(text: str) -> List[Dict]:
    """Calls BioPortal Annotator with a lung-cancer specific focus."""
    url = "https://data.bioontology.org/annotator"
    params = {
        "text": text,
        "ontologies": "SNOMEDCT",
        "longest_only": "true",
        "whole_word_only": "true",
        "apikey": BIOPORTAL_API_KEY,
    }
    try:
        # Retry logic for BioPortal API stability
        for _ in range(3):
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                break
        r.raise_for_status()
        data = r.json()
        
        results = []
        for ann in data:
            clz = ann.get("annotatedClass", {})
            results.append({
                "snomed_id": clz.get("@id", "").rsplit("/", 1)[-1],
                "prefLabel": clz.get("prefLabel", ""),
                "matched_text": ann.get("annotations", [{}])[0].get("text", "")
            })
        return results
    except Exception as e:
        print(f"BioPortal Error: {e}")
        return []

def apply_guardrails(note_text: str) -> str:
    """Injects SNOMED codes directly into the text (Neuro-Symbolic Layer)."""
    annotations = annotate_bioportal_snomed(note_text)
    if not annotations:
        return note_text
    
    # Filter for Lung-Cancer relevancy to avoid noise
    relevant_keywords = r"\b(lung|pulmon|bronch|adenocarcinoma|squamous|neoplasm|staging)\b"
    filtered = [a for a in annotations if re.search(relevant_keywords, a['prefLabel'], re.I)]
    
    # Simple Block Injection for the final note
    code_block = "\n\n[Ontology Grounding]\n" + "\n".join([f"- {a['prefLabel']}: {a['snomed_id']}" for a in filtered])
    return note_text + code_block

# =========================
# 3. GENERATION ENGINE
# =========================
# [Keep your existing Model Loading and Prompt Functions here]
# ... (Use your BitsAndBytesConfig and AutoModelForCausalLM setup) ...

def run_guarded_experiment(prompt_mode, params):
    """Generates the note and then applies the BioPortal safety layer."""
    # 1. Generate Raw Output
    raw_output = generate(prompt_mode, **params) # Using your previous generate() function
    
    # 2. Apply Post-Processing (The "Safety Layer")
    guarded_output = apply_guardrails(raw_output)
    
    # 3. Log to file
    with open(OUTPUT_LOG, "a") as f:
        f.write(f"\n--- RUN: {prompt_mode} ---\n{guarded_output}\n")
    
    return guarded_output

# =========================
# 4. EXECUTION LOOP
# =========================
if __name__ == "__main__":
    # Load your experiment CSVs
    combo_df = pd.read_csv("Full_Factorial_Combinations.csv")
    
    for _, row in combo_df.head(5).iterrows(): # Testing with first 5 for speed
        exp_params = {
            "max_length": int(row["max_length"]),
            "temperature": float(row["temperature"]),
            "top_p": float(row["top_p"]),
            "top_k": int(row["top_k"]),
            "do_sample": True
        }
        
        print(f"Running Guarded Generation for Temp: {exp_params['temperature']}")
        run_guarded_experiment("Few-Shot", exp_params)

    # Cleanup memory
    torch.cuda.empty_cache()
    gc.collect()
