import os
import csv
import torch
import gc
import pandas as pd
import uuid
from datetime import datetime
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ================= CONFIGURATION =================
MODELS_TO_RUN = [
    "wanglab/ClinicalCamel-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct" 
]

OFAT_CSV = "One-Factor-at-a-Time_Experiment.csv"
FULL_FACT_CSV = "Full_Factorial_Combinations.csv"

SYSTEM_PROMPT = (
    "You are an expert oncologist. "
    "Generate a synthetic clinical note for Lung Cancer Staging based on the user request. "
    "Follow AJCC 8th Edition strictly. "
    "Return the output as a valid JSON object if requested."
)

# ================= AUTH & SETUP =================
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN not set. Llama-3 and Camel require this!")
else:
    login(hf_token)

# ================= PROMPT TEMPLATES =================
def get_prompt(mode, run_params):
    base_case = "Patient: 65-year-old Male, 3cm tumor in right upper lobe (T2), no node involvement (N0), no metastasis (M0). Histology: Adenocarcinoma. "
    
    if mode == "Zero-Shot":
        return base_case + "Provide full staging note in JSON format."
    elif mode == "One-Shot":
        example = 'Example: {"stage_group": "IA", "T": "T1b", "N": "N0", "M": "M0", "histology": "Squamous Cell"}\n'
        return example + base_case + "Provide full staging note in JSON."
    elif mode == "Few-Shot":
        examples = (
            'Case 1: {"stage_group": "IA", "T": "T1b"}\n'
            'Case 2: {"stage_group": "IIIB", "T": "T4", "N": "N2"}\n'
            'Case 3: {"stage_group": "IV", "M": "M1a"}\n'
        )
        return examples + base_case + "Provide full staging note in JSON."
    elif mode == "Terminology-Challenge":
        return "Patient has a 'large lung lump' approx 4cm and 'spread to lymph glands'. Generate a standard TNM Staging JSON."
    elif mode == "Bias-Check":
        return "Generate a complete synthetic patient record for a new lung cancer admission. Include Age, Sex, and TNM Staging. Return as JSON."
    elif mode == "Chain-of-Thought":
        return base_case + "First, think step-by-step and explain the correct Stage Group based on AJCC 8th Edition. Then, provide the final JSON."
    
    return base_case

def format_input(tokenizer, model_name, user_prompt, system_prompt):
    """Handles formatting differences between Llama-3 and Camel."""
    
    # 1. Llama-3 (Standard Chat Template)
    if "Llama-3" in model_name:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

    # 2. ClinicalCamel (Often lacks chat template in config, uses Vicuna/Alpaca style)
    # Fallback to manual formatting if apply_chat_template fails or is generic
    try:
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
            return tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    except:
        pass # Fallback to manual

    # Manual Format for Camel/Alpaca style models
    full_text = f"USER: {system_prompt}\n\n{user_prompt}\nASSISTANT:"
    return tokenizer(full_text, return_tensors="pt").input_ids.to("cuda")

# ================= LOGGING =================
def initialize_csv(model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = model_name.replace("/", "_")
    filename = f"results_{clean_name}_{timestamp}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "timestamp", "model", "experiment_mode", "max_length", "temperature", "top_p", "top_k", "do_sample", "seed", "prompt", "output"])
    return filename

def save_result(filename, run_data):
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(run_data)

# ================= ENGINE =================
def run_experiments_for_model(model_name):
    print(f"\n{'='*60}\nSTARTING: {model_name}\n{'='*60}")
    
    # Load CSVs
    try:
        df_ofat = pd.read_csv(OFAT_CSV)
        df_full = pd.read_csv(FULL_FACT_CSV)
        experiment_queue = pd.concat([df_ofat, df_full], ignore_index=True)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        return

    output_csv = initialize_csv(model_name)
    print(f"Saving to: {output_csv}")

    # Load Model (Strict 4-bit)
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Fix padding token for Camel/Llama if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"FAILED to load {model_name}: {e}")
        return

    # Run Loop
    TEST_MODES = ["Zero-Shot", "One-Shot", "Few-Shot", "Terminology-Challenge", "Bias-Check", "Chain-of-Thought"]
    
    for idx, row in experiment_queue.iterrows():
        params = {
            "max_new_tokens": int(row["max_length"]),
            "temperature": float(row["temperature"]),
            "top_p": float(row["top_p"]),
            "top_k": int(row["top_k"]),
            "do_sample": str(row["do_sample"]).lower() in ["true", "1", "yes"]
        }
        
        print(f"[{idx+1}/{len(experiment_queue)}] Config: T={params['temperature']}")

        for mode in TEST_MODES:
            # Bias-Check needs repeats? Let's do 1 run per config for now to save time, 
            # or add a loop here if you want n=10.
            
            user_prompt = get_prompt(mode, params)
            input_ids = format_input(tokenizer, model_name, user_prompt, SYSTEM_PROMPT)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        pad_token_id=tokenizer.eos_token_id,
                        **params
                    )
                
                decoded = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                # Debug Print: Check if we got anything
                if not decoded.strip():
                    print(f"  [WARNING] Empty output for {mode}!")
                
                save_result(output_csv, [
                    str(uuid.uuid4()), datetime.now().isoformat(), model_name, mode,
                    params["max_new_tokens"], params["temperature"], params["top_p"], 
                    params["top_k"], params["do_sample"], "None",
                    user_prompt, decoded
                ])
            except Exception as e:
                print(f"  [ERROR] Generation failed for {mode}: {e}")

    # Cleanup
    print(f"Finished {model_name}. Cleaning memory...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    for m in MODELS_TO_RUN:
        run_experiments_for_model(m)
