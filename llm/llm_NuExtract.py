# Import necessary libraries
import pandas as pd
import numpy as np
import re
import os
import math
import json
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ftfy
import gc

# ---------- 2.0 Configurate ----------

current_dir = os.path.dirname(os.path.abspath("__file__")) if "__file__" in globals() else os.getcwd()
data_file = os.path.join(current_dir, "data", "df_cleaned.csv")
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)
trunc_log_path = os.path.join(output_dir, "truncated_rows_log.txt")

model_name = "numind/NuExtract-1.5"
batch_size = 1
max_length = 2048
max_new_tokens = 512
chunk_size = 5  # process 5 rows at a time

final_file = os.path.join(output_dir, "df_final_with_extracted_features.csv")
json_file = os.path.join(output_dir, "raw_outputs.jsonl")

# ---------- 2.1 Load data ----------

df_cleaned = pd.read_csv(data_file)
print("Loaded:", data_file)
print(df_cleaned.head())

# ---------- 2.2 Load NuExtract-1.5 on GPU ----------

print("Loading model......")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

print("Model and tokenizer loaded.")

# ---------- 2.3 JSON template and keyword lists ----------

template_dict = {
    "luxury_features": "",
    "transport_mentions": "",
    "school_mentions": "",
    "renovation_mentions": ""
}
template_str = json.dumps(template_dict, indent=4)

luxury_keywords = [
    # ===== Level 1 (Ultra-luxury / top-tier signals) =====
    "penthouse", "most exclusive", "exclusive development", "epitome of luxury", "unparalleled luxury",
    "luxury living", "world-class", "iconic", "coveted address", "prestigious address", "prime position",
    "prime residential address", "mayfair", "knightsbridge", "belgravia", "chelsea barracks",
    "panoramic views", "breathtaking views", "360º view",

    # ===== Level 2 (Luxury services / security / access control) =====
    "24-hour concierge", "concierge", "security", "first class security", "gated", "secure gates",
    "secure", "private gated road", "private road",

    # ===== Level 3 (Luxury amenities / lifestyle facilities) =====
    "swimming pool", "spa", "gym", "gymnasium", "sauna", "steam room", "treatment room", "cinema room",
    "home cinema", "wine cellar", "wine room", "billiards room", "personal training facilities",
    "leisure facilities", "business centre", "private meeting rooms", "chef's kitchen",

    # ===== Level 4 (High-end outdoor / layout / building features) =====
    "roof terrace", "private terrace", "terrace", "balcony", "private garden", "landscaped garden",
    "floor-to-ceiling windows", "high ceilings", "lift", "passenger lift", "glass lift", "garage",
    "secure parking",

    # ===== Level 5 (Interior quality / design / finishes) =====
    "bespoke", "bespoke joinery", "interior designed", "celebrated interior", "high specification",
    "state-of-the-art", "state of the art", "very high standard", "finished to the highest specification",
    "new benchmark of quality", "natural stone", "marble", "integrated appliances", "underfloor heating",
    "air conditioning",

    # ===== Level 6 (Premium rooms / layout language) =====
    "master bedroom suite", "dressing room", "en-suite bathroom", "reception rooms", "formal dining",
    "drawing room", "library", "study", "staff accommodation", "staff flat", "self-contained staff lodge",
    "mews house", "porticoed entrance", "impressive entrance hall",

    # ===== Level 7 (Luxury adjectives / marketing words) =====
    "luxury", "luxurious",
]

transport_keywords = [
    # ===== Level 1: Stations / lines / rail / roads / airports =====
    "station", "underground", "tube", "overground", "rail", "train", "london underground",
    "underground stations", "tube stations", "victoria station", "sloane square station",
    "knightsbridge station", "hyde park corner station", "green park station",
    "notting hill gate station", "holland park station", "regent's park station",
    "piccadilly line", "central line", "bakerloo line", "circle line", "district line",
    "national rail services", "heathrow airport", "m4", "m3", "bus", "bus routes",

    # ===== Level 2: Walking distance / access / connectivity phrases =====
    "transport links", "excellent transport links", "good transport links", "fantastic travel links",
    "travel links", "road links", "motorway", "well-connected", "well connected", "accessible",
    "easy access", "easy access to", "quick access", "excellent access", "excellent connections",
    "good connections", "within walking distance", "walking distance", "short walk", "just a short walk",
    "easy walking distance", "within easy reach of", "close to", "close to transport",
    "well-positioned for", "moments from", "stone's throw away", "just a stone's throw away",
    "minutes away", "minutes from", "less than a mile away", "approximately 0.3 miles away",
    "approximately 0.4 miles away"
]

school_keywords = [
    "good schools", "schools", "school", "excellent schools", "well-served by excellent schools"
]

renovation_keywords = [
    "renovated", "newly renovated", "recently renovated",
    "refurbished", "newly refurbished", "recently refurbished",
    "modernised", "modernized", "upgraded", "redecorated",
    "refitted", "updated", "brand new", "rebuilt", "reconstructed",
    "remodeled", "redesigned", "reimagined", "renewed", "reconditioned",
    "excellent condition", "good condition", "immaculate condition",
    "turnkey", "move-in ready", "ready to move into",
    "restored", "redeveloped", "reconfigured",
    "completely refurbished", "fully refurbished",
    "finished", "new kitchen", "new bathrooms"
]

# Use a reduced set of strong luxury keywords in the prompt (no extra GPU/RAM cost)
top_luxury_keywords = luxury_keywords[:40]

def build_prompt(text: str) -> str:
    instructions = f"""
You are extracting structured information from a London real-estate listing.

GENERAL RULES
- Read the ENTIRE listing carefully.
- For each field, return as MANY relevant key phrases as you can find (up to 8), separated by semicolons (;).
- Each phrase must be copied VERBATIM from the text (no paraphrasing).
- Do NOT invent information. If nothing relevant is found for a field, set that field to "" (empty string).
- Avoid full sentences; use compact phrases only.
- If at least 3 relevant phrases exist for a field, return at least 3.

FIELD DEFINITIONS

1) "luxury_features":
   - Phrases that indicate luxury amenities, services, finishes, or exclusive character.
   - Include EVERY phrase that includes or is clearly similar to any of these keywords:
     {", ".join(top_luxury_keywords)}
   - Do not skip phrases that match these keywords, even if they seem repetitive.
   - Examples of good outputs:
     "indoor swimming pool"; "spa with sauna and steam room"; "24 hour concierge";
     "landscaped private garden"; "home cinema"; "temperature-controlled wine cellar".

2) "transport_mentions":
   - Phrases that describe public transport, road access, or how easy it is to reach key areas.
   - Include EVERY phrase that includes or is clearly similar to any of these keywords:
     {", ".join(transport_keywords)}
   - Examples:
     "short walk to Knightsbridge Underground Station";
     "excellent transport links to the City and the West End";
     "within walking distance of Victoria Station".

3) "school_mentions":
   - Phrases that mention schools, school quality, or proximity to education.
   - Include EVERY phrase that includes or is clearly similar to any of these keywords:
     {", ".join(school_keywords)}
   - Examples:
     "excellent local schools";
     "close to top independent schools";
     "within the catchment area of outstanding primary schools".

4) "renovation_mentions":
   - Phrases that describe renovation, refurbishment, modernisation, or condition of the property.
   - Include EVERY phrase that includes or is clearly similar to any of these keywords:
     {", ".join(renovation_keywords)}
   - Examples:
     "newly refurbished throughout";
     "recently renovated to a high specification";
     "turnkey condition";
     "comprehensively redeveloped".

OUTPUT FORMAT
- Return a single JSON object exactly matching this template:
{template_str}

- Each value must be a single string containing zero or more phrases separated by semicolons.
- Do NOT add extra keys or commentary.
"""
    return f"""<|input|>
{instructions.strip()}

### Text:
{text}

<|output|>"""

# ---------- 2.4 Append the truncated row into log file ----------

def log_truncation(idx, original_len, kept_len):
    with open(trunc_log_path, "a", encoding="utf-8") as f:
        f.write(
            f"[{datetime.now().isoformat()}] "
            f"row_index={idx}, original_chars={original_len}, kept_chars={kept_len}\n"
        )

# ---------- 2.5 Run NuExtract on a batch of texts ----------

def nuextract_batch(texts, row_indices, batch_size, max_length, max_new_tokens):
    device_local = model.device
    prompts = []
    # track which prompts were truncated at character level
    for idx, t in zip(row_indices, texts):
        prompt = build_prompt(t)
        # rough char-length check before tokenization
        if len(prompt) > max_length * 4:  # heuristic: ~4 chars per token
            log_truncation(idx, len(prompt), max_length * 4)
            # keep the FIRST part (instructions + start of listing) to reduce confusion
            prompt = prompt[:max_length * 4]
        prompts.append(prompt)

    outputs = []

    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length
            ).to(device_local)

            pred_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )

            decoded = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            for out in decoded:
                if "<|output|>" in out:
                    outputs.append(out.split("<|output|>", 1)[1].strip())
                else:
                    outputs.append(out.strip())
    return outputs

# ---------- 2.6 Robust JSON extraction (one line per input) + hybrid merge ----------

empty_obj = {
    "luxury_features": "",
    "transport_mentions": "",
    "school_mentions": "",
    "renovation_mentions": ""
}

def extract_last_json(text: str):
    # Take the model output and: 1) find the last {...} block 2) parse it as JSON 3) ensure all expected keys exist.
    start = text.rfind("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return empty_obj.copy(), None

    candidate = text[start:end+1]
    # collapse whitespace to keep it one line
    candidate_one_line = re.sub(r"\s+", " ", candidate).strip()

    try:
        obj = json.loads(candidate_one_line)
    except json.JSONDecodeError:
        return empty_obj.copy(), None

    # Ensure keys
    for k in empty_obj.keys():
        obj.setdefault(k, "")

    return obj, candidate_one_line

# Simple keyword matcher (runs on CPU; very cheap)
def find_keyword_phrases(text: str, keywords):
    if not isinstance(text, str):
        return []
    t = text.lower()
    hits = []
    for kw in keywords:
        if kw.lower() in t:
            hits.append(kw)
    # remove duplicates while preserving order
    seen = set()
    unique_hits = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            unique_hits.append(h)
    return unique_hits

def merge_llm_and_keywords(listing_text: str, obj: dict, max_phrases: int = 8) -> dict:
    # Helper to split current semicolon string into list
    def split_field(value: str):
        if not isinstance(value, str):
            return []
        return [x.strip() for x in value.split(";") if x.strip()]

    # 1) luxury
    lux_from_llm = split_field(obj.get("luxury_features", ""))
    lux_kw_hits = find_keyword_phrases(listing_text, luxury_keywords)
    lux_merged = []
    for x in lux_from_llm + lux_kw_hits:
        if x not in lux_merged:
            lux_merged.append(x)
    obj["luxury_features"] = "; ".join(lux_merged[:max_phrases])

    # 2) transport
    trans_from_llm = split_field(obj.get("transport_mentions", ""))
    trans_kw_hits = find_keyword_phrases(listing_text, transport_keywords)
    trans_merged = []
    for x in trans_from_llm + trans_kw_hits:
        if x not in trans_merged:
            trans_merged.append(x)
    obj["transport_mentions"] = "; ".join(trans_merged[:max_phrases])

    # 3) schools
    school_from_llm = split_field(obj.get("school_mentions", ""))
    school_kw_hits = find_keyword_phrases(listing_text, school_keywords)
    school_merged = []
    for x in school_from_llm + school_kw_hits:
        if x not in school_merged:
            school_merged.append(x)
    obj["school_mentions"] = "; ".join(school_merged[:max_phrases])

    # 4) renovation
    reno_from_llm = split_field(obj.get("renovation_mentions", ""))
    reno_kw_hits = find_keyword_phrases(listing_text, renovation_keywords)
    reno_merged = []
    for x in reno_from_llm + reno_kw_hits:
        if x not in reno_merged:
            reno_merged.append(x)
    obj["renovation_mentions"] = "; ".join(reno_merged[:max_phrases])

    return obj

# ---------- 2.7 Loop over dataset in chunks of chunk_size rows to run model ----------

# 1) Write CSV header once (empty file with header)
if not os.path.exists(final_file):
    sample_df = df_cleaned.iloc[:1, :]
    dummy_extracted = pd.DataFrame([empty_obj])
    dummy_final = pd.concat([sample_df.reset_index(drop=True), dummy_extracted], axis=1)
    dummy_final.iloc[0:0].to_csv(final_file, index=False)  # write only header

# 2) Ensure JSONL file exists but DO NOT clear if you want resume
if not os.path.exists(json_file):
    open(json_file, "w", encoding="utf-8").close()

n_rows = len(df_cleaned)

# Count how many rows were already processed (1 line per row)
with open(json_file, "r", encoding="utf-8") as f:
    processed_count = sum(1 for _ in f)

print("Already processed rows:", processed_count)

resume_from = processed_count  # first row index to process next

for start in range(resume_from, n_rows, chunk_size):  # use n_rows to replace 500 after testing
    end = min(start + chunk_size, n_rows)

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"GPU Memory Free: {free_mem / 1024**2:.2f} MB")
    print(f"Processing rows {start} to {end - 1}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df_sample = df_cleaned.iloc[start:end, :]
    texts = df_sample["listingDescription"].fillna("").astype(str).tolist()
    row_indices = df_sample.index.tolist()

    print("Running NuExtract on", len(texts), "descriptions...")
    raw_outputs = nuextract_batch(
        texts,
        row_indices,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    print("Got outputs:", len(raw_outputs))

    for i, out in enumerate(raw_outputs):
        print(f"RAW {i}:", out)

    # Parse JSON, one line per input, then apply hybrid merge
    parsed = []
    clean_json_strings = []

    for raw_text, listing_text in zip(raw_outputs, texts):
        obj, clean_str = extract_last_json(raw_text)
        # hybrid merge with deterministic keyword hits
        obj = merge_llm_and_keywords(listing_text, obj, max_phrases=8)

        parsed.append(obj)
        if clean_str is None:
            clean_str = json.dumps(obj, ensure_ascii=False)
        clean_json_strings.append(clean_str)

    # Append JSONL
    with open(json_file, "a", encoding="utf-8") as f:
        for line in clean_json_strings:
            f.write(line.strip() + "\n")

    print("Appended to JSONL:", json_file)

    # Build and append df_final chunk
    df_extracted = pd.DataFrame(parsed)
    df_final_chunk = pd.concat([df_sample.reset_index(drop=True), df_extracted], axis=1)
    print("df_final_chunk:\n", df_final_chunk)

    df_final_chunk.to_csv(
        final_file,
        mode="a",
        index=False,
        header=False,  # header already written once
    )

    # Free Python & CUDA memory before next chunk
    del raw_outputs, df_extracted, df_final_chunk
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
