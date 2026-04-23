#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Shot Entailment Feature Scoring for Real Estate Listings
Fixed version with better error handling and compatibility
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Force CPU mode to avoid GPU compatibility issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Now import torch
import torch
from transformers import pipeline
from tqdm import tqdm
import time

# ============================================
# CONFIGURATION
# ============================================
print("=" * 70)
print("ZERO-SHOT ENTAILMENT FEATURE SCORING FOR REAL ESTATE LISTINGS")
print("=" * 70)

# Set paths - FIXED THE PATH
current_dir = os.path.dirname(os.path.abspath("__file__")) if "__file__" in globals() else os.getcwd()
# Fix the path - remove extra "llm" duplication
output_dir = os.path.join(current_dir, "output")
input_file = os.path.join(output_dir, "df_cleaned.csv")
output_file = os.path.join(output_dir, "df_feature_extract_zero_shot_score.csv")

print(f"Current directory: {current_dir}")
print(f"Output directory: {output_dir}")
print(f"Input file: {input_file}")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# ============================================
# LOAD INPUT DATA
# ============================================
print(f"\n[1/5] Loading input data...")

try:
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"✓ Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"✓ Columns: {', '.join(df.columns.tolist())}")
except FileNotFoundError:
    print(f"✗ Error: Input file not found at {input_file}")
    print("Please ensure df_cleaned.csv exists in the output directory.")
    exit(1)
except Exception as e:
    print(f"✗ Error loading file: {e}")
    exit(1)

# ============================================
# INITIALIZE ZERO-SHOT CLASSIFIER
# ============================================
print(f"\n[2/5] Loading zero-shot entailment model...")

# Use a more compatible model
model_name = "cross-encoder/nli-distilroberta-base"  # More compatible than BART
print(f"✓ Loading model: {model_name}")
print("  (This may take 30-60 seconds on first run)")

try:
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1,  # Force CPU
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Trying fallback to another model...")
    try:
        # Last resort - tiny model
        classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1,
        )
        print("✓ Fallback model loaded successfully!")
    except Exception as e:
        print(f"✗ Critical error loading model: {e}")
        exit(1)

# ============================================
# DEFINE FEATURE HYPOTHESES
# ============================================
print(f"\n[3/5] Defining feature hypotheses...")

feature_hypotheses = {
    'luxury': [
        "This property has luxury amenities.",
        "This property offers high-end features.",
        "This is a luxury home.",
        "This property has premium finishes.",
        "This property includes concierge or pool facilities.",
    ],
    
    'transport': [
        "This property has good transport links.",
        "This property is close to public transportation.",
        "This property is near a tube station.",
        "This property has excellent connectivity.",
        "This property is within walking distance of transport.",
    ],
    
    'school': [
        "This property is near good schools.",
        "This property is in a good school catchment area.",
        "This property has access to excellent schools.",
        "This property is close to educational facilities.",
        "This property is well-served by local schools.",
    ],
    
    'renovation': [
        "This property has been recently renovated.",
        "This property is newly refurbished.",
        "This property has been modernized.",
        "This property is in excellent condition.",
        "This property has updated interiors.",
    ]
}

# Combine ALL hypotheses into one list
all_hypotheses = []
for hyps in feature_hypotheses.values():
    all_hypotheses.extend(hyps)

# Track which indices belong to which feature
feature_indices = {}
start_idx = 0
for feature, hyps in feature_hypotheses.items():
    end_idx = start_idx + len(hyps)
    feature_indices[feature] = list(range(start_idx, end_idx))
    start_idx = end_idx

print(f"✓ Combined {len(all_hypotheses)} hypotheses across 4 features")
for feature, hyps in feature_hypotheses.items():
    print(f"  - {feature}: {len(hyps)} hypotheses")

# ============================================
# SCORING FUNCTION
# ============================================
def clean_description(desc):
    """Clean and truncate description for model input."""
    if pd.isna(desc) or not isinstance(desc, str) or desc.strip() == "":
        return None
    
    # Remove common encoding artifacts
    desc = desc.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Truncate if too long
    if len(desc) > 2000:  # Conservative limit
        desc = desc[:2000] + "..."
    
    return desc.strip()

# ============================================
# PROCESS ALL LISTINGS
# ============================================
print(f"\n[4/5] Processing {len(df)} listings with zero-shot entailment...")
print(f"Using ONE model call per listing")

# Initialize score columns
for feature in feature_hypotheses.keys():
    df[f"{feature}_score"] = 0.0

# Process each listing with progress bar
start_time = time.time()
success_count = 0
error_count = 0

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring listings"):
    description = clean_description(row['listingDescription'])
    
    if description is None:
        error_count += 1
        continue
    
    try:
        # SINGLE model call for ALL hypotheses
        result = classifier(
            description,
            all_hypotheses,
            hypothesis_template="{}",
            multi_label=True
        )
        
        # Extract scores for each feature
        scores = result['scores']
        for feature, indices in feature_indices.items():
            feature_scores = [scores[i] for i in indices]
            df.at[idx, f"{feature}_score"] = round(max(feature_scores), 4)
        
        success_count += 1
        
    except Exception as e:
        error_count += 1
        if error_count <= 5:  # Only show first 5 errors
            print(f"\nWarning: Error processing row {idx}: {type(e).__name__}: {e}")
        continue

elapsed_time = time.time() - start_time
print(f"\n✓ Processing complete in {elapsed_time:.1f} seconds")
print(f"✓ Successful: {success_count} rows")
print(f"✓ Failed: {error_count} rows")

# ============================================
# VERIFY AND SAVE RESULTS
# ============================================
print(f"\n[5/5] Saving results...")

# Display sample of non-zero scores if they exist
print(f"\nSample of scored listings (first 10 rows):")
sample_cols = ['price', 'luxury_score', 'transport_score', 'school_score', 'renovation_score']
sample_df = df[sample_cols].head(10)
print(sample_df.to_string())

# Show statistics
print(f"\nScore statistics:")
for feature in feature_hypotheses.keys():
    col = f"{feature}_score"
    non_zero = (df[col] > 0).sum()
    print(f"  {col}: mean={df[col].mean():.4f}, non-zero={non_zero}/{len(df)}")

# Save to CSV
try:
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✓ File saved successfully to: {output_file}")
except Exception as e:
    print(f"✗ Error saving file: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("PROCESSING SUMMARY")
print("=" * 70)
print(f"Input file:  {input_file}")
print(f"Output file: {output_file}")
print(f"Listings processed: {len(df)}")
print(f"Successful: {success_count}")
print(f"Failed: {error_count}")
print(f"Processing time: {elapsed_time:.1f} seconds")
print(f"Model used: {model_name}")
print("=" * 70)
print("\n✓ Script completed successfully!")