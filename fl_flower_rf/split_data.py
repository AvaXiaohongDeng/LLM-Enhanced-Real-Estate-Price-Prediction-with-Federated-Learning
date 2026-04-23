# split_data.py
import os
import pandas as pd
import numpy as np

# Path to the cleaned data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "output", "preprocess_csv", "df_q1.csv")

# Load the full dataset
df = pd.read_csv(data_path)

# Shuffle rows
df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Compute sizes for three roughly equal parts
n = len(df_shuffled)
n1 = n // 3
n2 = n // 3
n3 = n - n1 - n2  # remainder goes to client 3

# Split into three dataframes
df1 = df_shuffled.iloc[:n1]
df2 = df_shuffled.iloc[n1:n1 + n2]
df3 = df_shuffled.iloc[n1 + n2:]

# Save them as separate client CSVs
output_dir = current_dir  # Save to the same directory as split_data.py
df1.to_csv(os.path.join(output_dir, "data1.csv"), index=False)
df2.to_csv(os.path.join(output_dir, "data2.csv"), index=False)
df3.to_csv(os.path.join(output_dir, "data3.csv"), index=False)

print("Saved data1.csv, data2.csv, data3.csv")
