# split_data_lstm.py
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_data_for_lstm(csv_path: str) -> pd.DataFrame:
    """
    Prepare house price data for LSTM federated learning.
    - Loads CSV
    - Creates SaleDate from YrSold/MoSold
    - Sorts by date
    - Moves SalePrice to last column (required for LSTM)
    """
    print(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Data shape: {df.shape}")
    
    # Create datetime from year/month
    df['SaleDate'] = pd.to_datetime(
        df['YrSold'].astype(str) + '-' + 
        df['MoSold'].astype(str).str.zfill(2) + '-01'
    )
    
    # Sort chronologically for time series
    df = df.sort_values('SaleDate')
    print(f"Date range: {df['SaleDate'].min().date()} to {df['SaleDate'].max().date()}")
    
    # Move SalePrice to last column (LSTM expects target last)
    cols = [c for c in df.columns if c != 'SalePrice'] + ['SalePrice']
    df = df[cols]
    print(f"Target 'SalePrice' moved to last column")
    
    return df

def split_data_to_clients(df: pd.DataFrame, n_clients: int = 3) -> list:
    """
    Split dataframe into n_clients parts by time.
    Returns list of client dataframes.
    """
    client_dfs = []
    n_rows = len(df)
    rows_per_client = n_rows // n_clients
    
    for i in range(n_clients):
        start = i * rows_per_client
        end = (i + 1) * rows_per_client if i < n_clients - 1 else n_rows
        
        client_df = df.iloc[start:end].copy()
        client_dfs.append(client_df)
        
        # Print client stats
        print(f"\nClient {i+1}: {len(client_df)} rows")
        price = client_df['SalePrice']
        print(f"  SalePrice: ${price.mean():,.0f} ± ${price.std():,.0f}")
        print(f"  Date range: {client_df['SaleDate'].min().date()} to {client_df['SaleDate'].max().date()}")
    
    return client_dfs

def save_client_data(client_dfs: list, output_dir: str):
    """Save each client's data to CSV files."""
    for i, df in enumerate(client_dfs, 1):
        file_path = os.path.join(output_dir, f"data{i}.csv")
        df.to_csv(file_path, index=False)
        print(f"  Saved: data{i}.csv ({len(df)} rows)")

def main():
    """Main function to split data for LSTM FL."""
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "output", "preprocess_csv", "df_q1.csv")
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: File not found: {data_path}")
        print("Please check the path to df_q1.csv")
        return
    
    # Prepare data
    df = prepare_data_for_lstm(data_path)
    
    # One-hot encode FULL df once
    df_full = pd.get_dummies(df, drop_first=True)

    # Save the list of feature columns (excluding target)
    target_col = "SalePrice"
    feature_cols = [c for c in df_full.columns if c != target_col]

    schema_path = os.path.join(script_dir, "lstm_feature_columns.txt")
    with open(schema_path, "w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")
    
    # Split for clients
    print(f"\n{'='*50}")
    print("SPLITTING DATA FOR 3 CLIENTS")
    print('='*50)
    client_dfs = split_data_to_clients(df, n_clients=3)
    
    # Save files
    print(f"\n{'='*50}")
    print("SAVING CLIENT DATA FILES")
    print('='*50)
    save_client_data(client_dfs, script_dir)
    
    # Verify files
    print(f"\n{'='*50}")
    print("VERIFICATION")
    print('='*50)
    for i in range(1, 4):
        file_path = os.path.join(script_dir, f"data{i}.csv")
        if os.path.exists(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"✓ data{i}.csv ({size_kb:.1f} KB)")
        else:
            print(f"✗ data{i}.csv (missing)")
    
    print(f"\n{'='*50}")
    print("COMPLETE: Data ready for LSTM federated learning!")
    print('='*50)
    print(f"Files saved to: {script_dir}")
    print("Each file has SalePrice as the last column for LSTM training.")

if __name__ == "__main__":
    main()