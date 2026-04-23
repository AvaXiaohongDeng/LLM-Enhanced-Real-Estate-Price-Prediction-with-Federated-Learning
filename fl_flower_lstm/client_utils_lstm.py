# client_utils_lstm.py
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

def load_series_dataset(
    filename: str,
    seq_len: int = 10,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple sliding-window time-series dataset from CSV.

    Assumes SalePrice is the target; we will move it to column 0,
    and all other columns follow a fixed global schema.
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, filename)
    df = pd.read_csv(csv_path)

    target_col = "SalePrice"

    # Load global feature schema
    schema_path = os.path.join(base_dir, "lstm_feature_columns.txt")
    with open(schema_path, "r", encoding="utf-8") as f:
        feature_cols = [line.strip() for line in f.readlines()]

    # One‑hot encode all non‑numeric columns
    df_enc = pd.get_dummies(df, drop_first=True)

    # Separate target and features
    y = df_enc[target_col].astype("float32")
    X = df_enc.drop(columns=[target_col])

    # Reindex features to global schema (missing columns -> 0)
    X = X.reindex(columns=feature_cols, fill_value=0.0)

    # Ensure target is the first column as expected by the rest of the code
    values = np.column_stack([y.values, X.values]).astype("float32")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for i in range(len(values) - seq_len):
        window = values[i : i + seq_len]
        target = values[i + seq_len, 0]
        X_list.append(window)
        y_list.append(target)

    X = np.stack(X_list)
    y = np.array(y_list)

    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test


def create_lstm_model(input_shape) -> keras.Model:
    """Very small LSTM regression model for demo."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),  # price
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def get_model_parameters(model: keras.Model):
    """Convert Keras weights to list of NumPy arrays for Flower."""
    return model.get_weights()


def set_model_parameters(model: keras.Model, params) -> None:
    model.set_weights(params)
