# client_utils.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Any
import os

def load_dataset(filename: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load this client's local CSV (target in first column)."""

    # Resolve path relative to this file so it works from any CWD
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, filename)

    df = pd.read_csv(csv_path)

    # Target = first column, Features = others
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    # One‑hot encode categorical feature columns
    X = pd.get_dummies(X, drop_first=True)

    # Random 80/20 train/test split for this client
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    return X_train, y_train, X_test, y_test


def get_params(model: RandomForestRegressor) -> List[Any]:
    """Return a simple list of hyperparameters (demo only, not real RF weights)."""
    return [
        model.n_estimators,
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf,
        model.max_features,
        model.n_jobs,
    ]


def set_params(model: RandomForestRegressor, params: List[Any]) -> RandomForestRegressor:
    """Set hyperparameters from the received list (demo only)."""
    model.n_estimators = int(params[0])
    model.max_depth = int(params[1]) if params[1] is not None else None
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    model.max_features = float(params[4])
    model.n_jobs = int(params[5])
    if n_jobs == 0:  
        n_jobs = -1
    model.n_jobs = n_jobs    
    return model