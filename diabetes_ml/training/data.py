"""
Data loading and splitting utilities — no heavy ML dependencies.
"""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET = "diabetes"


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def split_data(X, y):
    """70 / 15 / 15 stratified split, random_state=42."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15 / 0.85,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )
    log.info(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
