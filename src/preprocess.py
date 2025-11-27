# src/preprocess.py
"""
Load CSV, do simple cleaning, and write train/test CSVs to output_prefix.

Provides:
    preprocess_csv(local_csv_path: str, output_prefix: str) -> (train_path, test_path)

Behavior:
- Reads CSV into pandas
- Drops rows with all-NA
- Detects target column: 'target' if present else last column
- Uses pd.get_dummies for categorical encoding (simple approach for PoC)
- Splits into train/test (80/20)
- Writes CSVs to <output_prefix>/train.csv and <output_prefix>/test.csv
- Returns the two local file paths
"""

from typing import Tuple
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("preprocess")
logging.basicConfig(level=logging.INFO)


def _detect_target_column(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    return df.columns[-1]


def preprocess_csv(local_csv_path: str, output_prefix: str = "/tmp") -> Tuple[str, str]:
    """
    Preprocess CSV and write train/test CSV files.

    Args:
        local_csv_path: local path to raw CSV file
        output_prefix: directory where train.csv and test.csv will be saved

    Returns:
        (train_csv_local_path, test_csv_local_path)
    """
    logger.info("Reading CSV from %s", local_csv_path)
    df = pd.read_csv(local_csv_path)

    # Basic cleaning
    logger.info("Initial shape: %s", df.shape)
    df = df.dropna(how="all")
    logger.info("After dropping all-NA rows: %s", df.shape)

    # Detect target column
    target_col = _detect_target_column(df)
    logger.info("Detected target column: %s", target_col)

    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Simple one-hot encode categorical features (PoC)
    X = pd.get_dummies(X)

    # Re-attach target
    processed = X.copy()
    processed[target_col] = y.values

    # Train/test split
    train_df, test_df = train_test_split(processed, test_size=0.2, random_state=42, stratify=processed[target_col] if processed[target_col].nunique() > 1 else None)
    os.makedirs(output_prefix, exist_ok=True)
    train_path = os.path.join(output_prefix, "train.csv")
    test_path = os.path.join(output_prefix, "test.csv")

    logger.info("Writing train to %s (shape=%s)", train_path, train_df.shape)
    train_df.to_csv(train_path, index=False)

    logger.info("Writing test to %s (shape=%s)", test_path, test_df.shape)
    test_df.to_csv(test_path, index=False)

    return train_path, test_path