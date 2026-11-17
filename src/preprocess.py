"""
preprocess.py
Data cleaning and One-Hot encoding for Swiggy restaurant dataset.
Inputs:  data/swiggy.csv
Outputs: data/cleaned_data.csv, data/encoded_data.csv, models/encoder.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

RAW_PATH = os.path.join(DATA_DIR, "swiggy.csv")
CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
ENCODED_PATH = os.path.join(DATA_DIR, "encoded_data.csv")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load raw CSV."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe."""
    df = df.copy()

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Replace '--' ratings with NaN
    df["rating"] = df["rating"].replace("--", np.nan)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Parse rating_count: "500+ ratings" -> 500
    if df["rating_count"].dtype == object:
        df["rating_count"] = (
            df["rating_count"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", np.nan)
        )
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

    # Remove ₹ symbol and commas from cost
    if df["cost"].dtype == object:
        df["cost"] = (
            df["cost"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    # Drop rows with missing critical fields
    df.dropna(subset=["name", "city", "cuisine", "rating", "cost"], inplace=True)

    # Fill rating_count with median
    median_rc = df["rating_count"].median()
    df["rating_count"] = df["rating_count"].fillna(median_rc)

    df.reset_index(drop=True, inplace=True)
    return df


def encode(df: pd.DataFrame):
    """One-Hot encode categorical features and return (encoded_df, encoder)."""
    categorical_cols = ["city", "cuisine"]
    numerical_cols = ["rating", "rating_count", "cost"]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = encoder.fit_transform(df[categorical_cols])
    cat_col_names = encoder.get_feature_names_out(categorical_cols).tolist()

    numerical_data = df[numerical_cols].values
    full_data = np.hstack([numerical_data, cat_encoded])
    full_cols = numerical_cols + cat_col_names

    encoded_df = pd.DataFrame(full_data, columns=full_cols)
    return encoded_df, encoder


def preprocess(raw_path: str = RAW_PATH) -> None:
    """Full preprocessing pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading data from {raw_path} ...")
    df_raw = load_raw(raw_path)
    print(f"  Raw rows: {len(df_raw)}")

    df_clean = clean(df_raw)
    print(f"  Cleaned rows: {len(df_clean)}")

    df_encoded, encoder = encode(df_clean)
    print(f"  Encoded columns: {len(df_encoded.columns)}")

    df_clean.to_csv(CLEANED_PATH, index=False)
    df_encoded.to_csv(ENCODED_PATH, index=False)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    print(f"  Saved: {CLEANED_PATH}")
    print(f"  Saved: {ENCODED_PATH}")
    print(f"  Saved: {ENCODER_PATH}")
    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess()
