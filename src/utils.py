"""
NEW FILE: utils.py (Robust path version)
Utility functions to load unique values from cleaned data for Streamlit UI.
Resolves data directory robustly regardless of current working directory.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")


def _load_cleaned() -> pd.DataFrame:
    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(
            f"cleaned_data.csv not found at {CLEANED_PATH}. Run python src/preprocess.py first."
        )
    return pd.read_csv(CLEANED_PATH)


def get_unique_cities() -> list:
    """Return sorted list of unique cities from cleaned data."""
    df = _load_cleaned()
    return sorted(df["city"].dropna().unique().tolist())


def get_unique_cuisines() -> list:
    """Return sorted list of unique cuisines from cleaned data."""
    df = _load_cleaned()
    return sorted(df["cuisine"].dropna().unique().tolist())
