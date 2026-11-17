"""K-means-based recommendation engine for Swiggy restaurants."""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CLEANED_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
ENCODED_PATH = os.path.join(DATA_DIR, "encoded_data.csv")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

NUMERICAL_COLUMNS = ["rating", "rating_count", "cost"]
CLUSTER_COUNT = 8
_cache: dict = {}


def load_data():
    """Load aligned source and encoded data, then fit and cache the clusters."""
    if _cache:
        return (
            _cache["cleaned"],
            _cache["encoded"],
            _cache["encoder"],
            _cache["scaler"],
            _cache["features"],
            _cache["clusters"],
        )

    for path in (CLEANED_PATH, ENCODED_PATH, ENCODER_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required project data not found at {path}. Run preprocess.py first."
            )

    cleaned = pd.read_csv(CLEANED_PATH)
    encoded = pd.read_csv(ENCODED_PATH)
    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    if len(cleaned) != len(encoded):
        raise ValueError(
            "cleaned_data.csv and encoded_data.csv must have matching row counts."
        )
    if cleaned.empty:
        raise ValueError("No restaurant rows are available for clustering.")
    missing_columns = [column for column in NUMERICAL_COLUMNS if column not in encoded]
    if missing_columns:
        raise ValueError(
            f"encoded_data.csv is missing required columns: {', '.join(missing_columns)}"
        )

    scaler = StandardScaler()
    features = encoded.copy()
    features[NUMERICAL_COLUMNS] = scaler.fit_transform(
        encoded[NUMERICAL_COLUMNS]
    )
    feature_values = features.to_numpy(dtype=float)
    distinct_rows = np.unique(feature_values, axis=0).shape[0]
    cluster_count = min(CLUSTER_COUNT, len(features), distinct_rows)
    clusters = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    ).fit(feature_values)

    _cache.update(
        cleaned=cleaned,
        encoded=encoded,
        encoder=encoder,
        scaler=scaler,
        features=feature_values,
        clusters=clusters,
    )
    return cleaned, encoded, encoder, scaler, feature_values, clusters


def build_user_vector(
    user_input: dict,
    encoder,
    encoded_columns: list,
    scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    """Encode and scale the user's preferences in the same feature space."""
    categorical_input = pd.DataFrame(
        [[user_input["city"], user_input["cuisine"]]],
        columns=["city", "cuisine"],
    )
    cat_encoded = encoder.transform(categorical_input)
    numerical = np.array(
        [[
            user_input.get("rating", 3.5),
            user_input.get("rating_count", 100),
            user_input.get("cost", 300),
        ]],
        dtype=float,
    )
    if scaler is not None:
        numerical = scaler.transform(
            pd.DataFrame(numerical, columns=NUMERICAL_COLUMNS)
        )

    user_vector = np.hstack([numerical, cat_encoded])
    if user_vector.shape[1] != len(encoded_columns):
        raise ValueError(
            "The encoded user preferences do not match encoded_data.csv columns."
        )
    return user_vector


def get_recommendations(user_input: dict, top_n: int = 10) -> pd.DataFrame:
    """Recommend eligible restaurants from the user's assigned K-means cluster."""
    cleaned, encoded, encoder, scaler, features, clusters = load_data()

    city = user_input.get("city", "")
    cuisine = user_input.get("cuisine", "")
    min_rating = user_input.get("rating", 0)
    max_cost = user_input.get("cost", float("inf"))
    min_rating_count = user_input.get("rating_count", 0)

    mask = pd.Series(True, index=cleaned.index)
    if city:
        mask &= cleaned["city"].str.lower() == city.lower()
    if cuisine:
        mask &= cleaned["cuisine"].str.lower().str.contains(cuisine.lower(), na=False)
    mask &= cleaned["rating"] >= min_rating
    mask &= cleaned["cost"] <= max_cost
    mask &= cleaned["rating_count"] >= min_rating_count

    eligible_indices = np.flatnonzero(mask.to_numpy())
    if len(eligible_indices) == 0:
        return pd.DataFrame()

    user_vector = build_user_vector(
        user_input,
        encoder,
        encoded.columns.tolist(),
        scaler,
    )
    target_cluster = clusters.predict(user_vector)[0]
    cluster_indices = eligible_indices[
        clusters.labels_[eligible_indices] == target_cluster
    ]
    candidate_indices = cluster_indices if len(cluster_indices) else eligible_indices

    distances = np.linalg.norm(features[candidate_indices] - user_vector, axis=1)
    results = cleaned.iloc[candidate_indices].copy()
    results["similarity_score"] = 1 / (1 + distances)
    return (
        results.nlargest(top_n, "similarity_score")
        .reset_index(drop=True)
    )
