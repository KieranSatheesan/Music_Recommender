"""
Utility functions for loading and preparing track-level feature data
for content-based models (cosine similarity, kNN, clustering, etc.).

This module expects a processed feature file at:
    data/processed/combined_features.csv

with columns including:
    track_id, danceability, energy, loudness, speechiness,
    acousticness, instrumentalness, liveness, valence, tempo,
    key, mode, duration_ms, time_signature, ...

Core entry point:
    load_feature_matrix(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Paths / configuration
# -------------------------------------------------------------------

# PROJECT_ROOT: .../Music_Recommender
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
COMBINED_FEATURES_PATH = DATA_PROCESSED / "combined_features.csv"

# Core audio features to use for modelling / similarity
CORE_FEATURE_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
    "key",
    "mode",
]


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _check_features_path(path: Path = COMBINED_FEATURES_PATH) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"combined_features.csv not found at {path}. "
            f"Make sure you have run the feature-building script and that "
            f"data/processed/combined_features.csv exists."
        )


def _impute_small_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute small amounts of missing data for selected columns.

    Currently:
      - duration_ms: fill NaN with median
      - time_signature: fill NaN with median (likely 4.0)

    Returns a *new* DataFrame (does not modify in-place).
    """
    df = df.copy()

    if "duration_ms" in df.columns:
        median_duration = df["duration_ms"].median()
        df["duration_ms"] = df["duration_ms"].fillna(median_duration)

    if "time_signature" in df.columns:
        median_ts = df["time_signature"].median()
        df["time_signature"] = df["time_signature"].fillna(median_ts)

    return df


def _standardize_and_normalize(X: np.ndarray) -> np.ndarray:
    """
    Standardize features (per column) to mean 0, std 1,
    then L2-normalize each row (track vector).

    Args:
        X: (n_samples, n_features) float array.

    Returns:
        X_norm: same shape as X, standardized + L2-normalized.
    """
    # Standardize columns
    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0  # avoid division by zero

    X_std = (X - means) / stds

    # L2-normalize rows
    norms = np.linalg.norm(X_std, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X_std / norms

    return X_norm


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def load_combined_features() -> pd.DataFrame:
    """
    Load the full combined_features.csv as a DataFrame.

    Returns:
        df: DataFrame with at least 'track_id' and CORE_FEATURE_COLS present.
    """
    _check_features_path(COMBINED_FEATURES_PATH)
    df = pd.read_csv(COMBINED_FEATURES_PATH)

    # Ensure track_id is string
    if "track_id" not in df.columns:
        raise KeyError(
            "Expected column 'track_id' in combined_features.csv but it was not found."
        )

    df["track_id"] = df["track_id"].astype(str)
    return df


def load_feature_matrix(
    allowed_track_ids: Optional[Iterable[str]] = None,
    core_features: Optional[Iterable[str]] = None,
    standardize: bool = True,
    l2_normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], pd.DataFrame]:
    """
    Build a clean feature matrix for content-based models.

    Steps:
      1. Load combined_features.csv
      2. (Optional) filter to allowed_track_ids
      3. Impute small missing values for duration_ms, time_signature
      4. Drop rows with missing values in core feature columns
      5. Extract feature matrix X for core features
      6. Standardize (per feature) and L2-normalize (per track), if requested
      7. Build track_id -> index mapping

    Args:
        allowed_track_ids:
            Optional iterable of track_ids to keep (e.g. intersection with MPD).
            If None, all tracks in combined_features.csv are used.
        core_features:
            Optional iterable of feature column names to use. If None, uses CORE_FEATURE_COLS.
        standardize:
            If True, standardize each feature dimension to mean 0, std 1.
        l2_normalize:
            If True, L2-normalize each row (track vector).

    Returns:
        X: (n_tracks, n_features) numpy array of float32
        track_ids: (n_tracks,) numpy array of track_id strings
        track_id_to_idx: dict mapping track_id -> row index in X
        feat_df: filtered DataFrame containing 'track_id' + feature columns used
    """
    df = load_combined_features()

    # Optional filter to a set of allowed track_ids
    if allowed_track_ids is not None:
        allowed_set = {str(tid) for tid in allowed_track_ids}
        df = df[df["track_id"].isin(allowed_set)].copy()

    # Choose feature columns
    feature_cols = list(core_features) if core_features is not None else list(CORE_FEATURE_COLS)

    # Ensure all required columns exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"The following required feature columns are missing from combined_features.csv: {missing_cols}"
        )

    # Impute small missing values for certain columns
    df = _impute_small_missing_values(df)

    # Drop any rows that still have NaNs in the selected feature columns
    before_rows = len(df)
    df = df.dropna(subset=feature_cols).copy()
    after_rows = len(df)

    if after_rows < before_rows:
        print(
            f"[load_feature_matrix] Dropped {before_rows - after_rows} rows due to NaNs "
            f"in core feature columns."
        )

    # Extract track_ids and feature matrix
    track_ids = df["track_id"].astype(str).values
    X = df[feature_cols].astype(np.float32).values

    # Standardize and/or normalize if requested
    if standardize or l2_normalize:
        # Standardize then normalize; if only one is requested, we handle accordingly
        # by turning off the other operation logically.
        means = X.mean(axis=0, keepdims=True)
        stds = X.std(axis=0, keepdims=True)
        stds[stds == 0] = 1.0

        if standardize:
            X = (X - means) / stds

        if l2_normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms

    # Build index mapping
    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    return X, track_ids, track_id_to_idx, df[["track_id"] + feature_cols].copy()
