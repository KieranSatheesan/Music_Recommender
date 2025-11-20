#!/usr/bin/env python
"""
Build a combined track-level feature dataset from multiple raw sources.

- Reads from: data/raw/
- Writes to:  data/processed/combined_features.parquet

Included sources:
    - tracks_features.csv
    - spotify_data.csv
    - spotify_features_data_2023.csv
    - tracks.csv
    - data.csv
    - raw songs dataset.csv
    - SpotifyAudioFeaturesApril2019.csv
    - Music Info.csv
    - songs_with_audio_feature.csv
    - 2mil_dataset.json

Excluded:
    - 278k_labelled_uri.csv
    - Music.csv
    - MPD playlist JSONs
"""

from pathlib import Path
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

# Core audio feature columns we care about
BASE_FEATURE_COLS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Some common extras that are useful if available
EXTRA_FEATURE_COLS = [
    "duration_ms",
    "year",
    "popularity",
    "explicit",
    "time_signature",
]

# Dataset configs: path (relative to repo root) + candidate ID columns
DATASET_CONFIGS = [
    {
        "name": "tracks_features.csv",
        "path": Path("data/raw/tracks_features.csv"),
        "id_candidates": ["id", "track_id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "spotify_data.csv",
        "path": Path("data/raw/spotify_data.csv"),
        "id_candidates": ["track_id", "id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "spotify_features_data_2023.csv",
        "path": Path("data/raw/spotify_features_data_2023.csv"),
        "id_candidates": ["id", "track_id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "tracks.csv",
        "path": Path("data/raw/tracks.csv"),
        "id_candidates": ["id", "track_id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "data.csv",
        "path": Path("data/raw/data.csv"),
        "id_candidates": ["id", "track_id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "raw songs dataset.csv",
        "path": Path("data/raw/raw songs dataset.csv"),
        "id_candidates": ["id", "track_id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "SpotifyAudioFeaturesApril2019.csv",
        "path": Path("data/raw/SpotifyAudioFeaturesApril2019.csv"),
        "id_candidates": ["id", "track_id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "Music Info.csv",
        "path": Path("data/raw/Music Info.csv"),
        "id_candidates": ["spotify_id", "track_id", "id", "uri", "track_uri"],
    },
    {
        "name": "songs_with_audio_feature.csv",
        "path": Path("data/raw/songs_with_audio_feature.csv"),
        "id_candidates": ["track_id", "id", "spotify_id", "uri", "track_uri"],
    },
    {
        "name": "2mil_dataset.json",
        "path": Path("data/raw/2mil_dataset.json"),
        "id_candidates": ["track_uri", "track_id", "id", "spotify_id", "uri"],
    },
]


# -----------------------------
# Helper functions
# -----------------------------

def extract_track_id_from_uri(track_uri: str) -> str | None:
    """
    Convert 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI' → '0UaMYEvWZi0ZqiDOoHU3YI'.
    If it's already just an ID or some other string, return as-is.
    """
    if not isinstance(track_uri, str):
        return None
    parts = track_uri.split(":")
    if len(parts) >= 3 and parts[-2] == "track":
        return parts[-1]
    return track_uri


def pick_id_column(df: pd.DataFrame, candidates: list[str], dataset_name: str) -> str:
    """
    Return the first matching column name in `candidates` that exists in df.
    Raise a clear error if none are found.
    """
    for col in candidates:
        if col in df.columns:
            print(f"[{dataset_name}] Using ID column: '{col}'")
            return col
    raise ValueError(
        f"No ID column found in {dataset_name}. "
        f"Tried candidates: {candidates}, available: {list(df.columns)}"
    )


def load_one_dataset(config: dict) -> pd.DataFrame:
    """
    Load one dataset, normalise to a canonical 'track_id', and keep only
    track_id + relevant feature columns.
    """
    name = config["name"]
    path = config["path"]
    id_candidates = config["id_candidates"]

    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")

    print(f"\nLoading {name} from {path} ...")

    # Choose loader based on suffix
    if path.suffix.lower() in [".csv"]:
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".json"]:
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type for {name}: {path.suffix}")

    print(f"  Shape: {df.shape}")

    # Find appropriate ID column
    id_col = pick_id_column(df, id_candidates, name)

    # Build canonical 'track_id'
    id_series = df[id_col].dropna().astype(str)

    if "uri" in id_col.lower():
        track_ids = id_series.apply(extract_track_id_from_uri)
    else:
        track_ids = id_series

    df = df.loc[track_ids.index].copy()
    df["track_id"] = track_ids

    # Columns to keep: track_id + any of the base + extra feature columns that exist
    wanted_cols = ["track_id"]
    for col in BASE_FEATURE_COLS + EXTRA_FEATURE_COLS:
        if col in df.columns and col not in wanted_cols:
            wanted_cols.append(col)

    df_out = df[wanted_cols].dropna(subset=["track_id"]).drop_duplicates(subset=["track_id"])

    print(
        f"  → kept {len(df_out):,} rows with unique track_id, "
        f"{len(wanted_cols) - 1} feature columns."
    )
    return df_out


def combine_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate all per-dataset DataFrames and aggregate by track_id
    (taking the first non-null value for each feature).
    """
    print("\nConcatenating all datasets...")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined rows before dedupe: {len(combined):,}")

    # Group by track_id and aggregate first non-null per column
    grouped = (
        combined
        .sort_values("track_id")
        .groupby("track_id", as_index=False)
        .first()
    )

    print(f"  Unique track_ids after grouping: {len(grouped):,}")
    return grouped


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_features.parquet"

    dfs = []
    for cfg in DATASET_CONFIGS:
        df = load_one_dataset(cfg)
        dfs.append(df)

    master = combine_datasets(dfs)

    print(f"\nSaving combined features to: {output_path}")
    try:
        master.to_parquet(output_path, index=False)
        print("Done. (Parquet written successfully.)")
    except Exception as e:
        fallback_csv = output_dir / "combined_features.csv"
        print(f"Parquet write failed ({e!r}). Falling back to CSV: {fallback_csv}")
        master.to_csv(fallback_csv, index=False)
        print("Done. (CSV written successfully.)")


if __name__ == "__main__":
    main()
