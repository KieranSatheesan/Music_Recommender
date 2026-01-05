from pathlib import Path
from typing import Dict, Set, List, Optional
import json
import pandas as pd


def extract_track_id_from_uri(track_uri: str) -> Optional[str]:
    """
    Convert 'spotify:track:0UaMYEvWZi0ZqiDOoHU3YI' → '0UaMYEvWZi0ZqiDOoHU3YI'.
    If it's already just an ID, or some other string, return as-is.
    """
    if not isinstance(track_uri, str):
        return None
    parts = track_uri.split(":")
    if len(parts) >= 3 and parts[-2] == "track":
        return parts[-1]
    return track_uri


def get_id_set(df: pd.DataFrame, candidates: List[str], name: str) -> Set[str]:
    """
    Extract canonical Spotify track IDs from a dataframe.
    
    Args:
        df: DataFrame to extract IDs from
        candidates: List of possible ID column names to search
        name: Dataset name for logging
        
    Returns:
        Set of track ID strings
    """
    for col in candidates:
        if col in df.columns:
            series = df[col].dropna().astype(str)
            if "uri" in col.lower():
                ids = {extract_track_id_from_uri(x) for x in series}
            else:
                ids = set(series)
            print(f"[{name}] using ID column: '{col}' → {len(ids):,} unique IDs")
            return ids
    print(f"[{name}] WARNING: no ID column found among {candidates}. Returning empty set.")
    return set()


def load_mpd_track_ids(mpd_dir: Path, verbose: bool = True) -> Set[str]:
    """
    Load all unique track IDs from MPD slice files.
    
    Args:
        mpd_dir: Directory containing mpd.slice.*.json files
        verbose: Whether to print progress information
        
    Returns:
        Set of unique track IDs
    """
    try:
        from tqdm.auto import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        
    mpd_files = sorted(mpd_dir.glob("mpd.slice.*.json"))
    if verbose:
        print(f"Found {len(mpd_files)} MPD slice files")
    
    track_ids = set()
    playlist_count = 0
    track_count = 0
    
    iterator = tqdm(mpd_files) if use_tqdm else mpd_files
    
    for path in iterator:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        playlists = data.get("playlists", [])
        playlist_count += len(playlists)
        
        for pl in playlists:
            tracks = pl.get("tracks", [])
            track_count += len(tracks)
            for t in tracks:
                uri = t.get("track_uri")
                tid = extract_track_id_from_uri(uri)
                if tid:
                    track_ids.add(tid)
    
    if verbose:
        print(f"Total playlists: {playlist_count:,}")
        print(f"Total track entries: {track_count:,}")
        print(f"Unique track IDs: {len(track_ids):,}")
    
    return track_ids


def calculate_overlap_stats(name: str, feature_ids: Set[str], 
                           reference_ids: Set[str]) -> Dict[str, float]:
    """
    Calculate overlap statistics between two ID sets.
    
    Returns:
        Dictionary with keys: 'feature_size', 'overlap_count', 
        'pct_reference_covered', 'pct_feature_in_reference'
    """
    n_feat = len(feature_ids)
    n_ref = len(reference_ids)
    inter = feature_ids & reference_ids
    n_inter = len(inter)
    
    return {
        'feature_size': n_feat,
        'overlap_count': n_inter,
        'pct_reference_covered': (n_inter / n_ref * 100) if n_ref > 0 else 0,
        'pct_feature_in_reference': (n_inter / n_feat * 100) if n_feat > 0 else 0,
    }

def build_playlist_dataframe(mpd_files: List[Path], verbose: bool = True) -> pd.DataFrame:
    """
    Build a DataFrame with playlist-level metadata from MPD slice files.
    
    Args:
        mpd_files: List of paths to mpd.slice.*.json files
        verbose: Whether to show progress bar
        
    Returns:
        DataFrame with columns: pid, name, num_tracks, num_albums, 
        num_followers, collaborative, modified_at
    """
    try:
        from tqdm.auto import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False
    
    playlist_rows = []
    iterator = tqdm(mpd_files, desc="Building playlist table") if use_tqdm else mpd_files
    
    for path in iterator:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        playlists = data.get("playlists", [])
        
        for pl in playlists:
            playlist_rows.append({
                "pid": pl.get("pid"),
                "name": pl.get("name"),
                "num_tracks": pl.get("num_tracks"),
                "num_albums": pl.get("num_albums"),
                "num_followers": pl.get("num_followers"),
                "collaborative": pl.get("collaborative"),
                "modified_at": pl.get("modified_at"),
            })
    
    return pd.DataFrame(playlist_rows)


def build_track_dataframe(mpd_files: List[Path], verbose: bool = True) -> pd.DataFrame:
    """
    Build a DataFrame with track-level data from MPD slice files.
    
    Args:
        mpd_files: List of paths to mpd.slice.*.json files
        verbose: Whether to show progress bar
        
    Returns:
        DataFrame with columns: pid, track_uri, artist_uri, album_uri,
        track_name, artist_name, album_name, pos, duration_ms
    """
    try:
        from tqdm.auto import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False
    
    track_rows = []
    iterator = tqdm(mpd_files, desc="Building track table") if use_tqdm else mpd_files
    
    for path in iterator:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        playlists = data.get("playlists", [])
        
        for pl in playlists:
            pid = pl.get("pid")
            tracks = pl.get("tracks", [])
            for t in tracks:
                track_rows.append({
                    "pid": pid,
                    "track_uri": t.get("track_uri"),
                    "artist_uri": t.get("artist_uri"),
                    "album_uri": t.get("album_uri"),
                    "track_name": t.get("track_name"),
                    "artist_name": t.get("artist_name"),
                    "album_name": t.get("album_name"),
                    "pos": t.get("pos"),
                    "duration_ms": t.get("duration_ms"),
                })
    
    return pd.DataFrame(track_rows)


def show_random_playlists(playlists_df: pd.DataFrame, tracks_df: pd.DataFrame, 
                          n: int = 3, seed: int = 42) -> None:
    """
    Display a random sample of playlists with their metadata and first few tracks.
    
    Args:
        playlists_df: DataFrame with playlist metadata
        tracks_df: DataFrame with track-level data
        n: Number of playlists to sample
        seed: Random seed for reproducibility
    """
    import numpy as np
    
    rng = np.random.default_rng(seed)
    sample_pids = rng.choice(playlists_df["pid"].unique(), size=n, replace=False)
    print(f"Random sample of {n} playlists (pids): {sample_pids}\n")
    
    for pid in sample_pids:
        pl_meta = playlists_df[playlists_df["pid"] == pid].iloc[0]
        pl_tracks = tracks_df[tracks_df["pid"] == pid].sort_values("pos")
        
        print(f"PID: {pid}")
        print(f"Name: {pl_meta['name']}")
        print(f"Num tracks: {pl_meta['num_tracks']}")
        print(f"Num followers: {pl_meta['num_followers']}")
        print("First 5 tracks:")
        for _, row in pl_tracks.head(5).iterrows():
            print("  -", row["track_name"], "–", row["artist_name"])
        print("-" * 60)

def compute_dataset_overlap(set_a: Set[str], set_b: Set[str], 
                           name_a: str = "Set A", name_b: str = "Set B") -> pd.DataFrame:
    """
    Compute overlap statistics between two sets of track IDs.
    
    Args:
        set_a: First set of track IDs
        set_b: Second set of track IDs
        name_a: Name for first set
        name_b: Name for second set
        
    Returns:
        DataFrame with overlap statistics
    """
    intersection = set_a & set_b
    n_a = len(set_a)
    n_b = len(set_b)
    n_inter = len(intersection)
    
    summary_rows = [
        {
            "set": f"{name_a} only",
            "count": n_a - n_inter,
            "pct_of_a": (n_a - n_inter) / n_a * 100 if n_a > 0 else 0,
            "pct_of_b": 0.0,
        },
        {
            "set": f"{name_b} only",
            "count": n_b - n_inter,
            "pct_of_a": 0.0,
            "pct_of_b": (n_b - n_inter) / n_b * 100 if n_b > 0 else 0,
        },
        {
            "set": "Intersection (both)",
            "count": n_inter,
            "pct_of_a": n_inter / n_a * 100 if n_a > 0 else 0,
            "pct_of_b": n_inter / n_b * 100 if n_b > 0 else 0,
        },
    ]
    
    return pd.DataFrame(summary_rows)


def plot_overlap_venn(n_set_a: int, n_set_b: int, n_intersection: int,
                     label_a: str = "Set A", label_b: str = "Set B",
                     color_a: str = '#FF6B6B', color_b: str = '#4ECDC4',
                     figsize: tuple = (14, 5), title: str = None) -> None:
    """
    Create a styled Venn diagram showing overlap between two sets.
    
    Args:
        n_set_a: Size of first set
        n_set_b: Size of second set
        n_intersection: Size of intersection
        label_a: Label for first set
        label_b: Label for second set
        color_a: Color for first set
        color_b: Color for second set
        figsize: Figure size tuple
        title: Plot title
    """
    from matplotlib_venn import venn2
    import matplotlib.pyplot as plt
    
    alpha_value = 0.6
    edge_color = 'white'
    edge_width = 2
    title_shift = 0.14
    
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    v = venn2(
        subsets=(
            n_set_a - n_intersection,
            n_set_b - n_intersection,
            n_intersection
        ),
        set_labels=("", ""),
        set_colors=(color_a, color_b),
        alpha=alpha_value,
        ax=ax
    )
    
    ax.set_aspect('equal')
    
    # Customize patches
    for patch in v.patches:
        if patch:
            patch.set_edgecolor(edge_color)
            patch.set_linewidth(edge_width)
    
    # Customize subset labels
    for text in v.subset_labels:
        if text:
            text.set_fontsize(9)
            text.set_fontweight('bold')
            text.set_color('black')
    
    # Add custom labels
    left_circle_x = v.get_label_by_id('10').get_position()[0] if v.get_label_by_id('10') else -0.5
    right_circle_x = v.get_label_by_id('01').get_position()[0] if v.get_label_by_id('01') else 0.5
    
    ax.text(left_circle_x - 0.25, 0, label_a,
            fontsize=13, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color_a, alpha=0.3, edgecolor='none'))
    
    ax.text(right_circle_x + 0.785, 0, label_b,
            fontsize=13, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color_b, alpha=0.3, edgecolor='none'))
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=0, x=0.5 + title_shift)
    
    plt.tight_layout()
    plt.show()


def compute_missing_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Compute missing value statistics for specified columns.
    
    Args:
        df: DataFrame to analyze
        cols: List of column names to check
        
    Returns:
        DataFrame with missing value statistics
    """
    rows = []
    n = len(df)
    
    for col in cols:
        if col not in df.columns:
            continue
        mc = df[col].isna().sum()
        mp = mc / n * 100 if n > 0 else 0
        rows.append({
            "feature": col,
            "n_rows": n,
            "missing_count": mc,
            "missing_pct": round(mp, 3),
        })
    
    return pd.DataFrame(rows)


def comprehensive_feature_coverage(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    Compute comprehensive coverage statistics for all numeric features.
    
    Args:
        df: DataFrame to analyze
        exclude_cols: Columns to exclude from analysis (e.g., ['track_id'])
        
    Returns:
        DataFrame with coverage statistics sorted by missing/zero percentages
    """
    if exclude_cols is None:
        exclude_cols = []
    
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    rows = []
    n_rows = len(df)
    
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / n_rows * 100 if n_rows > 0 else 0
        
        zero_count = (df[col] == 0).sum()
        zero_pct = zero_count / n_rows * 100 if n_rows > 0 else 0
        
        rows.append({
            "feature": col,
            "dtype": str(df[col].dtype),
            "n_rows": n_rows,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 3),
            "zero_count": zero_count,
            "zero_pct": round(zero_pct, 3),
            "valid_nonzero_count": n_rows - missing_count - zero_count,
        })
    
    return (
        pd.DataFrame(rows)
        .sort_values(by=["missing_pct", "zero_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )