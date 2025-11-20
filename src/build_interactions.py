import json
from pathlib import Path

import pandas as pd


def extract_track_id(uri: str) -> str | None:
    """Convert 'spotify:track:ABC123' → 'ABC123'."""
    if not isinstance(uri, str):
        return None
    parts = uri.split(":")
    return parts[-1] if len(parts) >= 3 else None


def build_interactions() -> None:
    print("=== Building interactions.parquet and playlists.parquet from MPD ===")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MPD_DIR = PROJECT_ROOT / "data" / "raw" / "mpd"
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    INTERACTIONS_PATH = PROCESSED_DIR / "interactions.parquet"
    PLAYLISTS_PATH = PROCESSED_DIR / "playlists.parquet"

    if not MPD_DIR.exists():
        raise FileNotFoundError(f"MPD directory not found: {MPD_DIR}")

    mpd_files = sorted(MPD_DIR.rglob("mpd.slice.*.json"))
    print(f"Found {len(mpd_files)} MPD slice files")

    playlist_rows = []
    interaction_rows = []

    for fpath in mpd_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for pl in data.get("playlists", []):
            pid = pl.get("pid")

            # Playlist-level info
            playlist_rows.append({
                "pid": pid,
                "name": pl.get("name"),
                "num_tracks": pl.get("num_tracks"),
                "num_albums": pl.get("num_albums"),
                "num_followers": pl.get("num_followers"),
                "collaborative": pl.get("collaborative"),
                "modified_at": pl.get("modified_at"),
            })

            # Track-level interactions
            for t in pl.get("tracks", []):
                track_id = extract_track_id(t.get("track_uri"))
                interaction_rows.append({
                    "pid": pid,
                    "track_id": track_id,
                    "pos": t.get("pos"),
                    "duration_ms": t.get("duration_ms"),
                })

    print(f"Collected {len(playlist_rows):,} playlist rows")
    print(f"Collected {len(interaction_rows):,} interaction rows")

    playlists_df = pd.DataFrame(playlist_rows).drop_duplicates(subset=["pid"])
    interactions_df = pd.DataFrame(interaction_rows)

    # Drop rows without track_id (very rare/bad data)
    interactions_df = interactions_df.dropna(subset=["track_id"])

    print(f"\nUnique playlists: {playlists_df['pid'].nunique():,}")
    print(f"Rows in interactions_df: {len(interactions_df):,}")
    print(f"Unique tracks in interactions_df: {interactions_df['track_id'].nunique():,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting interactions to: {INTERACTIONS_PATH}")
    interactions_df.to_parquet(INTERACTIONS_PATH, index=False)

    print(f"Writing playlists to:   {PLAYLISTS_PATH}")
    playlists_df.to_parquet(PLAYLISTS_PATH, index=False)

    print("\n=== Done! interactions.parquet and playlists.parquet written. ===")


if __name__ == "__main__":
    build_interactions()
