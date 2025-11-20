import json
from pathlib import Path
import pandas as pd


def extract_track_id(uri: str) -> str:
    """Convert 'spotify:track:ABC123' → 'ABC123'."""
    if not isinstance(uri, str):
        return None
    parts = uri.split(":")
    return parts[-1] if len(parts) >= 3 else None


def build_track_metadata():
    print("=== Building track_metadata.csv from MPD ===")

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MPD_DIR = PROJECT_ROOT / "data" / "raw" / "mpd" / "data"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "track_metadata.csv"

    if not MPD_DIR.exists():
        raise FileNotFoundError(f"MPD directory not found: {MPD_DIR}")

    mpd_files = sorted(MPD_DIR.glob("mpd.slice.*.json"))
    print(f"Found {len(mpd_files)} MPD slice files")

    rows = []

    for fpath in mpd_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for playlist in data.get("playlists", []):
            for t in playlist.get("tracks", []):
                track_id = extract_track_id(t.get("track_uri"))

                rows.append({
                    "track_id": track_id,
                    "track_name": t.get("track_name"),
                    "artist_name": t.get("artist_name"),
                    "album_name": t.get("album_name"),
                    "artist_uri": t.get("artist_uri"),
                    "album_uri": t.get("album_uri"),
                    "duration_ms": t.get("duration_ms"),
                })

    print(f"Collected {len(rows):,} raw rows")

    df = pd.DataFrame(rows)

    # Drop rows with no track_id
    df = df.dropna(subset=["track_id"])

    # Deduplicate by track_id (keep the first non-null metadata)
    df = (
        df.groupby("track_id")
        .agg({
            "track_name": "first",
            "artist_name": "first",
            "album_name": "first",
            "artist_uri": "first",
            "album_uri": "first",
            "duration_ms": "first",
        })
        .reset_index()
    )

    print(f"Final unique tracks: {len(df):,}")
    print(f"Saving to: {OUTPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=== Done! ===")


if __name__ == "__main__":
    build_track_metadata()
