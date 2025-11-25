import os
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

from dotenv import load_dotenv

from recommender import (
    load_all_models,
    recommend_by_name_hybrid,
    search_tracks_by_name,
    describe_tracks,
)

load_dotenv() 

# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="Hybrid Music Recommender",
    layout="wide",  # <- use full width
)


# ---------- SPOTIFY HELPER (OPTIONAL) ----------

_SPOTIFY_TOKEN: Optional[str] = None


_SPOTIFY_TOKEN: Optional[str] = None


def _get_spotify_token() -> Optional[str]:
    """
    Get (and cache) a Spotify API token using Client Credentials flow.

    Tries these environment variables (in this order):

      SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
      SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET

    If not set or request fails, returns None and app simply won't show covers.
    """
    global _SPOTIFY_TOKEN
    if _SPOTIFY_TOKEN is not None:
        return _SPOTIFY_TOKEN

    # Prefer SPOTIFY_*, fall back to SPOTIPY_*
    client_id = (
        os.getenv("SPOTIFY_CLIENT_ID")
        or os.getenv("SPOTIPY_CLIENT_ID")
    )
    client_secret = (
        os.getenv("SPOTIFY_CLIENT_SECRET")
        or os.getenv("SPOTIPY_CLIENT_SECRET")
    )

    if not client_id or not client_secret:
        # No credentials provided -> silently disable covers
        return None

    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        _SPOTIFY_TOKEN = data.get("access_token")
    except Exception:
        _SPOTIFY_TOKEN = None

    return _SPOTIFY_TOKEN



def fetch_spotify_cover(track_name: str, artist_name: str) -> Optional[str]:
    """
    Fetch a single album cover URL from Spotify for (track_name, artist_name).

    Returns the URL of the largest album image, or None if unavailable / error.
    """
    token = _get_spotify_token()
    if token is None:
        return None

    query = f"track:{track_name} artist:{artist_name}"
    try:
        resp = requests.get(
            "https://api.spotify.com/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        items = data.get("tracks", {}).get("items", [])
        if not items:
            return None
        images = items[0].get("album", {}).get("images", [])
        if not images:
            return None
        # Typically sorted largest -> smallest
        return images[0].get("url")
    except Exception:
        return None


# ---------- MODEL LOADING (CACHED) ----------

@st.cache_resource
def init_models():
    return load_all_models()


models = init_models()


# ---------- DISPLAY HELPERS ----------

DISPLAY_COLS = [
    "track_name",
    "artist_name",
    "album_name",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "hybrid_score",
    "cosine_sim",
    "als_sim",
    "item2vec_sim",
]

DISPLAY_RENAME = {
    "track_name": "Song",
    "artist_name": "Artist",
    "album_name": "Album",
    "danceability": "Danceability",
    "energy": "Energy",
    "valence": "Valence",
    "tempo": "Tempo (BPM)",
    "hybrid_score": "Hybrid score",
    "cosine_sim": "Audio sim",
    "als_sim": "Playlist sim (ALS)",
    "item2vec_sim": "Seq sim (item2vec)",
}


def format_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects & renames columns into a clean, user-facing schema.
    """
    df = df.copy()
    for col in DISPLAY_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[DISPLAY_COLS]
    df = df.rename(columns=DISPLAY_RENAME)
    return df


def style_numeric(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Apply colour gradients to numeric columns (audio + similarity metrics).
    Green = close / high value, red = far / low value.
    """
    numeric_cols = [
        "Danceability",
        "Energy",
        "Valence",
        "Tempo (BPM)",
        "Hybrid score",
        "Audio sim",
        "Playlist sim (ALS)",
        "Seq sim (item2vec)",
    ]
    styler = df.style.format(precision=3)
    # RdYlGn: red -> yellow -> green; low = red, high = green
    styler = styler.background_gradient(subset=numeric_cols, cmap="RdYlGn")
    return styler


def style_seed_numeric(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    For the single seed row: force all numeric columns to the 'greenest' colour.
    """
    numeric_cols = [
        "Danceability",
        "Energy",
        "Valence",
        "Tempo (BPM)",
        "Hybrid score",
        "Audio sim",
        "Playlist sim (ALS)",
        "Seq sim (item2vec)",
    ]

    styler = df.style.format(precision=3)

    def full_green(row):
        styles = []
        for col in row.index:
            if col in numeric_cols:
                styles.append("background-color: #006400; color: white;")  # dark green
            else:
                styles.append("")
        return styles

    styler = styler.apply(full_green, axis=1)
    return styler

# ---------- SIDEBAR / LEFT CONTROLS ----------

left_col, main_col = st.columns([1, 3])

with left_col:
    st.markdown("### ⚙️ Tuning (optional)")

    with st.expander("Similarity weights", expanded=False):
        w_cos = st.slider("Audio (cosine)", 0.0, 1.0, 0.35, 0.05)
        w_als = st.slider("Playlist co-occurrence (ALS)", 0.0, 1.0, 0.30, 0.05)
        w_i2v = st.slider("Sequence (item2vec)", 0.0, 1.0, 0.30, 0.05)
        w_cluster = st.slider("Cluster bonus", 0.0, 0.2, 0.03, 0.01)

    top_k = st.slider("Number of recommendations", 5, 50, 15, 5)


# ---------- MAIN UI ----------

with main_col:
    st.title("🎧 Hybrid Music Recommender")

    query = st.text_input("Search for a song", value="90210")

    seed_row = None
    recs = pd.DataFrame()
    hits_with_labels = None

    if query:
        # 1) Search up to 30 songs in the HYBRID universe
        hits = search_tracks_by_name(query, models=models, max_results=60)

        if hits.empty:
            st.warning("No matching tracks found in the hybrid universe.")
        else:
            # Build dropdown labels: "Song – Artist (Album)"
            hits = hits.reset_index(drop=True)
            option_labels = [
                f"{row['track_name']} – {row['artist_name']} ({row['album_name']})"
                for _, row in hits.iterrows()
            ]

            st.caption("Select the exact track you mean:")
            selected_label = st.selectbox(
                "Search results",
                options=option_labels,
                index=0,
                label_visibility="collapsed",  # hide label text
            )

            selected_idx = option_labels.index(selected_label)
            seed_metadata = hits.iloc[selected_idx]
            seed_tid = seed_metadata["track_id"]

            # Optional: fetch album art from Spotify
            cover_url = fetch_spotify_cover(
                seed_metadata["track_name"], seed_metadata["artist_name"]
            )

            # Compute recommendations when user clicks
            if st.button("Get recommendations"):
                with st.spinner("Computing recommendations..."):
                    seed_row, recs, _ = recommend_by_name_hybrid(
                        query,
                        models=models,
                        candidate_index=selected_idx,
                        w_cos=w_cos,
                        w_als=w_als,
                        w_i2v=w_i2v,
                        w_cluster=w_cluster,
                        top_k=top_k,
                    )

                # --- Seed track panel ---
                st.subheader("Seed track")

                # Build seed info with same schema + similarity = 1.0
                seed_feat = describe_tracks(
                    [seed_tid],
                    models=models,
                    extra_cols=["danceability", "energy", "valence", "tempo"],
                    top_n=1,
                )

                seed_feat = seed_feat.assign(
                    hybrid_score=1.0,
                    cosine_sim=1.0,
                    als_sim=1.0,
                    item2vec_sim=1.0,
                )

                seed_display = format_for_display(seed_feat)

                # Cover art on its own line, slightly smaller
                if cover_url:
                    st.image(cover_url, width=350)  # <- adjust size if you want
                else:
                    st.info("Album cover unavailable (no Spotify credentials or not found).")

                # Seed feature row table under the image
                st.dataframe(
                    style_seed_numeric(seed_display),   # <- new styling function below
                    use_container_width=True,
                )


                # --- Recommendations table ---
                st.subheader("Recommendations")

                if recs.empty:
                    st.info("No recommendations available for this seed track.")
                else:
                    recs_display = format_for_display(recs)
                    st.dataframe(
                        style_numeric(recs_display),
                        use_container_width=True,
                    )
