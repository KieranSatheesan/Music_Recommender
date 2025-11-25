import os
from typing import Optional, List, Dict

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

# ---------- ENV + PAGE CONFIG ----------

load_dotenv()  # load .env so SPOTIPY_* and SPOTIFY_* are available

st.set_page_config(
    page_title="Hybrid Music Recommender",
    layout="wide",  # full width
)

# Fixed number of recommendations
TOP_K = 20

# ---------- SPOTIFY HELPER (OPTIONAL) ----------

_SPOTIFY_TOKEN: Optional[str] = None


def _get_spotify_token() -> Optional[str]:
    """
    Get (and cache) a Spotify API token using Client Credentials flow.

    Supports either:
      - SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
      - SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET

    If not set or request fails, returns None and we just skip covers.
    """
    global _SPOTIFY_TOKEN
    if _SPOTIFY_TOKEN is not None:
        return _SPOTIFY_TOKEN

    client_id = (
        os.getenv("SPOTIFY_CLIENT_ID")
        or os.getenv("SPOTIPY_CLIENT_ID")
    )
    client_secret = (
        os.getenv("SPOTIFY_CLIENT_SECRET")
        or os.getenv("SPOTIPY_CLIENT_SECRET")
    )

    if not client_id or not client_secret:
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


def fetch_covers_for_hits(hits: pd.DataFrame) -> List[Optional[str]]:
    """
    Fetch cover URLs for a list of hits (track_name, artist_name),
    used just to visually show the search results.
    """
    urls: List[Optional[str]] = []
    for _, row in hits.iterrows():
        url = fetch_spotify_cover(row["track_name"], row["artist_name"])
        urls.append(url)
    return urls


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

DISPLAY_RENAME: Dict[str, str] = {
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

# Header colours you requested
HEADER_COLORS: Dict[str, str] = {
    # text columns
    "Song": "#FFF3B0",   # yellow
    "Artist": "#FFF3B0",
    "Album": "#FFF3B0",
    # similarity columns
    "Audio sim": "#D0E7FF",            # light blue
    "Playlist sim (ALS)": "#D0E7FF",
    "Seq sim (item2vec)": "#D0E7FF",
    # hybrid
    "Hybrid score": "#4C6FFF",        # dark blue
    # audio feature columns
    "Danceability": "#C9F7C2",        # green-ish
    "Energy": "#C9F7C2",
    "Valence": "#C9F7C2",
    "Tempo (BPM)": "#C9F7C2",
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


def _add_header_colors(styler: pd.io.formats.style.Styler, cols: List[str]):
    """
    Colour the column headers according to HEADER_COLORS.
    """
    table_styles = []
    for i, col in enumerate(cols):
        bg = HEADER_COLORS.get(col)
        if bg:
            table_styles.append({
                "selector": f"th.col_heading.level0.col{i}",
                "props": [("background-color", bg)],
            })
    return styler.set_table_styles(table_styles, overwrite=False)


def style_numeric(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Apply colour gradients to numeric columns (audio + similarity metrics),
    and header colours as specified.
    Dark green ~ strong similarity, dark red ~ weak similarity.
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

    # RdYlGn: low=red, high=green (what you wanted)
    styler = styler.background_gradient(
        subset=[c for c in numeric_cols if c in df.columns],
        cmap="RdYlGn"
    )

    styler = _add_header_colors(styler, list(df.columns))
    return styler


# ---------- LAYOUT: SIDEBAR / LEFT CONTROLS ----------

left_col, main_col = st.columns([1, 3])

with left_col:
    st.markdown("### ⚙️ Tuning (optional)")

    with st.expander("Similarity weights", expanded=False):
        w_cos = st.slider("Audio (cosine)", 0.0, 1.0, 0.35, 0.05)
        w_als = st.slider("Playlist co-occurrence (ALS)", 0.0, 1.0, 0.30, 0.05)
        w_i2v = st.slider("Sequence (item2vec)", 0.0, 1.0, 0.30, 0.05)
        w_cluster = st.slider("Cluster bonus", 0.0, 0.2, 0.03, 0.01)


# ---------- MAIN UI ----------

with main_col:
    st.title("🎧 Hybrid Music Recommender")

    query = st.text_input("Search for a song", value="90210")
    artist_filter = st.text_input("Filter by artist (optional)", value="")

    seed_row = None
    recs = pd.DataFrame()

    if query:
        # 1) Search up to 60 songs in the HYBRID universe
        hits = search_tracks_by_name(query, models=models, max_results=60)

        # Apply artist filter if present
        if artist_filter.strip():
            af = artist_filter.lower()
            hits = hits[
                hits["artist_name"].str.lower().str.contains(af, na=False)
            ]

        if hits.empty:
            st.warning("No matching tracks found in the hybrid universe.")
        else:
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
                label_visibility="collapsed",
            )
            selected_idx = option_labels.index(selected_label)

            # --- Show search hits with album covers (visual confirmation) ---
            st.markdown("##### Search results (with album art)")

            # Fetch covers for visible hits (can be up to 60)
            cover_urls = fetch_covers_for_hits(hits)

            # Show in a scrollable container-like grid
            for i, row in hits.iterrows():
                with st.container():
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        url = cover_urls[i]
                        if url:
                            st.image(url, use_column_width=True)
                    with c2:
                        # highlight the currently selected one
                        prefix = "✅ " if i == selected_idx else ""
                        st.markdown(
                            f"**{prefix}{row['track_name']}**  \n"
                            f"{row['artist_name']} — {row['album_name']}"
                        )
                st.markdown("---")

            # --- Compute recommendations button ---
            if st.button("Get recommendations"):
                seed_metadata = hits.iloc[selected_idx]
                seed_tid = seed_metadata["track_id"]

                with st.spinner("Computing recommendations..."):
                    # Wrapper uses track_name search & candidate_index internally
                    seed_row, recs, _ = recommend_by_name_hybrid(
                        query,
                        models=models,
                        candidate_index=selected_idx,
                        w_cos=w_cos,
                        w_als=w_als,
                        w_i2v=w_i2v,
                        w_cluster=w_cluster,
                        top_k=TOP_K,
                    )

                # --- Seed track panel ---
                st.subheader("Seed track")

                # Album art above the row
                seed_cover_url = fetch_spotify_cover(
                    seed_metadata["track_name"], seed_metadata["artist_name"]
                )
                if seed_cover_url:
                    st.image(seed_cover_url, use_column_width=True)
                else:
                    st.info("Album cover unavailable (Spotify not configured or track not found).")

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

                st.dataframe(
                    style_numeric(seed_display),
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
