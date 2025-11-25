import os
from typing import Optional, List

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from recommender.hybrid import (
    recommend_by_name_hybrid,
    search_tracks_by_name,
)

# -------------------------------------------------------------------------------------------------
# Streamlit + environment setup
# -------------------------------------------------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Hybrid Music Recommender",
    page_icon="🎧",
    layout="wide",
)


# -------------------------------------------------------------------------------------------------
# Spotify helpers (for album art)
# -------------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_spotify_client() -> Optional[spotipy.Spotify]:
    cid = os.getenv("SPOTIPY_CLIENT_ID")
    secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    if not cid or not secret:
        return None
    auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    return spotipy.Spotify(auth_manager=auth_manager)


@st.cache_data(show_spinner=False)
def get_album_image(track_name: str, artist_name: str) -> Optional[str]:
    sp = get_spotify_client()
    if sp is None:
        return None

    query = f"track:{track_name} artist:{artist_name}"
    try:
        res = sp.search(q=query, type="track", limit=1)
    except Exception:
        return None

    items = res.get("tracks", {}).get("items", [])
    if not items:
        return None

    images = items[0].get("album", {}).get("images", [])
    if not images:
        return None

    # Smallest image (last entry)
    return images[-1]["url"]


# -------------------------------------------------------------------------------------------------
# Styling helpers
# -------------------------------------------------------------------------------------------------

METRIC_COLS = [
    "Danceability",
    "Energy",
    "Valence",
    "Tempo (BPM)",
    "Audio sim",
    "Playlist sim (ALS)",
    "Seq sim (item2vec)",
    "Hybrid score",
]


def style_seed_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Seed track: keep numeric columns solid green."""
    styler = df.style.hide_index()

    def green_numeric(val):
        try:
            float(val)
        except (TypeError, ValueError):
            return ""
        # dark green background, white text
        return "background-color: #005f2f; color: white;"

    styler = styler.applymap(green_numeric, subset=METRIC_COLS)
    styler = styler.set_properties(subset=["Song", "Artist", "Album"],
                                   **{"font-weight": "bold"})
    return styler


def style_recs_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Recommendations: red→yellow→green gradient, Hybrid score last & bold."""
    cmap = LinearSegmentedColormap.from_list(
        "score_cmap",
        ["#8b0000", "#f0e442", "#006400"],  # dark red -> yellow -> dark green
    )

    styler = (
        df.style
        .hide_index()
        .background_gradient(cmap=cmap, subset=METRIC_COLS, axis=None)
        .set_properties(subset=["Hybrid score"], **{"font-weight": "bold"})
    )
    return styler


# -------------------------------------------------------------------------------------------------
# Layout: sidebar (tuning) + main content
# -------------------------------------------------------------------------------------------------

TOP_K = 15
MAX_SEARCH_RESULTS = 60

with st.sidebar:
    st.header("Tuning (optional)")
    st.caption("Adjust similarity weights")

    w_cos = st.slider("Audio features (cosine)", 0.0, 1.0, 0.35, 0.05)
    w_als = st.slider("Playlist co-occurrence (ALS)", 0.0, 1.0, 0.30, 0.05)
    w_i2v = st.slider("Sequence / item2vec", 0.0, 1.0, 0.30, 0.05)
    w_cluster = st.slider("Cluster bonus", 0.0, 0.2, 0.03, 0.01)

st.title("🎧 Hybrid Music Recommender")

# Search inputs
query_col, artist_col = st.columns([3, 2])
with query_col:
    query = st.text_input("Search for a song", value="runaway")

with artist_col:
    artist_filter = st.text_input("Filter by artist (optional)", value="")

# Search results (with album art) right under the inputs
hits = pd.DataFrame()
selected_idx = 0

if query.strip():
    hits = search_tracks_by_name(query.strip(), max_results=MAX_SEARCH_RESULTS)

    if artist_filter.strip():
        a = artist_filter.strip().lower()
        hits = hits[hits["artist_name"].str.lower().str.contains(a, na=False)]

    if hits.empty:
        st.warning("No matching tracks found in the hybrid universe.")
    else:
        st.markdown("**Select the exact track you mean:**")

        options = [
            f"{row['track_name']} — {row['artist_name']} ({row['album_name']})"
            for _, row in hits.iterrows()
        ]
        selected_idx = st.selectbox(
            "Matches",
            options=list(range(len(options))),
            format_func=lambda i: options[i],
            index=0,
            key="seed_select",
        )

        # Show compact album art list under the dropdown
        st.markdown("**Search results (with album art)**")
        for i, (_, row) in enumerate(hits.head(6).iterrows()):  # just first few
            img_url = get_album_image(row["track_name"], row["artist_name"])
            cols = st.columns([1, 4])
            with cols[0]:
                if img_url:
                    st.image(img_url, width=120)  # small!
            with cols[1]:
                st.markdown(
                    f"**{row['track_name']}**  \n"
                    f"{row['artist_name']} — *{row['album_name']}*"
                )


# Button to trigger recommendations
run_button = st.button("Get recommendations", type="primary")

if run_button and not hits.empty:
    # Call backend recommender
    seed_row, recs = recommend_by_name_hybrid(
        query=query.strip(),
        candidate_index=int(selected_idx),
        w_cos=w_cos,
        w_als=w_als,
        w_i2v=w_i2v,
        w_cluster=w_cluster,
        top_k=TOP_K,
        max_search_results=MAX_SEARCH_RESULTS,
    )

    if seed_row is None or recs is None or recs.empty:
        st.warning("No recommendations could be generated for this seed.")
    else:
        # -----------------------------------------------------------------------------------------
        # Prepare seed + recs tables
        # -----------------------------------------------------------------------------------------

        # Seed row as small 1-row DataFrame
        seed_df = pd.DataFrame([seed_row]).copy()
        seed_df.rename(
            columns={
                "track_name": "Song",
                "artist_name": "Artist",
                "album_name": "Album",
            },
            inplace=True,
        )

        # For metrics, we manually compute all = 1.0 (perfect similarity to itself)
        # or reuse from first recommendation row if you prefer.
        seed_df["Danceability"] = recs["danceability"].mean()
        seed_df["Energy"] = recs["energy"].mean()
        seed_df["Valence"] = recs["valence"].mean()
        seed_df["Tempo (BPM)"] = recs["tempo"].mean()
        seed_df["Audio sim"] = 1.0
        seed_df["Playlist sim (ALS)"] = 1.0
        seed_df["Seq sim (item2vec)"] = 1.0
        seed_df["Hybrid score"] = 1.0

        seed_columns = [
            "Song",
            "Artist",
            "Album",
            "Danceability",
            "Energy",
            "Valence",
            "Tempo (BPM)",
            "Audio sim",
            "Playlist sim (ALS)",
            "Seq sim (item2vec)",
            "Hybrid score",
        ]
        seed_df = seed_df[seed_columns]

        # Recommendations table
        recs_df = recs.copy()
        recs_df.rename(
            columns={
                "track_name": "Song",
                "artist_name": "Artist",
                "album_name": "Album",
                "danceability": "Danceability",
                "energy": "Energy",
                "valence": "Valence",
                "tempo": "Tempo (BPM)",
                "cosine_sim": "Audio sim",
                "als_sim": "Playlist sim (ALS)",
                "item2vec_sim": "Seq sim (item2vec)",
            },
            inplace=True,
        )

        rec_columns = [
            "Song",
            "Artist",
            "Album",
            "Danceability",
            "Energy",
            "Valence",
            "Tempo (BPM)",
            "Audio sim",
            "Playlist sim (ALS)",
            "Seq sim (item2vec)",
            "Hybrid score",
        ]
        recs_df = recs_df[rec_columns].head(TOP_K)

        # -----------------------------------------------------------------------------------------
        # Display seed (with album art above) + recommendations
        # -----------------------------------------------------------------------------------------

        st.markdown("---")
        st.subheader("Seed track")

        seed_img_url = get_album_image(seed_row["track_name"], seed_row["artist_name"])
        if seed_img_url:
            st.image(seed_img_url, width=160)

        st.dataframe(
            style_seed_table(seed_df),
            use_container_width=True,
        )

        st.subheader("Recommendations")
        st.dataframe(
            style_recs_table(recs_df),
            use_container_width=True,
        )
