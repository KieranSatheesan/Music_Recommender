# app.py (from project root)
import streamlit as st
from recommender import load_all_models, recommend_by_name_hybrid, search_tracks_by_name

@st.cache_resource
def init_models():
    return load_all_models()

models = init_models()

st.title("🎧 Hybrid Music Recommender")

query = st.text_input("Search for a song", value="90210")

if query:
    hits = search_tracks_by_name(query, models=models)
    if hits.empty:
        st.warning("No matching tracks found.")
    else:
        st.write("Search results:")
        st.dataframe(hits.reset_index(drop=True))

        idx = st.number_input(
            "Choose row index as seed (0-based)",
            min_value=0,
            max_value=len(hits) - 1,
            value=0,
            step=1,
        )

        w_cos = st.slider("Audio (cosine)", 0.0, 1.0, 0.35, 0.05)
        w_als = st.slider("ALS", 0.0, 1.0, 0.30, 0.05)
        w_i2v = st.slider("item2vec", 0.0, 1.0, 0.30, 0.05)
        w_cluster = st.slider("Cluster bonus", 0.0, 0.2, 0.03, 0.01)
        top_k = st.slider("Number of recommendations", 5, 50, 15, 5)

        if st.button("Get recommendations"):
            with st.spinner("Computing recommendations..."):
                seed_row, recs, _ = recommend_by_name_hybrid(
                    query,
                    models=models,
                    candidate_index=int(idx),
                    w_cos=w_cos,
                    w_als=w_als,
                    w_i2v=w_i2v,
                    w_cluster=w_cluster,
                    top_k=top_k,
                )

            st.subheader("Seed track")
            st.write(seed_row)

            st.subheader("Recommendations")
            st.dataframe(
                recs[
                    [
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
                ].round(3)
            )
