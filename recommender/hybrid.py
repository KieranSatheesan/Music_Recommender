# recommender/hybrid.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import RecommenderModels


# Hyperparams for candidate generation
COS_TOPN = 1000
ALS_TOPN = 1000
I2V_TOPN = 1000
HYBRID_CANDIDATE_POOL = 2000


# --- Basic helpers ---


def describe_tracks(
    track_ids,
    models: RecommenderModels,
    extra_cols=None,
    top_n=None,
) -> pd.DataFrame:
    track_ids = list(track_ids)

    meta_df_u = models.meta_df_u
    feat_df_u = models.feat_df_u

    meta_sub = (
        meta_df_u[["track_id", "track_name", "artist_name", "album_name"]]
        .drop_duplicates()
        .set_index("track_id")
        .loc[lambda df: df.index.intersection(track_ids)]
        .reset_index()
    )

    if extra_cols:
        feat_idx = feat_df_u.set_index("track_id")
        common_ids = feat_idx.index.intersection(track_ids)
        feat_sub = feat_idx.loc[common_ids, extra_cols].reset_index()
        meta_sub = meta_sub.merge(feat_sub, on="track_id", how="left")

    if top_n is not None:
        meta_sub = meta_sub.head(top_n)

    return meta_sub


def search_tracks_by_name(
    query: str,
    models: RecommenderModels,
    max_results: int = 10,
) -> pd.DataFrame:
    meta_df_u = models.meta_df_u
    q = query.lower()
    hits = (
        meta_df_u.loc[
            meta_df_u["track_name"].str.lower().str.contains(q, na=False),
            ["track_id", "track_name", "artist_name", "album_name"],
        ]
        .drop_duplicates()
        .head(max_results)
    )
    return hits


def get_cluster_id(track_id: str, models: RecommenderModels):
    try:
        return int(models.track_cluster_df.loc[track_id, "cluster_id"])
    except KeyError:
        return None


# --- Single-model scorers ---


def cosine_scores_for_track(
    track_id: str,
    models: RecommenderModels,
    top_n: int = COS_TOPN,
) -> pd.DataFrame:
    if track_id not in models.tid_to_feat_idx_u:
        return pd.DataFrame(columns=["track_id", "cosine_sim"])

    idx = models.tid_to_feat_idx_u[track_id]
    vec = models.X_norm_u[idx : idx + 1]

    distances, indices = models.knn_cosine_u.kneighbors(vec, n_neighbors=top_n + 1)
    distances = distances[0]
    indices = indices[0]

    sims = 1.0 - distances
    neigh_tids = models.feat_df_u["track_id"].values[indices]

    df = pd.DataFrame(
        {
            "track_id": neigh_tids,
            "cosine_sim": sims,
        }
    )
    df = df[df["track_id"] != track_id].reset_index(drop=True)
    return df


def als_scores_for_track(
    track_id: str,
    models: RecommenderModels,
    top_n: int = ALS_TOPN,
) -> pd.DataFrame:
    als_model = models.als_model
    tid_to_item_idx_als = models.tid_to_item_idx_als
    idx_to_tid_als = models.idx_to_tid_als

    if track_id not in tid_to_item_idx_als:
        return pd.DataFrame(columns=["track_id", "als_sim"])

    item_idx = tid_to_item_idx_als[track_id]
    n_items_als = als_model.item_factors.shape[0]
    if not (0 <= item_idx < n_items_als):
        return pd.DataFrame(columns=["track_id", "als_sim"])

    sim_items, sim_scores = als_model.similar_items(
        itemid=item_idx,
        N=top_n + 1,
    )

    neigh_tids = idx_to_tid_als[sim_items]

    df = pd.DataFrame(
        {
            "track_id": neigh_tids,
            "als_sim": sim_scores,
        }
    )
    df = df[df["track_id"] != track_id].reset_index(drop=True)
    return df.head(top_n)


def item2vec_scores_for_track(
    track_id: str,
    models: RecommenderModels,
    top_n: int = I2V_TOPN,
) -> pd.DataFrame:
    kv = models.item2vec_kv
    if track_id not in kv:
        return pd.DataFrame(columns=["track_id", "item2vec_sim"])

    most_sim = kv.most_similar(track_id, topn=top_n + 1)

    neigh_tids = []
    neigh_scores = []
    for tid, score in most_sim:
        if tid == track_id:
            continue
        neigh_tids.append(tid)
        neigh_scores.append(score)
        if len(neigh_tids) >= top_n:
            break

    df = pd.DataFrame(
        {
            "track_id": neigh_tids,
            "item2vec_sim": neigh_scores,
        }
    )
    return df


# --- Hybrid scoring ---


def hybrid_scores_for_track(
    track_id: str,
    models: RecommenderModels,
    w_cos: float = 0.3,
    w_als: float = 0.3,
    w_i2v: float = 0.3,
    w_cluster: float = 0.1,
    top_k: int = 30,
    candidate_pool: int = HYBRID_CANDIDATE_POOL,
    debug_coverage: bool = False,
) -> pd.DataFrame:
    """
    Compute hybrid scores for neighbours of a single seed track.

    Returns a dataframe with:
      track_id, cosine_sim, als_sim, item2vec_sim, same_cluster, hybrid_score
    """
    # 1) Candidates from each model
    df_cos = cosine_scores_for_track(track_id, models=models, top_n=candidate_pool)
    df_als = als_scores_for_track(track_id, models=models, top_n=candidate_pool)
    df_i2v = item2vec_scores_for_track(track_id, models=models, top_n=candidate_pool)

    # 2) Union
    df = df_cos.merge(df_als, on="track_id", how="outer")
    df = df.merge(df_i2v, on="track_id", how="outer")

    if df.empty:
        return df

    # Ensure numeric + fill NaNs as 0 ("no signal")
    for col in ["cosine_sim", "als_sim", "item2vec_sim"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
        df[col] = df[col].fillna(0.0)

    # Debug coverage
    if debug_coverage:
        n_cand = len(df)
        cov_cos = (df["cosine_sim"] > 0).mean() * 100
        cov_als = (df["als_sim"] > 0).mean() * 100
        cov_i2v = (df["item2vec_sim"] > 0).mean() * 100
        print(
            f"[COVERAGE] seed={track_id}, candidates={n_cand}, "
            f"cosine={cov_cos:.1f}%, ALS={cov_als:.1f}%, item2vec={cov_i2v:.1f}%"
        )

    # 3) Same-cluster bonus
    seed_cluster = get_cluster_id(track_id, models=models)
    df["same_cluster"] = df["track_id"].apply(
        lambda tid: 1.0 if (seed_cluster is not None and get_cluster_id(tid, models=models) == seed_cluster) else 0.0
    )

    # 4) Normalise each similarity to [0,1] within this candidate set
    def _norm(col: str) -> pd.Series:
        vals = df[col].values
        max_val = np.max(vals)
        if max_val <= 0:
            return pd.Series(np.zeros_like(vals, dtype=np.float32), index=df.index)
        return pd.Series((vals / max_val).astype(np.float32), index=df.index)

    df["cosine_norm"] = _norm("cosine_sim")
    df["als_norm"] = _norm("als_sim")
    df["item2vec_norm"] = _norm("item2vec_sim")

    # 5) Require any signal at all
    has_any_signal = (
        (df["cosine_norm"] > 0)
        | (df["als_norm"] > 0)
        | (df["item2vec_norm"] > 0)
    )
    df = df[has_any_signal]

    if df.empty:
        return df

    # 6) Hybrid score
    df["hybrid_score"] = (
        w_cos * df["cosine_norm"]
        + w_als * df["als_norm"]
        + w_i2v * df["item2vec_norm"]
        + w_cluster * df["same_cluster"]
    )

    # 7) Drop the seed, sort, top_k
    df = df[df["track_id"] != track_id]
    df = df.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)

    return df


def recommend_by_name_hybrid(
    query: str,
    candidate_index: int = 0,
    w_cos: float = 0.3,
    w_als: float = 0.3,
    w_i2v: float = 0.3,
    w_cluster: float = 0.1,
    top_k: int = 20,
    candidate_pool: int = HYBRID_CANDIDATE_POOL,
    max_search_results: int = 60,  # <-- match the app dropdown
):
    """
    Convenience wrapper:
      - fuzzy search by track_name within HYBRID universe
      - pick one candidate by index (safely clamped)
      - compute hybrid recommendations
      - attach metadata + audio features

    Returns:
      seed_row (Series), rec_df (DataFrame)
    """

    # --- 1. search ---
    hits = search_tracks_by_name(query, max_results=max_search_results)

    if hits.empty:
        print(f"No matches for query '{query}'")
        return None, pd.DataFrame()

    # Clamp candidate_index into valid range instead of raising
    if candidate_index < 0:
        candidate_index = 0
    if candidate_index >= len(hits):
        candidate_index = len(hits) - 1

    print("Search results:")
    display(hits.reset_index(drop=True))

    seed_row = hits.iloc[candidate_index]
    seed_tid = seed_row["track_id"]

    print("\nChosen seed track:")
    display(seed_row.to_frame().T)

    # --- 2. hybrid scores ---
    df_scores = hybrid_scores_for_track(
        seed_tid,
        w_cos=w_cos,
        w_als=w_als,
        w_i2v=w_i2v,
        w_cluster=w_cluster,
        top_k=top_k,
        candidate_pool=candidate_pool,
        verbose=True,
    )

    if df_scores.empty:
        print("No hybrid candidates found.")
        return seed_row, df_scores

    # --- 3. attach metadata + a few audio features ---
    rec_df = describe_tracks(
        df_scores["track_id"].tolist(),
        extra_cols=["danceability", "energy", "valence", "tempo"],
        top_n=None,
    )

    rec_df = rec_df.merge(df_scores, on="track_id", how="left")
    rec_df = rec_df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    print("\nHybrid recommendations:")
    display(rec_df.head(top_k))

    return seed_row, rec_df

