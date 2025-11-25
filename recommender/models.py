# recommender/models.py

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from scipy.sparse import coo_matrix
import implicit
from gensim.models import Word2Vec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class RecommenderModels:
    # Core dataframes
    feat_df_u: pd.DataFrame
    meta_df_u: pd.DataFrame
    inter_df_u: pd.DataFrame

    # Metadata shortcut
    track_df: pd.DataFrame  # ["track_id","track_name","artist_name","album_name"]

    # Audio feature space
    feature_cols: list[str]
    scaler_u: StandardScaler
    X_norm_u: np.ndarray
    tid_to_feat_idx_u: dict[str, int]
    knn_cosine_u: NearestNeighbors

    # Item2Vec
    item2vec_kv: any  # gensim KeyedVectors

    # ALS
    als_model: implicit.als.AlternatingLeastSquares
    tid_to_item_idx_als: dict[str, int]
    idx_to_tid_als: np.ndarray

    # Clusters
    track_cluster_df: pd.DataFrame


def load_item2vec_model() -> Word2Vec:
    item2vec_path = MODELS_DIR / "item2vec_word2vec.model"
    if not item2vec_path.exists():
        raise FileNotFoundError(f"Item2Vec model not found at: {item2vec_path}")
    model = Word2Vec.load(str(item2vec_path))
    return model


def load_raw_data():
    features_path = DATA_PROCESSED / "combined_features.csv"
    interactions_path = DATA_PROCESSED / "interactions.parquet"
    track_meta_path = DATA_PROCESSED / "track_metadata.csv"

    feat_df = pd.read_csv(features_path)
    meta_df = pd.read_csv(track_meta_path)
    inter_df = pd.read_parquet(interactions_path)

    # Ensure track_id is string everywhere
    for df in (feat_df, meta_df, inter_df):
        df["track_id"] = df["track_id"].astype(str)

    return feat_df, meta_df, inter_df


def build_hybrid_universe(
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    item2vec_kv,
):
    # First: tracks present in features + meta + interactions
    track_ids_all = (
        set(feat_df["track_id"])
        & set(meta_df["track_id"])
        & set(inter_df["track_id"])
    )

    feat_df = feat_df[feat_df["track_id"].isin(track_ids_all)].reset_index(drop=True)
    meta_df = meta_df[meta_df["track_id"].isin(track_ids_all)].reset_index(drop=True)
    inter_df = inter_df[inter_df["track_id"].isin(track_ids_all)].reset_index(drop=True)

    # Now intersect with item2vec vocab
    track_universe = (
        set(feat_df["track_id"])
        & set(meta_df["track_id"])
        & set(inter_df["track_id"])
        & set(item2vec_kv.key_to_index.keys())
    )

    feat_df_u = feat_df[feat_df["track_id"].isin(track_universe)].reset_index(drop=True)
    meta_df_u = meta_df[meta_df["track_id"].isin(track_universe)].reset_index(drop=True)
    inter_df_u = inter_df[inter_df["track_id"].isin(track_universe)].reset_index(drop=True)

    # Simple metadata index
    track_df = (
        meta_df_u[["track_id", "track_name", "artist_name", "album_name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return feat_df_u, meta_df_u, inter_df_u, track_df


def build_feature_space(feat_df_u: pd.DataFrame):
    # Choose numeric feature columns (everything except ids / non-numerics)
    non_feature_cols = {"track_id", "explicit"}
    feature_cols = [
        c for c in feat_df_u.columns
        if c not in non_feature_cols and np.issubdtype(feat_df_u[c].dtype, np.number)
    ]

    feat_mat_u = feat_df_u[feature_cols].copy()
    feat_mat_u = feat_mat_u.fillna(feat_mat_u.mean())

    scaler_u = StandardScaler()
    X_scaled_u = scaler_u.fit_transform(feat_mat_u.values)

    row_norms_u = np.linalg.norm(X_scaled_u, axis=1, keepdims=True)
    row_norms_u[row_norms_u == 0.0] = 1.0
    X_norm_u = X_scaled_u / row_norms_u

    track_ids_u = feat_df_u["track_id"].values
    tid_to_feat_idx_u = {tid: i for i, tid in enumerate(track_ids_u)}

    knn_cosine_u = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_cosine_u.fit(X_norm_u)

    return feature_cols, scaler_u, X_norm_u, tid_to_feat_idx_u, knn_cosine_u


def train_als_full(inter_df: pd.DataFrame):
    als_inter = inter_df.copy()

    pid_codes = als_inter["pid"].astype("category")       # playlists = users
    tid_codes = als_inter["track_id"].astype("category")  # tracks = items

    als_inter["pid_idx"] = pid_codes.cat.codes
    als_inter["tid_idx"] = tid_codes.cat.codes

    n_users = als_inter["pid_idx"].nunique()
    n_items = als_inter["tid_idx"].nunique()

    alpha = 1.0
    data = np.ones(len(als_inter), dtype=np.float32) * alpha
    rows = als_inter["pid_idx"].values
    cols = als_inter["tid_idx"].values

    user_items = coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()

    als_model = implicit.als.AlternatingLeastSquares(
        factors=64,
        regularization=0.01,
        iterations=10,
        random_state=42,
    )
    als_model.fit(user_items)

    idx_to_tid_als = np.array(tid_codes.cat.categories)
    n_items_als = als_model.item_factors.shape[0]

    assert n_items_als == n_items == len(idx_to_tid_als)

    tid_to_item_idx_als = {tid: i for i, tid in enumerate(idx_to_tid_als)}

    return als_model, tid_to_item_idx_als, idx_to_tid_als


def cluster_tracks(X_norm_u: np.ndarray, track_ids_u: np.ndarray, n_clusters: int = 20):
    kmeans_tracks = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
    )
    track_cluster_labels = kmeans_tracks.fit_predict(X_norm_u)

    track_cluster_df = pd.DataFrame(
        {
            "track_id": track_ids_u,
            "cluster_id": track_cluster_labels,
        }
    ).set_index("track_id")

    return track_cluster_df


# --- MAIN ENTRYPOINT ---


_models_cache: RecommenderModels | None = None


def load_all_models() -> RecommenderModels:
    """
    Build and cache all models / structures needed for the hybrid recommender.
    Safe to call multiple times; only builds once per Python process.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    # 1. Load raw data
    feat_df, meta_df, inter_df = load_raw_data()

    # 2. Load item2vec
    item2vec_model = load_item2vec_model()
    item2vec_kv = item2vec_model.wv

    # 3. Build HYBRID universe
    feat_df_u, meta_df_u, inter_df_u, track_df = build_hybrid_universe(
        feat_df, meta_df, inter_df, item2vec_kv
    )

    # 4. Feature space + cosine index
    (
        feature_cols,
        scaler_u,
        X_norm_u,
        tid_to_feat_idx_u,
        knn_cosine_u,
    ) = build_feature_space(feat_df_u)

    # 5. ALS on FULL interactions
    als_model, tid_to_item_idx_als, idx_to_tid_als = train_als_full(inter_df)

    # 6. Clusters (on HYBRID universe)
    track_ids_u = feat_df_u["track_id"].values
    track_cluster_df = cluster_tracks(X_norm_u, track_ids_u, n_clusters=20)

    _models_cache = RecommenderModels(
        feat_df_u=feat_df_u,
        meta_df_u=meta_df_u,
        inter_df_u=inter_df_u,
        track_df=track_df,
        feature_cols=feature_cols,
        scaler_u=scaler_u,
        X_norm_u=X_norm_u,
        tid_to_feat_idx_u=tid_to_feat_idx_u,
        knn_cosine_u=knn_cosine_u,
        item2vec_kv=item2vec_kv,
        als_model=als_model,
        tid_to_item_idx_als=tid_to_item_idx_als,
        idx_to_tid_als=idx_to_tid_als,
        track_cluster_df=track_cluster_df,
    )

    return _models_cache
