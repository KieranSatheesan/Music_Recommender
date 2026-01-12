from typing import List, Set, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import random
import math


def describe_tracks(track_id_list: List[str], meta_df: pd.DataFrame, 
                   top_n: int = 10) -> pd.DataFrame:
    """
    Given a list of track_ids, return a DataFrame with track metadata.
    
    Args:
        track_id_list: List of Spotify track IDs
        meta_df: DataFrame containing track metadata
        top_n: Maximum number of results to return
        
    Returns:
        DataFrame with track_id, track_name, artist_name, album_name
    """
    df = pd.DataFrame({"track_id": list(track_id_list)})
    df = df.merge(meta_df, on="track_id", how="left")
    return df.head(top_n)


def most_similar_tracks(query_track_id: str, 
                       X_norm: np.ndarray,
                       track_ids: List[str],
                       track_id_to_idx: dict,
                       meta_df: pd.DataFrame,
                       top_k: int = 10) -> pd.DataFrame:
    """
    Find most similar tracks based on cosine similarity in feature space.
    
    Args:
        query_track_id: Spotify track ID to find similar tracks for
        X_norm: Normalized feature matrix (n_tracks, n_features)
        track_ids: List of track IDs corresponding to rows in X_norm
        track_id_to_idx: Mapping from track_id to row index in X_norm
        meta_df: DataFrame containing track metadata
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame with recommendations including similarity scores and metadata
    """
    if query_track_id not in track_id_to_idx:
        raise ValueError(f"track_id {query_track_id} not found in feature matrix.")
    
    q_idx = track_id_to_idx[query_track_id]
    q_vec = X_norm[q_idx]
    
    # Cosine similarity = dot product since rows are L2-normalized
    sims = X_norm @ q_vec
    
    # Exclude self
    sims[q_idx] = -1.0
    
    # Get top_k indices
    top_idxs = np.argsort(sims)[-top_k:][::-1]
    top_ids = [track_ids[i] for i in top_idxs]
    top_sims = sims[top_idxs]
    
    result = pd.DataFrame({
        "track_id": top_ids,
        "similarity": top_sims,
    })
    
    # Attach metadata
    result = result.merge(meta_df, on="track_id", how="left")
    
    return result


def find_track_ids_by_name(query: str, meta_df: pd.DataFrame, 
                           max_results: int = 10) -> pd.DataFrame:
    """
    Case-insensitive search for tracks by name.
    
    Args:
        query: Search string
        meta_df: DataFrame containing track metadata
        max_results: Maximum number of results to return
        
    Returns:
        DataFrame with matching tracks
    """
    mask = meta_df["track_name"].str.contains(query, case=False, na=False)
    results = meta_df[mask].head(max_results)
    return results[["track_id", "track_name", "artist_name", "album_name"]]


def evaluate_cosine_on_playlists(interactions: pd.DataFrame,
                                 X_norm: np.ndarray,
                                 track_ids: List[str],
                                 track_id_to_idx: dict,
                                 meta_df: pd.DataFrame,
                                 num_playlists: int = 500,
                                 top_k: int = 10,
                                 seed: int = 0) -> Tuple[float, float, int]:
    """
    Evaluate recommendation quality using held-out playlist data.
    
    For each sampled playlist:
    - Pick one seed track
    - Generate recommendations
    - Measure recall and hit rate against other tracks in the playlist
    
    Args:
        interactions: DataFrame with columns [pid, track_id]
        X_norm: Normalized feature matrix
        track_ids: List of track IDs
        track_id_to_idx: Mapping from track_id to index
        meta_df: Track metadata DataFrame
        num_playlists: Number of playlists to evaluate on
        top_k: Number of recommendations to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (avg_recall_at_k, avg_hit_rate_at_k, n_eval_playlists)
    """
    from collections import defaultdict
    import random
    
    rng = random.Random(seed)
    
    # Build mapping: pid -> list of track_ids
    playlist_to_tracks = defaultdict(list)
    for row in interactions.itertuples(index=False):
        playlist_to_tracks[row.pid].append(row.track_id)
    
    all_pids = [pid for pid, tracks in playlist_to_tracks.items() if len(tracks) >= 2]
    if len(all_pids) == 0:
        raise ValueError("No playlists with 2+ tracks after filtering.")
    
    sampled_pids = rng.sample(all_pids, min(num_playlists, len(all_pids)))
    
    recalls = []
    hits = []
    
    for pid in sampled_pids:
        tracks = playlist_to_tracks[pid]
        seed_track = rng.choice(tracks)
        relevant = set(tracks) - {seed_track}
        
        if not relevant:
            continue
        
        try:
            rec_df = most_similar_tracks(
                seed_track, X_norm, track_ids, track_id_to_idx, meta_df, top_k=top_k
            )
        except ValueError:
            continue
        
        rec_ids = list(rec_df["track_id"])
        rec_set = set(rec_ids)
        
        n_hits = len(relevant & rec_set)
        recall = n_hits / len(relevant)
        hit = 1.0 if n_hits > 0 else 0.0
        
        recalls.append(recall)
        hits.append(hit)
    
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    avg_hit_rate = float(np.mean(hits)) if hits else 0.0
    
    return avg_recall, avg_hit_rate, len(recalls)


def recommend_by_name_with_differences(
    query: str,
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    X_norm: np.ndarray,
    track_ids: List[str],
    track_id_to_idx: dict,
    candidate_index: int = 0,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Search for tracks by name and provide recommendations with feature differences.
    
    For each recommendation, identifies which audio feature differs most from
    the seed track (in normalized units).
    
    Args:
        query: Search string for track name
        feat_df: DataFrame with track features
        meta_df: DataFrame with track metadata
        X_norm: Normalized feature matrix
        track_ids: List of track IDs
        track_id_to_idx: Mapping from track_id to index
        candidate_index: Which search result to use as seed
        top_k: Number of recommendations
        
    Returns:
        DataFrame with recommendations including feature differences
    """
    # Identify feature columns
    feature_cols = [c for c in feat_df.columns if c != "track_id"]
    
    # Compute global statistics
    feature_stats = feat_df[feature_cols].agg(["mean", "std"])
    feat_means = feature_stats.loc["mean"]
    feat_stds = feature_stats.loc["std"].replace(0, 1.0)
    
    # Search by name
    matches = find_track_ids_by_name(query, meta_df, max_results=20)
    
    if matches.empty:
        raise ValueError(f"No tracks found matching '{query}'")
    
    if candidate_index < 0 or candidate_index >= len(matches):
        raise IndexError(
            f"candidate_index {candidate_index} out of range for {len(matches)} matches"
        )
    
    print("Search results:")
    print(matches.reset_index(drop=True))
    
    # Choose seed track
    chosen = matches.iloc[candidate_index]
    seed_id = chosen["track_id"]
    
    print("\nChosen seed track:")
    print(chosen[["track_name", "artist_name", "album_name", "track_id"]])
    
    # Get recommendations
    recs_basic = most_similar_tracks(
        seed_id, X_norm, track_ids, track_id_to_idx, meta_df, top_k=top_k
    )
    
    # Merge in features
    recs = recs_basic.merge(
        feat_df[["track_id"] + feature_cols],
        on="track_id",
        how="left"
    )
    
    local_feature_cols = [c for c in feature_cols if c in recs.columns]
    if not local_feature_cols:
        raise ValueError("No feature columns found in recommendations.")
    
    # Seed features
    seed_features = (
        feat_df[feat_df["track_id"] == seed_id]
        .set_index("track_id")
        .loc[seed_id, local_feature_cols]
        .astype(float)
    )
    
    local_means = feat_means[local_feature_cols]
    local_stds = feat_stds[local_feature_cols].replace(0, 1.0)
    
    # Compute normalized differences
    seed_z = (seed_features - local_means) / local_stds
    
    most_diff_features = []
    most_diff_values = []
    
    for _, row in recs.iterrows():
        rec_feat = row[local_feature_cols].astype(float)
        rec_z = (rec_feat - local_means) / local_stds
        
        diff_z = (rec_z - seed_z).abs()
        max_feature = diff_z.idxmax()
        max_value = diff_z[max_feature]
        
        most_diff_features.append(max_feature)
        most_diff_values.append(max_value)
    
    recs["most_diff_feature"] = most_diff_features
    recs["most_diff_abs_zdiff"] = most_diff_values
    
    # Display seed features
    display_seed = (
        pd.DataFrame(seed_features)
        .T.assign(
            track_id=seed_id,
            track_name=chosen["track_name"],
            artist_name=chosen["artist_name"],
            album_name=chosen["album_name"],
        )[["track_id", "track_name", "artist_name", "album_name"] + local_feature_cols]
    )
    
    print("\nSeed track audio features:")
    print(display_seed)
    
    cols_to_show = (
        ["track_name", "artist_name", "album_name", "similarity"] +
        local_feature_cols +
        ["most_diff_feature", "most_diff_abs_zdiff"]
    )
    
    print("\nRecommendations with feature differences:")
    print(recs[cols_to_show])
    
    return recs

def plot_similarity_distribution(sims: np.ndarray, 
                                 seed_track_name: str,
                                 seed_artist: str,
                                 figsize: tuple = (5, 6)) -> None:
    """
    Plot histogram of cosine similarities to a seed track.
    
    Args:
        sims: Array of similarity scores
        seed_track_name: Name of seed track
        seed_artist: Artist of seed track
        figsize: Figure size tuple
    """
    # Filter out self-similarity (-1.0 markers)
    valid_sims = sims[sims > -1]
    
    plt.figure(figsize=figsize)
    plt.hist(valid_sims, bins=60, color='#1DB954', alpha=0.7, 
            edgecolor='black', linewidth=0.5)
    
    mean_sim = np.mean(valid_sims)
    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_sim:.3f}')
    
    plt.xlabel("Cosine Similarity to Seed Track", fontsize=10, fontweight='bold')
    plt.ylabel("Number of Tracks", fontsize=10, fontweight='bold')
    plt.title(f"Distribution of Cosine Similarities to '{seed_track_name}' by {seed_artist}",
             fontsize=10, fontweight='bold', pad=15)
    plt.legend(fontsize=8, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_pca_recommendations(X_norm: np.ndarray,
                            track_ids: List[str],
                            track_id_to_idx: dict,
                            seed_track_id: str,
                            top_rec_ids: List[str],
                            meta_df: pd.DataFrame,
                            n_background: int = 200000,
                            seed: int = 123,
                            figsize: tuple = (10, 8)) -> None:
    """
    Create PCA visualization of seed track, recommendations, and background tracks.
    
    Args:
        X_norm: Normalized feature matrix
        track_ids: List of track IDs
        track_id_to_idx: Mapping from track_id to index
        seed_track_id: ID of seed track
        top_rec_ids: List of recommended track IDs
        meta_df: DataFrame with track metadata
        n_background: Number of random background tracks to plot
        seed: Random seed
        figsize: Figure size tuple
    """
    np.random.seed(seed)
    
    all_indices = np.arange(len(track_ids))
    bg_indices = np.random.choice(all_indices, 
                                  size=min(n_background, len(all_indices)), 
                                  replace=False)
    
    q_idx = track_id_to_idx[seed_track_id]
    top_indices = np.array([track_id_to_idx[tid] for tid in top_rec_ids])
    
    indices_to_plot = np.unique(np.concatenate([[q_idx], top_indices, bg_indices]))
    X_subset = X_norm[indices_to_plot]
    
    # PCA to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_subset)
    
    global_to_2d_idx = {g: i for i, g in enumerate(indices_to_plot)}
    
    seed_2d = X_2d[global_to_2d_idx[q_idx]]
    top_2d = X_2d[[global_to_2d_idx[i] for i in top_indices]]
    bg_2d = X_2d[[global_to_2d_idx[i] for i in bg_indices if i in global_to_2d_idx]]
    
    top_rec_idx = top_indices[0]
    top_rec_2d = X_2d[global_to_2d_idx[top_rec_idx]]
    
    # Get track names
    seed_name = meta_df[meta_df["track_id"] == seed_track_id]["track_name"].values[0]
    seed_artist = meta_df[meta_df["track_id"] == seed_track_id]["artist_name"].values[0]
    top_rec_name = meta_df[meta_df["track_id"] == top_rec_ids[0]]["track_name"].values[0]
    top_rec_artist = meta_df[meta_df["track_id"] == top_rec_ids[0]]["artist_name"].values[0]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(bg_2d[:, 0], bg_2d[:, 1], s=8, alpha=0.15, c='gray', 
              label="Other tracks")
    
    ax.scatter(top_2d[1:, 0], top_2d[1:, 1], s=80, alpha=0.7, c="#7EF7A8",
              edgecolors='black', linewidth=0.5, label="Top neighbours", zorder=3)
    
    ax.scatter(top_rec_2d[0], top_rec_2d[1], s=120, alpha=0.9, c="#078502",
              edgecolors='black', linewidth=1.5, label="Top recommendation", 
              zorder=4, marker='D')
    
    ax.scatter(seed_2d[0], seed_2d[1], s=200, marker="*", c="#FF0000",
              edgecolors='black', linewidth=1.5, label="Seed track", zorder=5)
    
    # Calculate offsets
    x_range = bg_2d[:, 0].max() - bg_2d[:, 0].min()
    y_range = bg_2d[:, 1].max() - bg_2d[:, 1].min()
    offset_x = x_range * 0.12
    offset_y = y_range * 0.12
    
    # Add labels
    ax.annotate(f"{seed_name}\n({seed_artist})",
               xy=(seed_2d[0], seed_2d[1]),
               xytext=(seed_2d[0] + offset_x, seed_2d[1] + offset_y),
               bbox=dict(boxstyle="round,pad=0.5", facecolor='red', 
                        alpha=0.7, edgecolor='black'),
               fontsize=11, fontweight='bold', color='white',
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5),
               zorder=6)
    
    top_rec_display = (top_rec_name if len(top_rec_name) <= 25 
                      else top_rec_name[:22] + "...")
    ax.annotate(f"{top_rec_display}\n({top_rec_artist})",
               xy=(top_rec_2d[0], top_rec_2d[1]),
               xytext=(top_rec_2d[0] + offset_x, top_rec_2d[1] - offset_y * 1.5),
               bbox=dict(boxstyle="round,pad=0.5", facecolor='#078502',
                        alpha=0.8, edgecolor='black'),
               fontsize=11, fontweight='bold',
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', lw=1.5),
               zorder=6)
    
    ax.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='black')
    ax.set_title("PCA Projection of Feature Space\nSeed Track vs. Neighbours vs. Random Tracks",
                fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
                 fontsize=15, fontweight='bold')
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

def fit_knn_models(X_z: np.ndarray, X_pca5: np.ndarray, X_pca8: np.ndarray,
                   n_neighbors: int = 51) -> Dict[str, Dict]:
    """
    Fit multiple kNN models with different metrics and feature spaces.
    
    Args:
        X_z: Z-score normalized features
        X_pca5: PCA-reduced features (5 components)
        X_pca8: PCA-reduced features (8 components)
        n_neighbors: Number of neighbors to fit (includes self)
        
    Returns:
        Dictionary of fitted models with their specifications
    """
    models = {}
    
    def fit_single(name, X, metric):
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(X)
        models[name] = {
            "nn": nn,
            "X": X,
            "metric": metric,
        }
        print(f"Fitted model '{name}' on X.shape={X.shape}, metric='{metric}'")
    
    fit_single("cosine_z", X_z, "cosine")
    fit_single("euclidean_z", X_z, "euclidean")
    fit_single("manhattan_z", X_z, "manhattan")
    fit_single("pca5_euclidean", X_pca5, "euclidean")
    fit_single("pca8_euclidean", X_pca8, "euclidean")
    
    return models


def knn_recommend(
    model_name: str,
    models: Dict[str, Dict],
    seed_track_id: str,
    track_ids: List[str],
    track_id_to_idx: Dict[str, int],
    meta_df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Generate recommendations using a specified kNN model.
    
    Args:
        model_name: Name of the model to use
        models: Dictionary of fitted kNN models
        seed_track_id: Track ID to find neighbors for
        track_ids: List of all track IDs
        track_id_to_idx: Mapping from track_id to index
        meta_df: DataFrame with track metadata
        top_k: Number of recommendations to return
        
    Returns:
        DataFrame with recommendations including distances and metadata
    """
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(models.keys())}")
    if seed_track_id not in track_id_to_idx:
        raise ValueError(f"track_id {seed_track_id} not in track_id_to_idx.")
    
    nn = models[model_name]["nn"]
    X = models[model_name]["X"]
    
    idx = track_id_to_idx[seed_track_id]
    seed_vec = X[idx].reshape(1, -1)
    
    distances, indices = nn.kneighbors(seed_vec, n_neighbors=top_k + 1)
    distances = distances[0]
    indices = indices[0]
    
    # Exclude self if present
    mask = indices != idx
    distances = distances[mask]
    indices = indices[mask]
    
    distances = distances[:top_k]
    indices = indices[:top_k]
    
    neighbor_ids = [track_ids[i] for i in indices]
    
    # Convert distance to similarity for cosine metric
    metric = models[model_name]["metric"]
    if metric == "cosine":
        similarities = 1.0 - distances
    else:
        similarities = [np.nan] * len(distances)
    
    recs = pd.DataFrame({
        "track_id": neighbor_ids,
        "distance": distances,
        "similarity": similarities,
    })
    
    recs = recs.merge(meta_df, on="track_id", how="left")
    
    return recs


def compare_metrics_for_track_name(
    query: str,
    models: Dict[str, Dict],
    track_ids: List[str],
    track_id_to_idx: Dict[str, int],
    meta_df: pd.DataFrame,
    candidate_index: int = 0,
    top_k: int = 10,
    model_names: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Compare recommendations across multiple models for a search query.
    
    Args:
        query: Search string for track name
        models: Dictionary of fitted kNN models
        track_ids: List of all track IDs
        track_id_to_idx: Mapping from track_id to index
        meta_df: DataFrame with track metadata
        candidate_index: Which search result to use as seed
        top_k: Number of recommendations
        model_names: List of models to compare (None = all)
        
    Returns:
        Dictionary mapping model names to recommendation DataFrames
    """
    if model_names is None:
        model_names = list(models.keys())
    
    # Search helper (inline for simplicity, or import from another module)
    mask = meta_df["track_name"].str.contains(query, case=False, na=False)
    matches = meta_df[mask].head(20)
    
    if matches.empty:
        raise ValueError(f"No tracks found matching '{query}'")
    
    if not (0 <= candidate_index < len(matches)):
        raise IndexError(f"candidate_index {candidate_index} out of range")
    
    print("Search results:")
    print(matches.reset_index(drop=True)[["track_id", "track_name", "artist_name", "album_name"]])
    
    chosen = matches.iloc[candidate_index]
    seed_id = chosen["track_id"]
    
    print("\nChosen seed track:")
    print(chosen[["track_name", "artist_name", "album_name", "track_id"]])
    
    results = {}
    for mname in model_names:
        print(f"\n=== Recommendations for model: {mname} ===")
        recs = knn_recommend(mname, models, seed_id, track_ids, track_id_to_idx, 
                            meta_df, top_k=top_k)
        print(recs[["track_name", "artist_name", "album_name", "distance", "similarity"]])
        results[mname] = recs
    
    return results


def evaluate_knn_model(
    model_name: str,
    models: Dict[str, Dict],
    track_ids: List[str],
    track_id_to_idx: Dict[str, int],
    meta_df: pd.DataFrame,
    playlist_to_tracks: Dict[int, List[str]],
    playlist_ids_eval: List[int],
    num_playlists: int = 500,
    top_k: int = 10,
    seed: int = 0,
) -> Tuple[float, float, int]:
    """
    Evaluate a kNN model using playlist-based held-out testing.
    
    For each playlist:
    - Pick one seed track
    - Measure how many other playlist tracks appear in top_k recommendations
    
    Args:
        model_name: Name of model to evaluate
        models: Dictionary of fitted models
        track_ids: List of track IDs
        track_id_to_idx: Track ID to index mapping
        meta_df: Metadata DataFrame
        playlist_to_tracks: Mapping from playlist ID to track list
        playlist_ids_eval: List of playlist IDs to sample from
        num_playlists: Number of playlists to evaluate
        top_k: Cutoff for recommendations
        seed: Random seed
        
    Returns:
        Tuple of (avg_recall, avg_hit_rate, n_evaluated)
    """
    rng = random.Random(seed)
    
    sampled_pids = rng.sample(
        playlist_ids_eval,
        min(num_playlists, len(playlist_ids_eval))
    )
    
    recalls = []
    hits = []
    
    for pid in sampled_pids:
        tracks = playlist_to_tracks[pid]
        seed_track = rng.choice(tracks)
        relevant = set(tracks) - {seed_track}
        
        if not relevant:
            continue
        
        try:
            rec_df = knn_recommend(model_name, models, seed_track, track_ids,
                                  track_id_to_idx, meta_df, top_k=top_k)
        except ValueError:
            continue
        
        rec_ids = set(rec_df["track_id"])
        
        n_hits = len(relevant & rec_ids)
        recall = n_hits / len(relevant)
        hit = 1.0 if n_hits > 0 else 0.0
        
        recalls.append(recall)
        hits.append(hit)
    
    if not recalls:
        return 0.0, 0.0, 0
    
    avg_recall = float(np.mean(recalls))
    avg_hit_rate = float(np.mean(hits))
    return avg_recall, avg_hit_rate, len(recalls)

def average_precision_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute Average Precision@k for a single ranked list.
    
    Args:
        ranked_ids: List of recommended track IDs (ordered)
        relevant_ids: Set of ground-truth relevant track IDs
        k: Cutoff position
        
    Returns:
        AP@k score in [0, 1]
    """
    if not relevant_ids:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    for i, tid in enumerate(ranked_ids[:k], start=1):
        if tid in relevant_ids:
            hits += 1
            precision_i = hits / i
            sum_precisions += precision_i
    
    denom = min(len(relevant_ids), k)
    if denom == 0:
        return 0.0
    return sum_precisions / denom


def ndcg_at_k(ranked_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Compute NDCG@k for binary relevance.
    
    Args:
        ranked_ids: List of recommended track IDs (ordered)
        relevant_ids: Set of ground-truth relevant track IDs
        k: Cutoff position
        
    Returns:
        NDCG@k score in [0, 1]
    """
    # DCG
    dcg = 0.0
    for i, tid in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if tid in relevant_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    
    # IDCG (ideal DCG)
    max_rels = min(len(relevant_ids), k)
    if max_rels == 0:
        return 0.0
    
    idcg = 0.0
    for i in range(1, max_rels + 1):
        idcg += 1.0 / math.log2(i + 1)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_knn_model_multi_k(
    model_name: str,
    models: Dict,
    track_ids: List[str],
    track_id_to_idx: Dict[str, int],
    meta_df: pd.DataFrame,
    playlist_to_tracks: Dict[int, List[str]],
    playlist_ids_eval: List[int],
    Ks: Tuple[int, ...] = (5, 10, 20, 50),
    num_playlists: int = 500,
    seed: int = 0,
) -> Tuple[pd.DataFrame, int]:
    """
    Evaluate a kNN model across multiple K values with comprehensive metrics.
    
    Computes: Recall@K, HitRate@K, Precision@K, MAP@K, NDCG@K
    
    Args:
        model_name: Name of model to evaluate
        models: Dictionary of fitted models
        track_ids: List of track IDs
        track_id_to_idx: Track ID to index mapping
        meta_df: Metadata DataFrame
        playlist_to_tracks: Playlist to track list mapping
        playlist_ids_eval: List of playlist IDs to sample
        Ks: Tuple of K values to evaluate
        num_playlists: Number of playlists to evaluate
        seed: Random seed
        
    Returns:
        Tuple of (metrics_df, n_evaluated)
    """
    from src.recommender_helpers import knn_recommend
    
    rng = random.Random(seed)
    Ks = sorted(list(Ks))
    max_k = Ks[-1]
    
    sampled_pids = rng.sample(
        playlist_ids_eval,
        min(num_playlists, len(playlist_ids_eval))
    )
    
    # Accumulators
    stats = {
        "Recall": {k: [] for k in Ks},
        "HitRate": {k: [] for k in Ks},
        "Precision": {k: [] for k in Ks},
        "MAP": {k: [] for k in Ks},
        "NDCG": {k: [] for k in Ks},
    }
    
    n_eval = 0
    
    for pid in sampled_pids:
        tracks = playlist_to_tracks[pid]
        if len(tracks) < 2:
            continue
        
        seed_track = rng.choice(tracks)
        relevant = set(tracks) - {seed_track}
        if not relevant:
            continue
        
        try:
            rec_df = knn_recommend(model_name, models, seed_track, track_ids,
                                  track_id_to_idx, meta_df, top_k=max_k)
        except ValueError:
            continue
        
        ranked_ids = list(rec_df["track_id"])
        if not ranked_ids:
            continue
        
        n_eval += 1
        
        for k in Ks:
            topk = ranked_ids[:k]
            topk_set = set(topk)
            
            n_hits = len(relevant & topk_set)
            recall_k = n_hits / len(relevant)
            hit_k = 1.0 if n_hits > 0 else 0.0
            precision_k = n_hits / len(topk) if topk else 0.0
            ap_k = average_precision_at_k(ranked_ids, relevant, k)
            ndcg_k = ndcg_at_k(ranked_ids, relevant, k)
            
            stats["Recall"][k].append(recall_k)
            stats["HitRate"][k].append(hit_k)
            stats["Precision"][k].append(precision_k)
            stats["MAP"][k].append(ap_k)
            stats["NDCG"][k].append(ndcg_k)
    
    # Aggregate
    data = {}
    for metric_name, per_k in stats.items():
        data[metric_name] = {
            k: float(np.mean(vals)) if len(vals) > 0 else 0.0
            for k, vals in per_k.items()
        }
    
    metrics_df = pd.DataFrame(data).T
    metrics_df.columns = [f"@{k}" for k in Ks]
    metrics_df.index.name = f"{model_name} metrics"
    
    return metrics_df, n_eval

def plot_knn_metrics_comparison(multi_k_results: Dict[str, pd.DataFrame],
                                model_names: List[str],
                                K_LIST: List[int],
                                figsize: tuple = (14, 6)) -> None:
    """
    Create comprehensive visualization of kNN model performance.
    
    Args:
        multi_k_results: Dictionary mapping model names to metric DataFrames
        model_names: List of model names to plot
        K_LIST: List of K values
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'd']
    
    # Plot 1: HitRate@K
    for idx, mname in enumerate(model_names):
        df = multi_k_results[mname]
        ks = [int(col.strip("@")) for col in df.columns]
        axes[0].plot(
            ks,
            df.loc["HitRate"].values,
            marker=markers[idx % len(markers)],
            markersize=8,
            linewidth=2,
            color=colors[idx],
            label=mname,
            alpha=0.8
        )
    
    axes[0].set_xlabel("Recommendation List Size (K)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("HitRate@K", fontsize=12, fontweight='bold')
    axes[0].set_title("KNN Evaluation - HitRate@K vs K", fontsize=14, fontweight='bold')
    axes[0].set_xticks(K_LIST)
    axes[0].tick_params(axis='both', which='major', labelsize=10)
    axes[0].legend(fontsize=10, loc='best', framealpha=0.9, shadow=True)
    axes[0].grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    
    # Plot 2: NDCG@K
    for idx, mname in enumerate(model_names):
        df = multi_k_results[mname]
        ks = [int(col.strip("@")) for col in df.columns]
        axes[1].plot(
            ks,
            df.loc["NDCG"].values,
            marker=markers[idx % len(markers)],
            markersize=8,
            linewidth=2,
            color=colors[idx],
            label=mname,
            alpha=0.8
        )
    
    axes[1].set_xlabel("Recommendation List Size (K)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("NDCG@K", fontsize=12, fontweight='bold')
    axes[1].set_title("KNN Evaluation - NDCG@K vs K", fontsize=14, fontweight='bold')
    axes[1].set_xticks(K_LIST)
    axes[1].tick_params(axis='both', which='major', labelsize=10)
    axes[1].legend(fontsize=10, loc='upper left', framealpha=0.9, shadow=True)
    axes[1].grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_all_metrics_at_k(multi_k_results: Dict[str, pd.DataFrame],
                          model_names: List[str],
                          k: int = 10,
                          figsize: tuple = (12, 7)) -> None:
    """
    Create bar chart comparing all metrics at a specific K value.
    
    Args:
        multi_k_results: Dictionary mapping model names to metric DataFrames
        model_names: List of model names
        k: K value to visualize
        figsize: Figure size tuple
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(model_names))
    width = 0.15
    metrics = ['Recall', 'HitRate', 'Precision', 'MAP', 'NDCG']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        values = []
        for mname in model_names:
            df = multi_k_results[mname]
            values.append(df.loc[metric, f"@{k}"])
        
        ax.bar(x + i*width - 2*width, values, width, label=metric, 
              color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'KNN Evaluation - Performance Comparison at K={k}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9, shadow=True)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()