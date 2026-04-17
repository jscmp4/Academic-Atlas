"""
Stage 4: CLUSTER — UMAP dimensionality reduction + BERTopic/KMeans clustering.

Dual path:
- Primary: BERTopic (UMAP + HDBSCAN + c-TF-IDF topic labels)
- Fallback: UMAP + KMeans (if BERTopic import fails)

Outputs papers_clustered.csv with x, y, cluster_id, cluster_label, topic_words.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import (
    get_logger,
    load_config,
    load_papers_csv,
    save_papers_csv,
    setup_logging,
    CLUSTER_COLUMNS_EXTRA,
    DATA_DIR,
)

logger = get_logger("cluster")


# ---------------------------------------------------------------------------
# Quality summary
# ---------------------------------------------------------------------------
def print_cluster_summary(df: pd.DataFrame) -> None:
    """Print clustering quality summary to log and console."""
    total = len(df)
    n_clusters = df["cluster_id"].nunique()
    outliers = (df["cluster_id"] == -1).sum()
    outlier_pct = outliers / total * 100 if total > 0 else 0

    lines = [
        "",
        "=" * 60,
        "Clustering Summary",
        "=" * 60,
        f"  Total papers: {total:,}",
        f"  Clusters found: {n_clusters} (excluding outliers)" if outliers > 0 else f"  Clusters found: {n_clusters}",
    ]

    # Per-cluster stats
    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            continue
        mask = df["cluster_id"] == cid
        count = mask.sum()
        label = df.loc[mask, "cluster_label"].iloc[0] if mask.any() else "?"
        lines.append(f"  Cluster #{cid}: {count} papers - \"{label}\"")

    if outliers > 0:
        lines.append(f"  Outliers: {outliers} papers ({outlier_pct:.1f}%)")

    if outlier_pct > 40:
        lines.append(
            "  WARNING: Outlier ratio > 40%! Consider adjusting parameters:"
        )
        lines.append(
            "    - Lower min_cluster_size or min_samples in config.yaml"
        )

    lines.append("=" * 60)

    summary = "\n".join(lines)
    logger.info(summary)
    print(summary)


# ---------------------------------------------------------------------------
# BERTopic clustering (primary)
# ---------------------------------------------------------------------------
def cluster_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    config: dict,
    df: "pd.DataFrame | None" = None,
) -> tuple[list[int], list[str], np.ndarray]:
    """Cluster with BERTopic. Returns (topics, topic_labels, reduced_2d)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    clust_cfg = config.get("clustering", {})

    # UMAP for clustering (high-dim → 5D)
    umap_cluster = UMAP(
        n_neighbors=clust_cfg.get("n_neighbors", 15),
        n_components=clust_cfg.get("n_components_cluster", 5),
        min_dist=clust_cfg.get("min_dist_cluster", 0.0),
        metric="cosine",
        random_state=clust_cfg.get("random_state", 42),
    )

    # HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=clust_cfg.get("min_cluster_size", 10),
        min_samples=clust_cfg.get("min_samples", 5),
        prediction_data=True,
    )

    # Vectorizer with English stop words removed for better topic labels
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    logger.info("Running BERTopic clustering...")
    topic_model = BERTopic(
        umap_model=umap_cluster,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    # Reduce outliers if configured
    if clust_cfg.get("reduce_outliers", True):
        n_outliers_before = topics.count(-1)
        if n_outliers_before > 0:
            logger.info(f"Reducing {n_outliers_before} outliers...")
            topics = topic_model.reduce_outliers(docs, topics)
            n_outliers_after = topics.count(-1)
            logger.info(
                f"Outliers: {n_outliers_before} → {n_outliers_after}"
            )

    # Generate topic labels: OpenAlex topic majority vote, deduplicated across clusters
    has_topic = df is not None and "topic" in df.columns and df["topic"].notna().any()
    topic_label_map = {}
    used_labels = set()

    # Sort clusters by size (largest first) so the biggest cluster gets the "clean" label
    cluster_sizes = {}
    for tid in set(topics):
        cluster_sizes[tid] = sum(1 for t in topics if t == tid)
    sorted_tids = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)

    for tid in sorted_tids:
        indices = [i for i, t in enumerate(topics) if t == tid]
        if not indices:
            topic_label_map[tid] = f"Topic {tid}"
            continue

        label = None

        if has_topic:
            cluster_topics = df.iloc[indices]["topic"].dropna()
            cluster_topics = cluster_topics[cluster_topics != ""]
            if len(cluster_topics) > 0:
                # Try each topic in frequency order until we find an unused one
                for candidate in cluster_topics.value_counts().index:
                    if candidate not in used_labels:
                        label = candidate
                        break
                # All topics taken — differentiate with BERTopic keyword
                if label is None:
                    base = cluster_topics.value_counts().index[0]
                    kw = topic_model.get_topic(tid)
                    if kw and isinstance(kw, list):
                        label = f"{base} ({kw[0][0]})"
                    else:
                        label = f"{base} #{tid}"

        if label is None:
            kw = topic_model.get_topic(tid)
            if kw and isinstance(kw, list):
                label = kw[0][0].title()
            else:
                label = f"Topic {tid}"

        topic_label_map[tid] = label
        used_labels.add(label)

    topic_labels = [topic_label_map.get(t, "Outlier") for t in topics]

    # Build topic_words strings for CSV
    topic_words_list = []
    for t in topics:
        tw = topic_model.get_topic(t)
        if tw and isinstance(tw, list):
            topic_words_list.append("; ".join(f"{w}({s:.3f})" for w, s in tw[:10]))
        else:
            topic_words_list.append("")

    # UMAP for visualization (high-dim → 2D)
    logger.info("Running UMAP for 2D visualization...")
    umap_viz = UMAP(
        n_neighbors=clust_cfg.get("n_neighbors", 15),
        n_components=clust_cfg.get("n_components_viz", 2),
        min_dist=clust_cfg.get("min_dist_viz", 0.1),
        metric="cosine",
        random_state=clust_cfg.get("random_state", 42),
    )
    reduced_2d = umap_viz.fit_transform(embeddings)

    # Save BERTopic model
    try:
        model_dir = DATA_DIR / "bertopic_model"
        topic_model.save(str(model_dir), serialization="safetensors", save_ctfidf=True)
        logger.info(f"BERTopic model saved to {model_dir}")
    except Exception as e:
        logger.warning(f"Could not save BERTopic model: {e}")

    return topics, topic_labels, reduced_2d, topic_words_list


# ---------------------------------------------------------------------------
# KMeans clustering (fallback)
# ---------------------------------------------------------------------------
def cluster_kmeans(
    docs: list[str],
    embeddings: np.ndarray,
    config: dict,
) -> tuple[list[int], list[str], np.ndarray, list[str]]:
    """Fallback clustering: UMAP + KMeans + TF-IDF labels."""
    from sklearn.cluster import KMeans

    clust_cfg = config.get("clustering", {})
    n_clusters = clust_cfg.get("kmeans_n_clusters", 12)

    logger.info(f"Fallback: KMeans clustering with {n_clusters} clusters...")

    # UMAP → 2D (used for both clustering and visualization)
    try:
        from umap import UMAP
        umap_model = UMAP(
            n_neighbors=clust_cfg.get("n_neighbors", 15),
            n_components=2,
            min_dist=clust_cfg.get("min_dist_viz", 0.1),
            metric="cosine",
            random_state=clust_cfg.get("random_state", 42),
        )
        reduced_2d = umap_model.fit_transform(embeddings)
    except ImportError:
        logger.warning("UMAP not available, using PCA for dimensionality reduction")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=clust_cfg.get("random_state", 42))
        reduced_2d = pca.fit_transform(embeddings)

    # KMeans on original embeddings
    km = KMeans(
        n_clusters=n_clusters,
        random_state=clust_cfg.get("random_state", 42),
        n_init=10,
    )
    topics = km.fit_predict(embeddings).tolist()

    # Generate labels using TF-IDF top words per cluster
    logger.info("Generating cluster labels via TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    topic_label_map = {}
    for cid in range(n_clusters):
        indices = [i for i, t in enumerate(topics) if t == cid]
        if not indices:
            topic_label_map[cid] = f"Cluster {cid}"
            continue
        cluster_tfidf = tfidf_matrix[indices].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_label_map[cid] = ", ".join(top_words)

    topic_labels = [topic_label_map.get(t, f"Cluster {t}") for t in topics]

    # Topic words with scores
    topic_words_list = []
    for t in topics:
        indices_for_cluster = [i for i, tt in enumerate(topics) if tt == t]
        if indices_for_cluster:
            cluster_tfidf = tfidf_matrix[indices_for_cluster].mean(axis=0).A1
            top_idx = cluster_tfidf.argsort()[-10:][::-1]
            words = [f"{feature_names[i]}({cluster_tfidf[i]:.3f})" for i in top_idx]
            topic_words_list.append("; ".join(words))
        else:
            topic_words_list.append("")

    return topics, topic_labels, reduced_2d, topic_words_list


# ---------------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------------
def cluster_papers(
    config: dict | None = None,
    papers_path: Path | None = None,
    embeddings_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run clustering pipeline. Returns DataFrame with cluster columns added."""
    if config is None:
        config = load_config()

    if papers_path is None:
        papers_path = DATA_DIR / "papers.csv"
    if embeddings_path is None:
        embeddings_path = DATA_DIR / "embeddings.npy"
    if output_path is None:
        output_path = DATA_DIR / "papers_clustered.csv"

    # Load data
    df = load_papers_csv(papers_path)
    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded {len(df)} papers, embeddings shape: {embeddings.shape}")

    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(df)} papers but {embeddings.shape[0]} embeddings. "
            f"Re-run embed.py."
        )

    # Prepare documents for topic extraction
    docs = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        abstract = str(row.get("abstract", ""))
        docs.append(f"{title}. {abstract}" if abstract else title)

    # Try BERTopic, fall back to KMeans
    method = config.get("clustering", {}).get("method", "bertopic")

    if method == "bertopic":
        try:
            topics, labels, reduced_2d, topic_words = cluster_bertopic(
                docs, embeddings, config, df=df
            )
        except ImportError as e:
            logger.warning(f"BERTopic not available ({e}), falling back to KMeans")
            topics, labels, reduced_2d, topic_words = cluster_kmeans(
                docs, embeddings, config
            )
    elif method == "kmeans":
        topics, labels, reduced_2d, topic_words = cluster_kmeans(
            docs, embeddings, config
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Add cluster columns to DataFrame
    df["x"] = reduced_2d[:, 0]
    df["y"] = reduced_2d[:, 1]
    df["cluster_id"] = topics
    df["cluster_label"] = labels
    df["topic_words"] = topic_words

    # Save
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved clustered data to {output_path}")

    # Print quality summary
    print_cluster_summary(df)

    return df


# ---------------------------------------------------------------------------
# In-memory clustering for search results (no file I/O)
# ---------------------------------------------------------------------------
def cluster_from_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_clusters: int | None = None,
) -> pd.DataFrame:
    """Cluster papers in memory from pre-computed embeddings.

    Uses KMeans (fast, no BERTopic dependency needed for search results).
    Auto-selects n_clusters based on dataset size if not specified.
    Returns DataFrame with x, y, cluster_id, cluster_label, topic_words added.
    """
    from sklearn.cluster import KMeans

    n = len(df)
    if n < 5:
        # Too few papers for meaningful clustering
        df = df.copy()
        df["x"] = 0.0
        df["y"] = 0.0
        df["cluster_id"] = 0
        df["cluster_label"] = "All papers"
        df["topic_words"] = ""
        return df

    if n_clusters is None:
        # Heuristic: sqrt(n/2), clamped to [3, 200]
        n_clusters = max(3, min(200, int((n / 2) ** 0.5)))

    logger.info(f"Clustering {n} papers into ~{n_clusters} clusters...")

    # UMAP → 2D
    try:
        from umap import UMAP
        umap_model = UMAP(
            n_neighbors=min(15, n - 1),
            n_components=2,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        reduced_2d = umap_model.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        reduced_2d = pca.fit_transform(embeddings)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    topics = km.fit_predict(embeddings).tolist()

    # Cluster labels: OpenAlex topic majority vote, deduplicated, TF-IDF fallback
    has_topic = "topic" in df.columns and df["topic"].notna().any()

    topic_label_map = {}
    used_labels = set()

    # Sort by cluster size (largest first gets the clean label)
    cluster_sizes = {cid: sum(1 for t in topics if t == cid) for cid in range(n_clusters)}
    sorted_cids = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)

    for cid in sorted_cids:
        indices = [i for i, t in enumerate(topics) if t == cid]
        if not indices:
            topic_label_map[cid] = f"Cluster {cid}"
            continue

        label = None
        if has_topic:
            cluster_topics = df.iloc[indices]["topic"].dropna()
            cluster_topics = cluster_topics[cluster_topics != ""]
            if len(cluster_topics) > 0:
                for candidate in cluster_topics.value_counts().index:
                    if candidate not in used_labels:
                        label = candidate
                        break

        if label is None:
            topic_label_map[cid] = f"Cluster {cid}"  # will be filled by TF-IDF
        else:
            topic_label_map[cid] = label
            used_labels.add(label)

    # TF-IDF fallback for clusters that didn't get a unique topic label
    needs_tfidf = [cid for cid, lbl in topic_label_map.items()
                   if lbl.startswith("Cluster ")]
    if needs_tfidf:
        docs = []
        for _, row in df.iterrows():
            title = str(row.get("title", ""))
            abstract = str(row.get("abstract", ""))
            docs.append(f"{title}. {abstract}" if abstract else title)
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        for cid in needs_tfidf:
            indices = [i for i, t in enumerate(topics) if t == cid]
            if indices:
                cluster_tfidf = tfidf_matrix[indices].mean(axis=0).A1
                top_idx = cluster_tfidf.argsort()[-3:][::-1]
                label = ", ".join(feature_names[i] for i in top_idx)
                topic_label_map[cid] = label
                used_labels.add(label)

    topic_labels = [topic_label_map.get(t, f"Cluster {t}") for t in topics]

    # Topic words (TF-IDF keywords for detail panel, always generated)
    docs = docs if needs_tfidf else []
    if not docs:
        for _, row in df.iterrows():
            title = str(row.get("title", ""))
            abstract = str(row.get("abstract", ""))
            docs.append(f"{title}. {abstract}" if abstract else title)
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
    topic_words_list = []
    for t in topics:
        indices_for_cluster = [i for i, tt in enumerate(topics) if tt == t]
        if indices_for_cluster:
            cluster_tfidf = tfidf_matrix[indices_for_cluster].mean(axis=0).A1
            top_idx = cluster_tfidf.argsort()[-10:][::-1]
            words = [f"{feature_names[i]}({cluster_tfidf[i]:.3f})" for i in top_idx]
            topic_words_list.append("; ".join(words))
        else:
            topic_words_list.append("")

    df = df.copy()
    df["x"] = reduced_2d[:, 0]
    df["y"] = reduced_2d[:, 1]
    df["cluster_id"] = topics
    df["cluster_label"] = topic_labels
    df["topic_words"] = topic_words_list

    print_cluster_summary(df)
    return df


# ---------------------------------------------------------------------------
# Author clustering
# ---------------------------------------------------------------------------
def cluster_authors(
    author_df: pd.DataFrame,
    author_embeddings: np.ndarray,
    n_clusters: int | None = None,
) -> pd.DataFrame:
    """Cluster authors using their embeddings.

    Args:
        author_df: DataFrame with author_id, name, institution, paper_count, total_citations
        author_embeddings: numpy array (n_authors, dim)
        n_clusters: number of clusters (auto if None)

    Returns DataFrame with x, y, cluster_id, cluster_label added.
    """
    from sklearn.cluster import KMeans

    n = len(author_df)
    logger.info(f"Clustering {n:,} authors...")

    if n < 5:
        df = author_df.copy()
        df["x"] = 0.0
        df["y"] = 0.0
        df["cluster_id"] = 0
        df["cluster_label"] = "All authors"
        return df

    if n_clusters is None:
        n_clusters = max(5, min(50, int((n / 3) ** 0.5)))

    # UMAP → 2D
    try:
        from umap import UMAP
        umap_model = UMAP(
            n_neighbors=min(15, n - 1),
            n_components=2,
            min_dist=0.3,
            metric="cosine",
            random_state=42,
        )
        reduced_2d = umap_model.fit_transform(author_embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        reduced_2d = pca.fit_transform(author_embeddings)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    topics = km.fit_predict(author_embeddings).tolist()

    # Label clusters by most common institution or top author
    topic_label_map = {}
    for cid in range(n_clusters):
        indices = [i for i, t in enumerate(topics) if t == cid]
        if not indices:
            topic_label_map[cid] = f"Group {cid}"
            continue
        # Get the highest-cited author in this cluster as representative
        cluster_slice = author_df.iloc[indices]
        top_author = cluster_slice.nlargest(1, "total_citations")
        if not top_author.empty:
            name = top_author.iloc[0]["name"]
            inst = top_author.iloc[0]["institution"]
            label = f"{name}"
            if inst:
                label += f" ({inst[:30]})"
            topic_label_map[cid] = label
        else:
            topic_label_map[cid] = f"Group {cid}"

    topic_labels = [topic_label_map.get(t, f"Group {t}") for t in topics]

    df = author_df.copy()
    df["x"] = reduced_2d[:, 0]
    df["y"] = reduced_2d[:, 1]
    df["cluster_id"] = topics
    df["cluster_label"] = topic_labels

    logger.info(f"Author clustering done: {n_clusters} clusters")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    config = load_config()
    df = cluster_papers(config)
    print(f"\nDone! {len(df)} papers clustered → data/papers_clustered.csv")
