"""
Stage 3: EMBED — Generate paper and author embeddings.

Paper embeddings: sentence-transformers with automatic model fallback.
Author embeddings: citation-weighted average of their paper embeddings.

Fallback chain: specter2 → mpnet → minilm
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    get_logger,
    load_config,
    load_papers_csv,
    setup_logging,
    DATA_DIR,
    PAPER_COLUMNS_REQUIRED,
)

logger = get_logger("embed")

# Model registry
MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "type": "sentence-transformers",
        "dim": 384,
        "sep": ". ",  # title. abstract
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "type": "sentence-transformers",
        "dim": 768,
        "sep": ". ",
    },
    "specter2": {
        "name": "allenai/specter2",
        "type": "transformers",
        "dim": 768,
        "sep": " [SEP] ",  # SPECTER2 expects title [SEP] abstract
    },
}

FALLBACK_ORDER = ["specter2", "mpnet", "minilm"]


# ---------------------------------------------------------------------------
# Embedding functions per model type
# ---------------------------------------------------------------------------
def _embed_sentence_transformers(texts: list[str], model_name: str) -> np.ndarray:
    """Embed using sentence-transformers library."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading sentence-transformers model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return np.array(embeddings)


def _embed_specter2(texts: list[str]) -> np.ndarray:
    """Embed using SPECTER2 (transformers + adapters)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading SPECTER2 model (allenai/specter2)...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2")
    model = AutoModel.from_pretrained("allenai/specter2")
    model.eval()

    logger.info(f"Encoding {len(texts)} documents with SPECTER2...")
    all_embeddings = []
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(embeddings)

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Progress: {i + len(batch)}/{len(texts)}")

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Main embedding function with fallback
# ---------------------------------------------------------------------------
def generate_embeddings(
    config: dict | None = None,
    papers_path: Path | None = None,
    force: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Generate embeddings for papers. Returns (embeddings_array, paper_ids).

    Implements automatic fallback: requested model → mpnet → minilm.
    Checks metadata for skip conditions (same model, same papers).
    """
    if config is None:
        config = load_config()

    if papers_path is None:
        papers_path = DATA_DIR / "papers.csv"

    embeddings_path = DATA_DIR / "embeddings.npy"
    meta_path = DATA_DIR / "embeddings_meta.json"

    # Load papers
    df = load_papers_csv(papers_path, columns=["openalex_id", "title", "abstract"])
    paper_ids = df["openalex_id"].tolist()
    logger.info(f"Loaded {len(paper_ids)} papers for embedding")

    # Check if we can skip
    requested_model = config.get("embedding", {}).get("model", "minilm")

    if not force and embeddings_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (
            meta.get("model") == requested_model
            and meta.get("paper_ids") == paper_ids
        ):
            logger.info(
                f"Embeddings already exist for {requested_model} with same papers. Skipping."
            )
            return np.load(embeddings_path), paper_ids

    # Handle empty dataset
    if len(paper_ids) == 0:
        logger.warning("No papers to embed!")
        dim = MODELS.get(requested_model, MODELS["minilm"])["dim"]
        embeddings = np.empty((0, dim))
        np.save(embeddings_path, embeddings)
        meta = {"model": requested_model, "model_name": MODELS.get(requested_model, MODELS["minilm"])["name"],
                "dim": dim, "n_papers": 0, "paper_ids": []}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return embeddings, paper_ids

    # Prepare texts
    model_info = MODELS.get(requested_model, MODELS["minilm"])
    sep = model_info["sep"]
    texts = []
    for _, row in df.iterrows():
        title = str(row.get("title", "")) or ""
        abstract = str(row.get("abstract", "")) or ""
        if abstract:
            texts.append(f"{title}{sep}{abstract}")
        else:
            texts.append(title)

    # Try embedding with fallback chain
    embeddings = None
    actual_model = requested_model

    # Build fallback order starting from requested model
    fallback = [requested_model]
    for m in FALLBACK_ORDER:
        if m not in fallback:
            fallback.append(m)

    for model_key in fallback:
        model_info = MODELS.get(model_key, MODELS["minilm"])
        try:
            if model_info["type"] == "transformers":
                embeddings = _embed_specter2(texts)
            else:
                embeddings = _embed_sentence_transformers(texts, model_info["name"])
            actual_model = model_key
            break
        except Exception as e:
            logger.warning(f"Failed to load {model_key}: {e}")
            if model_key != fallback[-1]:
                logger.info(f"Falling back to next model...")
            else:
                raise RuntimeError("All embedding models failed!") from e

    if actual_model != requested_model:
        logger.warning(
            f"Used fallback model '{actual_model}' instead of '{requested_model}'"
        )

    logger.info(f"Embeddings shape: {embeddings.shape} (model: {actual_model})")

    # Save embeddings
    np.save(embeddings_path, embeddings)

    # Save metadata (including paper IDs for future incremental support)
    meta = {
        "model": actual_model,
        "model_name": MODELS[actual_model]["name"],
        "dim": embeddings.shape[1],
        "n_papers": len(paper_ids),
        "paper_ids": paper_ids,  # For v2 incremental embedding
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved embeddings to {embeddings_path}")
    return embeddings, paper_ids


# ---------------------------------------------------------------------------
# Author embeddings: citation-weighted average of paper embeddings
# ---------------------------------------------------------------------------
def generate_author_embeddings(
    min_papers: int = 3,
    model_key: str = "minilm",
    force: bool = False,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Generate author embeddings from their papers' embeddings.

    For each author with >= min_papers papers:
    1. Get all their papers from paper_authors table
    2. Embed each paper (title + abstract)
    3. Compute citation-weighted average embedding

    Returns (author_embeddings, author_ids, author_df).
    author_df has columns: author_id, name, institution, paper_count, total_citations
    """
    db_path = DATA_DIR / "openalex.db"
    embeddings_path = DATA_DIR / "author_embeddings.npy"
    meta_path = DATA_DIR / "author_embeddings_meta.json"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Get active authors
    logger.info(f"Loading authors with >= {min_papers} papers...")
    authors = conn.execute(
        "SELECT author_id, name, institution, paper_count, total_citations "
        "FROM authors WHERE paper_count >= ? ORDER BY total_citations DESC",
        (min_papers,)
    ).fetchall()

    author_ids = [a["author_id"] for a in authors]
    logger.info(f"Found {len(author_ids):,} active authors")

    # Check skip
    if not force and embeddings_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if (meta.get("min_papers") == min_papers
                and meta.get("n_authors") == len(author_ids)
                and meta.get("model") == model_key):
            logger.info("Author embeddings already up to date. Skipping.")
            author_df = pd.DataFrame([dict(a) for a in authors])
            return np.load(embeddings_path), author_ids, author_df

    if not author_ids:
        logger.warning("No active authors found!")
        author_df = pd.DataFrame(columns=["author_id", "name", "institution", "paper_count", "total_citations"])
        return np.empty((0, MODELS[model_key]["dim"])), [], author_df

    # Step 1: Get all unique papers for these authors
    logger.info("Loading paper-author relationships...")
    # Build a mapping: author_id -> [(paper_id, citation_count), ...]
    author_papers = {}
    for aid in author_ids:
        author_papers[aid] = []

    # Batch query for efficiency
    batch_size = 500
    for i in range(0, len(author_ids), batch_size):
        batch_ids = author_ids[i:i+batch_size]
        placeholders = ",".join("?" * len(batch_ids))
        rows = conn.execute(f"""
            SELECT pa.author_id, pa.paper_id, p.title, p.abstract, p.citation_count
            FROM paper_authors pa
            JOIN papers p ON p.openalex_id = pa.paper_id
            WHERE pa.author_id IN ({placeholders})
        """, batch_ids).fetchall()
        for row in rows:
            author_papers[row["author_id"]].append({
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "citation_count": row["citation_count"],
            })

        if (i // batch_size) % 10 == 0 and i > 0:
            logger.info(f"  Loaded papers for {i+len(batch_ids)}/{len(author_ids)} authors...")

    conn.close()

    # Step 2: Collect all unique paper texts for embedding
    all_paper_ids = set()
    for papers in author_papers.values():
        for p in papers:
            all_paper_ids.add(p["paper_id"])

    logger.info(f"Embedding {len(all_paper_ids):,} unique papers...")

    # Build paper_id -> text mapping
    paper_id_list = []
    paper_texts = []
    for papers in author_papers.values():
        for p in papers:
            if p["paper_id"] not in paper_id_list:
                paper_id_list.append(p["paper_id"])
                title = p["title"] or ""
                abstract = p["abstract"] or ""
                paper_texts.append(f"{title}. {abstract}" if abstract else title)

    # Deduplicate
    seen = set()
    unique_ids = []
    unique_texts = []
    for pid, text in zip(paper_id_list, paper_texts):
        if pid not in seen:
            seen.add(pid)
            unique_ids.append(pid)
            unique_texts.append(text)

    # Embed all unique papers
    paper_embeddings = generate_embeddings_from_texts(unique_texts, model_key=model_key)
    paper_emb_map = {pid: paper_embeddings[i] for i, pid in enumerate(unique_ids)}

    # Step 3: Compute author embeddings (citation-weighted average)
    logger.info("Computing author embeddings (citation-weighted average)...")
    dim = paper_embeddings.shape[1]
    author_embeddings = np.zeros((len(author_ids), dim), dtype=np.float32)

    for idx, aid in enumerate(author_ids):
        papers = author_papers[aid]
        if not papers:
            continue

        weights = []
        embs = []
        for p in papers:
            if p["paper_id"] in paper_emb_map:
                # Weight: log(1 + citations) to avoid extreme dominance
                w = np.log1p(p["citation_count"])
                weights.append(max(w, 0.1))  # Minimum weight so uncited papers still count
                embs.append(paper_emb_map[p["paper_id"]])

        if embs:
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum()
            author_embeddings[idx] = np.average(embs, axis=0, weights=weights)

    # Normalize
    norms = np.linalg.norm(author_embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)
    author_embeddings = author_embeddings / norms

    # Save
    np.save(embeddings_path, author_embeddings)
    meta = {
        "model": model_key,
        "min_papers": min_papers,
        "n_authors": len(author_ids),
        "dim": dim,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"Author embeddings saved: {author_embeddings.shape}")

    author_df = pd.DataFrame([dict(a) for a in authors])
    return author_embeddings, author_ids, author_df


# ---------------------------------------------------------------------------
# In-memory embedding for search results (no file I/O)
# ---------------------------------------------------------------------------
def generate_embeddings_from_texts(
    texts: list[str],
    model_key: str = "minilm",
) -> np.ndarray:
    """Embed a list of texts directly (for search results visualization).

    Uses sentence-transformers by default. No file I/O — returns array in memory.
    """
    model_info = MODELS.get(model_key, MODELS["minilm"])

    if model_info["type"] == "transformers":
        return _embed_specter2(texts)
    else:
        return _embed_sentence_transformers(texts, model_info["name"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    config = load_config()
    embeddings, ids = generate_embeddings(config)
    print(f"\nDone! Embeddings: {embeddings.shape}")
    print(f"Model: {MODELS.get(config.get('embedding', {}).get('model', 'minilm'), {}).get('name', 'unknown')}")
