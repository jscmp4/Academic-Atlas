"""
Build derived query layers from Parquet (Lakehouse Lite — Step 2).

Every derived database is rebuilt from the Parquet source of truth.
If the schema changes, just re-run the relevant build command.

Usage:
    python build_derived.py analytics    # DuckDB for aggregation/analytics
    python build_derived.py search       # SQLite FTS5 for full-text search
    python build_derived.py authors      # Derive deduplicated authors.parquet
    python build_derived.py all          # Build everything
    python build_derived.py worldmap     # Sample top-cited → embed → cluster → worldmap CSV
"""

import sqlite3
import sys
import time
from pathlib import Path


PARQUET_DIR = Path("N:/academic-data/parquet")
DERIVED_DIR = Path("N:/academic-data/derived")

# Use forward slashes for DuckDB glob patterns (works on Windows too)
def _glob(pattern: str) -> str:
    return str(PARQUET_DIR / pattern).replace("\\", "/")


# ---------------------------------------------------------------------------
# (A) DuckDB Analytics
# ---------------------------------------------------------------------------

def build_analytics():
    """Create DuckDB database with full paper data + indexes for analytics."""
    import duckdb

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DERIVED_DIR / "analytics.duckdb"

    if db_path.exists():
        print(f"Removing old {db_path.name}...")
        db_path.unlink()

    works_glob = _glob("works_*.parquet")
    auth_glob = _glob("authorships_*.parquet")

    print(f"{'=' * 60}")
    print(f"Building DuckDB analytics: {db_path}")
    print(f"{'=' * 60}")

    conn = duckdb.connect(str(db_path))
    t0 = time.time()

    print("  Creating papers table from Parquet...")
    conn.execute(f"CREATE TABLE papers AS SELECT * FROM read_parquet('{works_glob}')")
    papers_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"  → {papers_count:,} papers loaded ({time.time() - t0:.0f}s)")

    print("  Creating indexes...")
    conn.execute("CREATE INDEX idx_year ON papers(publication_year)")
    conn.execute("CREATE INDEX idx_field ON papers(field)")
    conn.execute("CREATE INDEX idx_subfield ON papers(subfield)")
    conn.execute("CREATE INDEX idx_source ON papers(source_name)")
    conn.execute("CREATE INDEX idx_citations ON papers(cited_by_count)")
    conn.execute("CREATE INDEX idx_type ON papers(type)")
    conn.execute("CREATE INDEX idx_tier ON papers(journal_tier)")
    conn.execute("CREATE INDEX idx_language ON papers(language)")
    print(f"  → Indexes created ({time.time() - t0:.0f}s)")

    # Authorships as a view over Parquet (no need to duplicate into DuckDB)
    print("  Creating authorships view (reads Parquet on demand)...")
    conn.execute(f"CREATE VIEW authorships AS SELECT * FROM read_parquet('{auth_glob}')")

    # Summary stats
    print(f"\n  Stats:")
    print(f"  Papers: {papers_count:,}")

    for row in conn.execute("""
        SELECT field, COUNT(*) as n FROM papers
        WHERE field != '' GROUP BY field ORDER BY n DESC LIMIT 10
    """).fetchall():
        print(f"    {row[0]}: {row[1]:,}")

    elapsed = time.time() - t0
    conn.close()

    size_gb = db_path.stat().st_size / 1024 ** 3
    print(f"\n  Done in {elapsed:.0f}s — {db_path} ({size_gb:.1f} GB)")


# ---------------------------------------------------------------------------
# (B) SQLite Full-Text Search
# ---------------------------------------------------------------------------

def build_search():
    """Create SQLite database with FTS5 index for paper search.

    Two-phase approach:
      Phase 1: DuckDB bulk-writes Parquet → SQLite (fast columnar scan, no row-by-row INSERT)
      Phase 2: Python adds FTS5 index + supporting indexes on the populated table
    """
    import duckdb

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DERIVED_DIR / "search.db"
    db_path_posix = str(db_path).replace("\\", "/")

    if db_path.exists():
        print(f"Removing old {db_path.name}...")
        db_path.unlink()

    works_glob = _glob("works_*.parquet")

    print(f"{'=' * 60}")
    print(f"Building SQLite FTS5 search: {db_path}")
    print(f"{'=' * 60}")

    # Phase 1: DuckDB bulk export to SQLite
    print("  Phase 1: DuckDB → SQLite bulk export (articles only)...")
    t0 = time.time()

    conn = duckdb.connect()
    conn.execute("INSTALL sqlite")
    conn.execute("LOAD sqlite")
    conn.execute("SET enable_progress_bar = true")
    conn.execute("SET enable_progress_bar_print = true")
    conn.execute(f"ATTACH '{db_path_posix}' AS search_db (TYPE SQLITE)")

    conn.execute(f"""
        CREATE TABLE search_db.papers AS
        SELECT
            openalex_id,
            title,
            abstract,
            publication_year AS year,
            cited_by_count AS citation_count,
            source_name AS journal,
            doi,
            subfield,
            field,
            journal_tier
        FROM read_parquet('{works_glob}')
        WHERE type = 'article'
    """)

    total = conn.execute("SELECT COUNT(*) FROM search_db.papers").fetchone()[0]
    conn.close()
    print(f"  → {total:,} articles loaded ({time.time() - t0:.0f}s)")

    # Phase 2: Add FTS5 + indexes via Python sqlite3 (DuckDB can't create FTS5)
    from tqdm import tqdm

    print("  Phase 2: Building FTS5 index + supporting indexes...")
    t1 = time.time()

    sconn = sqlite3.connect(str(db_path))
    sconn.execute("PRAGMA journal_mode=WAL")
    sconn.execute("PRAGMA synchronous=OFF")
    sconn.execute("PRAGMA cache_size=-2000000")

    print("  Creating FTS5 virtual table...")
    sconn.execute("""
        CREATE VIRTUAL TABLE papers_fts USING fts5(
            title, abstract,
            content=papers, content_rowid=rowid
        )
    """)

    # Batch-insert into FTS5 with progress bar (instead of one giant 'rebuild')
    FTS_BATCH = 500_000
    max_rowid = sconn.execute("SELECT MAX(rowid) FROM papers").fetchone()[0] or 0

    print(f"  Indexing {total:,} papers into FTS5...")
    with tqdm(total=total, unit=" papers", desc="  FTS5") as pbar:
        lo = 1
        while lo <= max_rowid:
            hi = lo + FTS_BATCH - 1
            sconn.execute(f"""
                INSERT INTO papers_fts(rowid, title, abstract)
                SELECT rowid, title, abstract FROM papers
                WHERE rowid BETWEEN {lo} AND {hi}
            """)
            sconn.commit()
            inserted = sconn.execute(
                f"SELECT COUNT(*) FROM papers WHERE rowid BETWEEN {lo} AND {hi}"
            ).fetchone()[0]
            pbar.update(inserted)
            lo = hi + 1

    print(f"  → FTS5 index built in {time.time() - t1:.0f}s")

    print("  Creating supporting indexes...")
    idx_names = [
        ("idx_year", "year"), ("idx_citations", "citation_count"),
        ("idx_tier", "journal_tier"), ("idx_subfield", "subfield"),
        ("idx_field", "field"),
    ]
    for idx_name, col in tqdm(idx_names, desc="  Indexes", unit=" idx"):
        sconn.execute(f"CREATE INDEX {idx_name} ON papers({col})")
        sconn.commit()
    print(f"  → Indexes created ({time.time() - t1:.0f}s)")

    # Verify
    test = sconn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH 'machine learning'"
    ).fetchone()[0]
    print(f"  FTS5 verify: 'machine learning' → {test:,} matches")

    elapsed = time.time() - t0
    sconn.close()

    size_gb = db_path.stat().st_size / 1024 ** 3
    print(f"\n  Done in {elapsed:.0f}s — {db_path} ({size_gb:.1f} GB)")


# ---------------------------------------------------------------------------
# (C) Derive authors.parquet
# ---------------------------------------------------------------------------

def build_authors():
    """Derive deduplicated authors.parquet from authorships Parquet files.

    Two-pass approach to avoid OOM on 1B+ rows:
      Pass 1: GROUP BY per year file → temp Parquet partials
      Pass 2: GROUP BY across partials → final authors.parquet
    """
    import duckdb
    import shutil

    out_path = _glob("authors.parquet")
    tmp_dir = DERIVED_DIR / "tmp_authors"

    print(f"{'=' * 60}")
    print(f"Deriving authors.parquet from authorships")
    print(f"{'=' * 60}")

    # Clean up any previous partial run
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    auth_files = sorted(PARQUET_DIR.glob("authorships_*.parquet"))
    print(f"  Pass 1: Deduplicating {len(auth_files)} year files...")
    t0 = time.time()

    for i, af in enumerate(auth_files):
        af_posix = str(af).replace("\\", "/")
        out_partial = str(tmp_dir / af.name).replace("\\", "/")

        conn = duckdb.connect()
        conn.execute(f"""
            COPY (
                SELECT
                    author_id,
                    FIRST(author_name) AS name,
                    FIRST(orcid) FILTER (WHERE orcid != '' AND orcid IS NOT NULL) AS orcid
                FROM read_parquet('{af_posix}')
                WHERE author_id != ''
                GROUP BY author_id
            ) TO '{out_partial}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        conn.close()

        if (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{len(auth_files)}] {time.time() - t0:.0f}s")

    print(f"  Pass 1 done in {time.time() - t0:.0f}s")

    # Pass 2: merge partials (same author across years → keep one)
    print("  Pass 2: Merging across years...")
    t1 = time.time()
    partial_glob = str(tmp_dir / "authorships_*.parquet").replace("\\", "/")

    conn = duckdb.connect()
    conn.execute(f"""
        COPY (
            SELECT
                author_id,
                FIRST(name) AS name,
                FIRST(orcid) FILTER (WHERE orcid != '' AND orcid IS NOT NULL) AS orcid
            FROM read_parquet('{partial_glob}')
            GROUP BY author_id
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)

    total = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path}')").fetchone()[0]
    conn.close()

    # Clean up temp files
    shutil.rmtree(tmp_dir)

    elapsed = time.time() - t0
    print(f"  {total:,} unique authors in {elapsed:.0f}s")

    size_mb = Path(out_path.replace("/", "\\")).stat().st_size / 1024 ** 2
    print(f"  Output: {out_path} ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# (D) World Map — sample top-cited papers, embed, cluster
# ---------------------------------------------------------------------------

def build_worldmap():
    """Build a 'world map' of academia: top-cited papers from every field.

    Steps:
      1. DuckDB query: top N papers per field by citations
      2. Export to CSV
      3. Embed with sentence-transformers
      4. Cluster with BERTopic
      5. Save as data/worldmap_clustered.csv
    """
    import duckdb

    PAPERS_PER_FIELD = 2000
    db_path = DERIVED_DIR / "analytics.duckdb"
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    raw_path = data_dir / "worldmap_raw.csv"
    out_path = data_dir / "worldmap_clustered.csv"

    print(f"{'=' * 60}")
    print(f"Building World Map ({PAPERS_PER_FIELD} top-cited per field)")
    print(f"{'=' * 60}")

    # Step 1: Filter high-impact papers from Parquet (no sorting needed = fast)
    MIN_CITATIONS = 500
    print(f"  Step 1: Filtering articles with ≥{MIN_CITATIONS} citations from Parquet...")
    t0 = time.time()

    import pandas as pd
    works_glob = _glob("works_*.parquet")

    tmp_db = DERIVED_DIR / "tmp_worldmap.duckdb"
    if tmp_db.exists():
        tmp_db.unlink()
    conn = duckdb.connect(str(tmp_db))
    conn.execute("SET memory_limit = '4GB'")
    conn.execute("SET enable_progress_bar = true")
    conn.execute("SET enable_progress_bar_print = true")

    # Write filtered results to temp table first (disk-backed, not in-memory)
    conn.execute(f"""
        CREATE TABLE filtered AS
        SELECT
            openalex_id, title, abstract,
            cited_by_count AS citation_count,
            publication_year AS year,
            source_name AS journal,
            doi, field, subfield, topic, journal_tier
        FROM read_parquet('{works_glob}')
        WHERE type = 'article'
          AND cited_by_count >= {MIN_CITATIONS}
          AND field IS NOT NULL AND field != ''
          AND title IS NOT NULL AND title != ''
    """)
    total = conn.execute("SELECT COUNT(*) FROM filtered").fetchone()[0]
    print(f"  → {total:,} high-impact papers filtered ({time.time() - t0:.0f}s)")

    df = conn.execute("SELECT * FROM filtered").fetchdf()
    conn.close()
    tmp_db.unlink(missing_ok=True)
    # Also remove WAL file if exists
    wal = Path(str(tmp_db) + ".wal")
    wal.unlink(missing_ok=True)
    print(f"  → Loaded to DataFrame ({time.time() - t0:.0f}s)")

    # Square-root proportional sampling (academic standard for science maps)
    # Each field gets samples ∝ √(field_size), with a total target of ~50K
    TARGET_TOTAL = 50_000
    field_counts = df.groupby("field").size()
    sqrt_weights = field_counts.apply(lambda x: x ** 0.5)
    allocations = (sqrt_weights / sqrt_weights.sum() * TARGET_TOTAL).astype(int).clip(lower=100)

    parts = []
    for field, group in df.groupby("field"):
        n_take = min(len(group), allocations.get(field, 500))
        if len(group) > n_take:
            group = group.sample(n=n_take, random_state=42)
        parts.append(group)
        print(f"    {len(group):6,} / {field_counts[field]:6,} — {field}")
    df = pd.concat(parts, ignore_index=True)
    print(f"  → {len(df):,} papers (sqrt-proportional sampling)")

    # Drop abstract from memory after building text (saves ~60% of DataFrame RAM)
    # We'll keep it in CSV for later use but don't need it in memory during clustering
    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"  → Raw CSV saved: {raw_path}")

    # Step 2: Embed (then free model)
    import gc
    import numpy as np

    print(f"\n  Step 2: Embedding {len(df):,} papers...")
    t1 = time.time()

    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()

    from embed import generate_embeddings_from_texts
    embeddings = generate_embeddings_from_texts(texts)
    print(f"  → Embeddings: {embeddings.shape} ({time.time() - t1:.0f}s)")

    np.save(data_dir / "worldmap_embeddings.npy", embeddings)
    del texts
    gc.collect()

    # Step 3: KMeans cluster (lighter than BERTopic, ~200MB vs 6GB)
    print(f"\n  Step 3: Clustering with KMeans...")
    t2 = time.time()

    from cluster import cluster_from_embeddings
    df = cluster_from_embeddings(df, embeddings)
    print(f"  → {df['cluster_id'].nunique()} clusters ({time.time() - t2:.0f}s)")

    del embeddings
    gc.collect()

    # Save final
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Done in {time.time() - t0:.0f}s — {out_path}")
    print(f"  {len(df):,} papers, {df['cluster_id'].nunique()} clusters, "
          f"{df['field'].nunique()} fields")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python build_derived.py analytics    # DuckDB analytics DB")
        print("  python build_derived.py search       # SQLite FTS5 search DB")
        print("  python build_derived.py authors      # authors.parquet (deduped)")
        print("  python build_derived.py worldmap     # Top-cited sample → embed → cluster")
        print("  python build_derived.py all          # Build everything (except worldmap)")
        return

    action = sys.argv[1].lower()

    if action in ("analytics", "all"):
        build_analytics()
    if action in ("authors", "all"):
        build_authors()
    if action in ("search", "all"):
        build_search()
    if action == "worldmap":
        build_worldmap()

    if action not in ("analytics", "search", "authors", "all", "worldmap"):
        print(f"Unknown action: {action}")
        print("Valid actions: analytics, search, authors, worldmap, all")


if __name__ == "__main__":
    main()
