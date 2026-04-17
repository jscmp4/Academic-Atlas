"""
Build SQLite database from OpenAlex snapshot.

Streams through 583GB of .gz files, extracts IS-related papers,
and inserts directly into SQLite. Memory usage stays under 200MB.

Usage:
    python build_db.py                # Build full IS database
    python build_db.py --query        # Interactive query mode
    python build_db.py --export-top   # Export Tier1+2 subset to CSV
    python build_db.py --export-broad # Export Tier1+2+3 subset to CSV
"""

import gzip
import json
import sqlite3
import sys
import time
from pathlib import Path

from utils import PROJECT_ROOT, DATA_DIR, setup_logging, get_logger, reconstruct_abstract
from download_openalex import IS_SUBFIELDS, get_journal_tier, WORKS_DIR

logger = get_logger("build_db")

DB_PATH = DATA_DIR / "openalex.db"


def create_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            openalex_id TEXT PRIMARY KEY,
            title TEXT,
            abstract TEXT,
            citation_count INTEGER,
            year INTEGER,
            journal TEXT,
            doi TEXT,
            authors TEXT,
            subfield TEXT,
            field TEXT,
            topic TEXT,
            journal_tier TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON papers(journal_tier)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_citations ON papers(citation_count)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_journal ON papers(journal)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_subfield ON papers(subfield)")
    conn.commit()


def build_database(year_min: int = 2000, year_max: int = 2026):
    """Stream snapshot into SQLite. Low memory, resumable."""
    target_subfields = {s.lower() for s in IS_SUBFIELDS}

    conn = sqlite3.connect(str(DB_PATH))
    create_tables(conn)

    # Check how many we already have (for resume)
    existing = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    if existing > 0:
        print(f"Database already has {existing:,} papers.")
        resp = input("Drop and rebuild? [y/N]: ").strip().lower()
        if resp != "y":
            print("Keeping existing database. Use --query or --export-* options.")
            conn.close()
            return
        conn.execute("DELETE FROM papers")
        conn.commit()

    gz_files = sorted(WORKS_DIR.rglob("*.gz"))
    print(f"Building SQLite from {len(gz_files)} snapshot files...")
    print(f"Year range: {year_min}-{year_max}")
    print(f"Subfields: {len(IS_SUBFIELDS)}")
    print(f"Output: {DB_PATH}")
    print()

    t0 = time.time()
    files_done = 0
    total_scanned = 0
    papers_inserted = 0
    batch = []
    BATCH_SIZE = 5000

    for gz_file in gz_files:
        files_done += 1
        if files_done % 50 == 0:
            elapsed = int(time.time() - t0)
            print(f"  {files_done}/{len(gz_files)} files | "
                  f"{papers_inserted:,} papers | {total_scanned:,} scanned | "
                  f"{elapsed}s", flush=True)

        try:
            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line in f:
                    total_scanned += 1
                    try:
                        work = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if work.get("type") != "article":
                        continue

                    year = work.get("publication_year")
                    if not year or year < year_min or year > year_max:
                        continue

                    # Check subfield
                    pt = work.get("primary_topic") or {}
                    pt_subfield = (pt.get("subfield") or {}).get("display_name", "")
                    matched = pt_subfield.lower() in target_subfields
                    if not matched:
                        for topic in work.get("topics", []):
                            sf = (topic.get("subfield") or {}).get("display_name", "")
                            if sf.lower() in target_subfields:
                                matched = True
                                pt_subfield = sf  # Use the matched one
                                break

                    if not matched:
                        continue

                    inv_idx = work.get("abstract_inverted_index")
                    if not inv_idx:
                        continue
                    abstract = reconstruct_abstract(inv_idx)
                    if not abstract or len(abstract) < 50:
                        continue

                    primary_loc = work.get("primary_location") or {}
                    source = primary_loc.get("source") or {}
                    journal = source.get("display_name", "")

                    authorships = work.get("authorships", [])
                    authors = "; ".join(
                        a.get("author", {}).get("display_name", "")
                        for a in authorships[:10]
                    )

                    batch.append((
                        work.get("id", ""),
                        work.get("title", ""),
                        abstract,
                        work.get("cited_by_count", 0),
                        year,
                        journal,
                        work.get("doi", ""),
                        authors,
                        pt_subfield,
                        (pt.get("field") or {}).get("display_name", ""),
                        pt.get("display_name", ""),
                        get_journal_tier(journal),
                    ))

                    if len(batch) >= BATCH_SIZE:
                        conn.executemany(
                            "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            batch
                        )
                        conn.commit()
                        papers_inserted += len(batch)
                        batch = []

        except Exception:
            pass

    # Final batch
    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            batch
        )
        conn.commit()
        papers_inserted += len(batch)

    elapsed = int(time.time() - t0)
    print(f"\nDone! {papers_inserted:,} papers inserted in {elapsed}s")
    print(f"Database: {DB_PATH} ({DB_PATH.stat().st_size / 1024 / 1024:.0f} MB)")

    # Stats
    print_stats(conn)
    conn.close()


def print_stats(conn: sqlite3.Connection):
    """Print database statistics."""
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"\n{'='*60}")
    print(f"Database Stats: {total:,} papers")
    print(f"{'='*60}")

    print("\nBy journal tier:")
    for row in conn.execute(
        "SELECT journal_tier, COUNT(*) as n FROM papers GROUP BY journal_tier ORDER BY n DESC"
    ):
        print(f"  {row[0]}: {row[1]:,}")

    print("\nBy subfield (top 15):")
    for row in conn.execute(
        "SELECT subfield, COUNT(*) as n FROM papers GROUP BY subfield ORDER BY n DESC LIMIT 15"
    ):
        print(f"  {row[0]}: {row[1]:,}")

    print("\nTop 20 journals:")
    for row in conn.execute(
        "SELECT journal, journal_tier, COUNT(*) as n FROM papers "
        "GROUP BY journal ORDER BY n DESC LIMIT 20"
    ):
        print(f"  [{row[1]}] {row[0]}: {row[2]:,}")

    print("\nYear range:")
    row = conn.execute("SELECT MIN(year), MAX(year) FROM papers").fetchone()
    print(f"  {row[0]} - {row[1]}")


def export_subset(tier_filter: list[str], output_name: str, citation_min: int = 0):
    """Export a subset from SQLite to CSV for the pipeline."""
    import pandas as pd

    conn = sqlite3.connect(str(DB_PATH))
    placeholders = ",".join("?" * len(tier_filter))
    query = f"""
        SELECT * FROM papers
        WHERE journal_tier IN ({placeholders})
        AND citation_count > ?
        ORDER BY citation_count DESC
    """
    df = pd.read_sql_query(query, conn, params=tier_filter + [citation_min])
    conn.close()

    # Add 'streams' column for compatibility
    df["streams"] = df["subfield"]

    out_path = DATA_DIR / output_name
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Exported {len(df):,} papers to {out_path}")
    return df


def interactive_query():
    """Interactive SQL query mode."""
    conn = sqlite3.connect(str(DB_PATH))
    print_stats(conn)
    print("\nInteractive SQL mode. Type queries or 'quit' to exit.")
    print("Example: SELECT journal, COUNT(*) FROM papers WHERE journal_tier='Tier1-FT/UTD' GROUP BY journal")
    print()

    while True:
        try:
            query = input("SQL> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        try:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            if rows:
                # Print column headers
                cols = [d[0] for d in cursor.description]
                print("  |  ".join(cols))
                print("-" * 60)
                for row in rows[:50]:
                    print("  |  ".join(str(v)[:40] for v in row))
                if len(rows) > 50:
                    print(f"  ... ({len(rows)} total rows)")
            else:
                print("  (no results)")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    conn.close()


def create_author_tables(conn: sqlite3.Connection):
    """Create authors and paper_authors tables."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS authors (
            author_id TEXT PRIMARY KEY,
            name TEXT,
            institution TEXT,
            paper_count INTEGER DEFAULT 0,
            total_citations INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id TEXT,
            author_id TEXT,
            position INTEGER,
            PRIMARY KEY (paper_id, author_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pa_author ON paper_authors(author_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pa_paper ON paper_authors(paper_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_author_name ON authors(name)")
    conn.commit()


def build_authors():
    """Re-scan snapshot to extract structured author data for papers in our DB.

    Creates authors table (author_id, name, institution, paper_count, total_citations)
    and paper_authors junction table (paper_id, author_id, position).
    """
    conn = sqlite3.connect(str(DB_PATH))

    # Load set of paper IDs we care about
    print("Loading paper IDs from database...")
    paper_ids = set()
    cursor = conn.execute("SELECT openalex_id FROM papers")
    for row in cursor:
        paper_ids.add(row[0])
    print(f"  {len(paper_ids):,} papers to match")

    # Create tables (drop old data if exists)
    conn.execute("DROP TABLE IF EXISTS paper_authors")
    conn.execute("DROP TABLE IF EXISTS authors")
    create_author_tables(conn)

    gz_files = sorted(WORKS_DIR.rglob("*.gz"))
    print(f"Scanning {len(gz_files)} snapshot files for author data...")

    t0 = time.time()
    files_done = 0
    total_scanned = 0
    papers_matched = 0
    author_rows = {}  # author_id -> (name, institution)
    pa_batch = []     # (paper_id, author_id, position)
    BATCH_SIZE = 10000

    for gz_file in gz_files:
        files_done += 1
        if files_done % 50 == 0:
            elapsed = int(time.time() - t0)
            print(f"  {files_done}/{len(gz_files)} files | "
                  f"{papers_matched:,} matched | {len(author_rows):,} authors | "
                  f"{elapsed}s", flush=True)

        try:
            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line in f:
                    total_scanned += 1
                    try:
                        work = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    work_id = work.get("id", "")
                    if work_id not in paper_ids:
                        continue

                    papers_matched += 1
                    authorships = work.get("authorships", [])

                    for pos, authorship in enumerate(authorships):
                        author = authorship.get("author") or {}
                        author_id = author.get("id", "")
                        if not author_id:
                            continue

                        name = author.get("display_name", "")

                        # Get institution (first/most recent)
                        institution = ""
                        institutions = authorship.get("institutions", [])
                        if institutions:
                            institution = institutions[0].get("display_name", "")

                        # Track author info (keep latest institution seen)
                        if author_id not in author_rows or (institution and not author_rows[author_id][1]):
                            author_rows[author_id] = (name, institution)

                        pa_batch.append((work_id, author_id, pos))

                    if len(pa_batch) >= BATCH_SIZE:
                        conn.executemany(
                            "INSERT OR IGNORE INTO paper_authors VALUES (?,?,?)",
                            pa_batch
                        )
                        conn.commit()
                        pa_batch = []

        except Exception:
            pass

    # Final batch
    if pa_batch:
        conn.executemany(
            "INSERT OR IGNORE INTO paper_authors VALUES (?,?,?)",
            pa_batch
        )
        conn.commit()

    # Insert all authors
    print(f"\nInserting {len(author_rows):,} authors...")
    author_batch = [(aid, name, inst, 0, 0) for aid, (name, inst) in author_rows.items()]
    for i in range(0, len(author_batch), BATCH_SIZE):
        conn.executemany(
            "INSERT OR IGNORE INTO authors VALUES (?,?,?,?,?)",
            author_batch[i:i+BATCH_SIZE]
        )
    conn.commit()

    # Update paper_count and total_citations from actual data
    print("Computing author paper counts and citations...")
    conn.execute("""
        UPDATE authors SET
            paper_count = (
                SELECT COUNT(*) FROM paper_authors pa WHERE pa.author_id = authors.author_id
            ),
            total_citations = (
                SELECT COALESCE(SUM(p.citation_count), 0)
                FROM paper_authors pa
                JOIN papers p ON p.openalex_id = pa.paper_id
                WHERE pa.author_id = authors.author_id
            )
    """)
    conn.commit()

    elapsed = int(time.time() - t0)
    print(f"\nDone in {elapsed}s!")
    print(f"  Papers matched: {papers_matched:,}")
    print(f"  Authors: {len(author_rows):,}")

    # Stats
    pa_count = conn.execute("SELECT COUNT(*) FROM paper_authors").fetchone()[0]
    print(f"  Paper-author links: {pa_count:,}")

    active = conn.execute("SELECT COUNT(*) FROM authors WHERE paper_count >= 3").fetchone()[0]
    print(f"  Active authors (>=3 papers): {active:,}")

    top = conn.execute(
        "SELECT name, paper_count, total_citations FROM authors "
        "ORDER BY total_citations DESC LIMIT 10"
    ).fetchall()
    print("\n  Top 10 authors by citations:")
    for name, pc, tc in top:
        print(f"    {name}: {pc} papers, {tc:,} citations")

    conn.close()


def build_fts5_index():
    """Build FTS5 full-text search index on title and abstract columns."""
    conn = sqlite3.connect(str(DB_PATH))

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"Building FTS5 index on {total:,} papers...")

    # Drop existing FTS table if any
    conn.execute("DROP TABLE IF EXISTS papers_fts")

    # Create FTS5 virtual table (content-sync with papers table)
    conn.execute("""
        CREATE VIRTUAL TABLE papers_fts USING fts5(
            title, abstract,
            content=papers, content_rowid=rowid
        )
    """)

    # Populate the FTS index
    t0 = time.time()
    conn.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    conn.commit()

    elapsed = int(time.time() - t0)
    print(f"FTS5 index built in {elapsed}s")

    # Verify
    test = conn.execute(
        "SELECT COUNT(*) FROM papers_fts WHERE papers_fts MATCH 'machine learning'"
    ).fetchone()[0]
    print(f"Verification: 'machine learning' matches {test:,} papers")

    conn.close()
    print("Done!")


def main():
    setup_logging()

    if len(sys.argv) < 2:
        build_database()
    elif sys.argv[1] == "--query":
        interactive_query()
    elif sys.argv[1] == "--build-fts":
        build_fts5_index()
    elif sys.argv[1] == "--build-authors":
        build_authors()
    elif sys.argv[1] == "--export-top":
        export_subset(["Tier1-FT/UTD", "Tier2-Basket8"], "papers.csv", citation_min=0)
    elif sys.argv[1] == "--export-broad":
        export_subset(["Tier1-FT/UTD", "Tier2-Basket8", "Tier3-Strong"], "papers.csv", citation_min=0)
    else:
        print("Usage:")
        print("  python build_db.py               # Build database from snapshot")
        print("  python build_db.py --query        # Interactive SQL queries")
        print("  python build_db.py --build-fts    # Build FTS5 full-text search index")
        print("  python build_db.py --build-authors # Build authors + paper_authors tables")
        print("  python build_db.py --export-top   # Export Tier1+2 → papers.csv")
        print("  python build_db.py --export-broad # Export Tier1+2+3 → papers.csv")


if __name__ == "__main__":
    main()
