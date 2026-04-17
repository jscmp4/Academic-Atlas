"""
Extract full OpenAlex snapshot to Parquet (Lakehouse Lite — Step 1).

Streams through 583GB of .gz files, extracts ALL papers and authorships,
and writes year-partitioned Parquet files. Memory stays bounded via
periodic flushing (~50K rows per batch per year).

Parquet files are the single source of truth. All derived databases
(DuckDB, SQLite FTS5) are rebuilt from these files.

Output: N:/academic-data/parquet/
  - works_YYYY.parquet    (one per publication year)
  - authorships_YYYY.parquet (one per publication year)
  - authors.parquet       (derived later via build_derived.py)

Usage:
    python extract_to_parquet.py              # Full extraction
    python extract_to_parquet.py --test N     # Test with first N .gz files
    python extract_to_parquet.py --verify     # Verify output counts
    python extract_to_parquet.py --resume     # Resume interrupted extraction
"""

import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utils import reconstruct_abstract
from download_openalex import get_journal_tier, WORKS_DIR

OUTPUT_DIR = Path("N:/academic-data/parquet")
FLUSH_SIZE = 50_000  # rows buffered per year before flushing
PROGRESS_INTERVAL = 100  # print progress every N .gz files

# ---------------------------------------------------------------------------
# Parquet Schemas
# ---------------------------------------------------------------------------

PAPERS_SCHEMA = pa.schema([
    ("openalex_id", pa.string()),
    ("doi", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("publication_year", pa.int16()),
    ("publication_date", pa.string()),
    ("language", pa.string()),
    ("type", pa.string()),
    ("cited_by_count", pa.int32()),
    ("is_retracted", pa.bool_()),
    ("source_id", pa.string()),
    ("source_name", pa.string()),
    ("source_issn", pa.string()),
    ("landing_page_url", pa.string()),
    ("pdf_url", pa.string()),
    ("field", pa.string()),
    ("subfield", pa.string()),
    ("topic", pa.string()),
    ("referenced_works_count", pa.int32()),
    ("journal_tier", pa.string()),
])

AUTHORSHIPS_SCHEMA = pa.schema([
    ("paper_id", pa.string()),
    ("author_id", pa.string()),
    ("author_name", pa.string()),
    ("orcid", pa.string()),
    ("position", pa.string()),       # first / middle / last
    ("is_corresponding", pa.bool_()),
    ("institution", pa.string()),
])

PAPER_COLS = [f.name for f in PAPERS_SCHEMA]
AUTHORSHIP_COLS = [f.name for f in AUTHORSHIPS_SCHEMA]


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def _empty_buf(columns: list[str]) -> dict[str, list]:
    return {col: [] for col in columns}


def _buf_len(buf: dict[str, list]) -> int:
    first_col = next(iter(buf))
    return len(buf[first_col])


def _flush(year: int, buf: dict[str, list], writers: dict, schema: pa.Schema, prefix: str) -> int:
    """Write buffered column-lists to Parquet. Returns rows written."""
    n = _buf_len(buf)
    if n == 0:
        return 0

    if year not in writers:
        path = OUTPUT_DIR / f"{prefix}_{year}.parquet"
        writers[year] = pq.ParquetWriter(str(path), schema, compression="snappy")

    table = pa.table(buf, schema=schema)
    writers[year].write_table(table)
    return n


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_all(max_files: int | None = None, resume: bool = False):
    """Stream full OpenAlex snapshot into year-partitioned Parquet.

    Args:
        max_files: If set, only process first N .gz files (for testing).
        resume: If True, skip .gz files whose output already exists. Note:
                resume is approximate — it skips entire files, not rows.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gz_files = sorted(WORKS_DIR.rglob("*.gz"))
    if max_files:
        gz_files = gz_files[:max_files]

    # Resume support: track which files were fully processed
    progress_file = OUTPUT_DIR / "_progress.txt"
    done_files: set[str] = set()
    if resume and progress_file.exists():
        done_files = set(progress_file.read_text().strip().splitlines())
        print(f"Resuming: {len(done_files)} files already processed")

    print(f"{'=' * 60}")
    print(f"Extracting OpenAlex snapshot → Parquet")
    print(f"{'=' * 60}")
    print(f"  Source: {WORKS_DIR} ({len(gz_files)} .gz files)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Flush every {FLUSH_SIZE:,} rows/year")
    if max_files:
        print(f"  TEST MODE: processing only {max_files} files")
    print()

    # Column-oriented buffers: year → {col: [values]}
    paper_bufs: dict[int, dict] = defaultdict(lambda: _empty_buf(PAPER_COLS))
    auth_bufs: dict[int, dict] = defaultdict(lambda: _empty_buf(AUTHORSHIP_COLS))

    # ParquetWriters: year → writer (opened lazily, append mode via repeated write_table)
    paper_writers: dict[int, pq.ParquetWriter] = {}
    auth_writers: dict[int, pq.ParquetWriter] = {}

    total_papers = 0
    total_authorships = 0
    total_scanned = 0
    files_done = 0
    t0 = time.time()

    for file_idx, gz_file in enumerate(gz_files):
        # Skip already-processed files in resume mode
        if gz_file.name in done_files:
            files_done += 1
            continue

        if (file_idx + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = total_papers / elapsed if elapsed > 0 else 0
            files_remaining = len(gz_files) - file_idx
            eta_h = (files_remaining / ((file_idx + 1) / elapsed)) / 3600 if elapsed > 0 else 0
            print(
                f"  [{file_idx + 1}/{len(gz_files)}] "
                f"{total_papers:,} papers | {total_authorships:,} authorships | "
                f"{rate:,.0f} p/s | ETA {eta_h:.1f}h",
                flush=True,
            )

        try:
            with gzip.open(gz_file, "rt", encoding="utf-8") as f:
                for line in f:
                    total_scanned += 1
                    try:
                        work = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    year = work.get("publication_year")
                    if not year:
                        continue

                    # --- Paper fields ---
                    pt = work.get("primary_topic") or {}
                    primary_loc = work.get("primary_location") or {}
                    best_oa = work.get("best_oa_location") or {}
                    source = primary_loc.get("source") or {}
                    source_name = source.get("display_name") or ""

                    inv_idx = work.get("abstract_inverted_index")
                    abstract = reconstruct_abstract(inv_idx) if inv_idx else ""

                    buf = paper_bufs[year]
                    buf["openalex_id"].append(work.get("id") or "")
                    buf["doi"].append(work.get("doi") or "")
                    buf["title"].append(work.get("title") or "")
                    buf["abstract"].append(abstract)
                    buf["publication_year"].append(year)
                    buf["publication_date"].append(work.get("publication_date") or "")
                    buf["language"].append(work.get("language") or "")
                    buf["type"].append(work.get("type") or "")
                    buf["cited_by_count"].append(work.get("cited_by_count") or 0)
                    buf["is_retracted"].append(bool(work.get("is_retracted")))
                    buf["source_id"].append(source.get("id") or "")
                    buf["source_name"].append(source_name)
                    buf["source_issn"].append(source.get("issn_l") or "")
                    buf["landing_page_url"].append(primary_loc.get("landing_page_url") or "")
                    buf["pdf_url"].append(best_oa.get("pdf_url") or "")
                    buf["field"].append((pt.get("field") or {}).get("display_name") or "")
                    buf["subfield"].append((pt.get("subfield") or {}).get("display_name") or "")
                    buf["topic"].append(pt.get("display_name") or "")
                    buf["referenced_works_count"].append(work.get("referenced_works_count") or 0)
                    buf["journal_tier"].append(get_journal_tier(source_name))
                    total_papers += 1

                    # Flush papers if buffer full for this year
                    if _buf_len(buf) >= FLUSH_SIZE:
                        _flush(year, buf, paper_writers, PAPERS_SCHEMA, "works")
                        paper_bufs[year] = _empty_buf(PAPER_COLS)

                    # --- Authorships ---
                    work_id = work.get("id") or ""
                    corresponding_ids = set(work.get("corresponding_author_ids") or [])

                    for authorship in work.get("authorships", []):
                        author = authorship.get("author") or {}
                        author_id = author.get("id") or ""
                        if not author_id:
                            continue

                        institutions = authorship.get("institutions") or []
                        institution = institutions[0].get("display_name", "") if institutions else ""

                        abuf = auth_bufs[year]
                        abuf["paper_id"].append(work_id)
                        abuf["author_id"].append(author_id)
                        abuf["author_name"].append(author.get("display_name") or "")
                        abuf["orcid"].append(author.get("orcid") or "")
                        abuf["position"].append(authorship.get("author_position") or "")
                        abuf["is_corresponding"].append(
                            bool(authorship.get("is_corresponding"))
                            or author_id in corresponding_ids
                        )
                        abuf["institution"].append(institution)
                        total_authorships += 1

                        if _buf_len(abuf) >= FLUSH_SIZE:
                            _flush(year, abuf, auth_writers, AUTHORSHIPS_SCHEMA, "authorships")
                            auth_bufs[year] = _empty_buf(AUTHORSHIP_COLS)

        except Exception as e:
            print(f"  ⚠ Error in {gz_file.name}: {e}", flush=True)

        files_done += 1

        # Record progress for resume
        with open(progress_file, "a") as pf:
            pf.write(gz_file.name + "\n")

    # --- Final flush ---
    for y, buf in paper_bufs.items():
        _flush(y, buf, paper_writers, PAPERS_SCHEMA, "works")
    for y, buf in auth_bufs.items():
        _flush(y, buf, auth_writers, AUTHORSHIPS_SCHEMA, "authorships")

    for w in paper_writers.values():
        w.close()
    for w in auth_writers.values():
        w.close()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Extraction complete!")
    print(f"{'=' * 60}")
    print(f"  Scanned:      {total_scanned:,} lines")
    print(f"  Papers:       {total_papers:,}")
    print(f"  Authorships:  {total_authorships:,}")
    print(f"  Time:         {elapsed / 3600:.1f} hours ({elapsed:.0f}s)")
    print(f"  Output:       {OUTPUT_DIR}")

    parquet_files = sorted(OUTPUT_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    print(f"  Files:        {len(parquet_files)}")
    print(f"  Total size:   {total_size / 1024 ** 3:.1f} GB")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify():
    """Verify Parquet output counts and spot-check fields."""
    try:
        import duckdb
    except ImportError:
        print("duckdb not installed — falling back to PyArrow metadata check")
        _verify_pyarrow()
        return

    works_glob = str(OUTPUT_DIR / "works_*.parquet").replace("\\", "/")
    auth_glob = str(OUTPUT_DIR / "authorships_*.parquet").replace("\\", "/")

    conn = duckdb.connect()

    print(f"{'=' * 60}")
    print("Verifying Parquet output")
    print(f"{'=' * 60}")

    total = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{works_glob}')").fetchone()[0]
    print(f"\nTotal papers: {total:,}")

    print("\nPapers by year (recent):")
    for row in conn.execute(f"""
        SELECT publication_year, COUNT(*) as n
        FROM read_parquet('{works_glob}')
        GROUP BY publication_year ORDER BY publication_year DESC LIMIT 10
    """).fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    print("\nBy field (top 10):")
    for row in conn.execute(f"""
        SELECT field, COUNT(*) as n
        FROM read_parquet('{works_glob}')
        WHERE field != '' GROUP BY field ORDER BY n DESC LIMIT 10
    """).fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    print("\nBy type:")
    for row in conn.execute(f"""
        SELECT type, COUNT(*) as n
        FROM read_parquet('{works_glob}')
        WHERE type != '' GROUP BY type ORDER BY n DESC LIMIT 8
    """).fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    total_auth = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{auth_glob}')").fetchone()[0]
    print(f"\nTotal authorships: {total_auth:,}")

    unique_authors = conn.execute(
        f"SELECT COUNT(DISTINCT author_id) FROM read_parquet('{auth_glob}')"
    ).fetchone()[0]
    print(f"Unique authors: {unique_authors:,}")

    print("\nSpot check — 3 random highly-cited papers:")
    for row in conn.execute(f"""
        SELECT openalex_id, title, publication_year, cited_by_count, source_name, field
        FROM read_parquet('{works_glob}')
        WHERE title != '' AND cited_by_count > 500
        ORDER BY RANDOM() LIMIT 3
    """).fetchall():
        title = (row[1] or "")[:80]
        print(f"  [{row[2]}] {title}...")
        print(f"    ID={row[0]}  cites={row[3]:,}  source={row[4]}  field={row[5]}")

    # Check for null/empty rates
    print("\nField completeness:")
    for col in ["abstract", "doi", "field", "source_name", "language"]:
        filled = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{works_glob}') WHERE {col} != ''
        """).fetchone()[0]
        pct = filled / total * 100 if total else 0
        print(f"  {col}: {filled:,} ({pct:.1f}%)")

    conn.close()


def _verify_pyarrow():
    """Fallback verification without duckdb."""
    total = 0
    for f in sorted(OUTPUT_DIR.glob("works_*.parquet")):
        meta = pq.read_metadata(str(f))
        total += meta.num_rows
        print(f"  {f.name}: {meta.num_rows:,} rows")
    print(f"\nTotal papers: {total:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        extract_all()
    elif sys.argv[1] == "--verify":
        verify()
    elif sys.argv[1] == "--resume":
        extract_all(resume=True)
    elif sys.argv[1] == "--test":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        extract_all(max_files=n)
    else:
        print("Usage:")
        print("  python extract_to_parquet.py              # Full extraction")
        print("  python extract_to_parquet.py --test N     # Test with N .gz files")
        print("  python extract_to_parquet.py --resume     # Resume interrupted run")
        print("  python extract_to_parquet.py --verify     # Verify output")


if __name__ == "__main__":
    main()
