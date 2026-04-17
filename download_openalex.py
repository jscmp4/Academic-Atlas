"""
Download OpenAlex snapshot and extract IS-related papers.

Strategy: Filter by OpenAlex subfield (not just journal name) to get comprehensive
coverage, then tag each paper with journal tier (Top/A/B/Other).

Usage:
    python download_openalex.py download     # Download from S3
    python download_openalex.py extract      # Extract IS papers by subfield
    python download_openalex.py download+extract  # Both
"""

import gzip
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from utils import PROJECT_ROOT, DATA_DIR, setup_logging, get_logger, reconstruct_abstract

logger = get_logger("download")

SNAPSHOT_DIR = Path("N:/openalex-snapshot")
WORKS_DIR = SNAPSHOT_DIR / "data" / "works"
SOURCES_DIR = SNAPSHOT_DIR / "data" / "sources"
EXTRACTED_CSV = DATA_DIR / "papers.csv"

# ---------------------------------------------------------------------------
# IS-related subfields (from OpenAlex topic taxonomy)
# ---------------------------------------------------------------------------
IS_SUBFIELDS = {
    # Core IS
    "Information Systems",
    "Information Systems and Management",
    "Management Information Systems",
    # CS adjacent
    "Computer Science Applications",
    "Computer Networks and Communications",
    "Software",
    "Human-Computer Interaction",
    "Artificial Intelligence",
    # Business/Management adjacent
    "Management Science and Operations Research",
    "Strategy and Management",
    "Organizational Behavior and Human Resource Management",
    "Business and International Management",
    "Marketing",
    # Decision Sciences
    "Decision Sciences (miscellaneous)",
    "Statistics, Probability and Uncertainty",
    # Social/Communication
    "Communication",
    "Social Sciences (miscellaneous)",
    "Sociology and Political Science",
    "Library and Information Sciences",
}

# ---------------------------------------------------------------------------
# Journal tiers for IS research (based on established academic consensus)
#
# Tier 1: FT50 ∩ UTD24 — the 2 IS journals on both cross-disciplinary lists
#         + Management Science (publishes significant IS research)
# Tier 2: AIS Basket of 8 (remaining 6 not in Tier 1)
# Tier 3: ABDC A* / ABS 3-4 level IS journals
# Other:  All remaining IS-related journals
# ---------------------------------------------------------------------------
TIER_1_FT_UTD = {
    # Only 2 dedicated IS journals appear on BOTH FT50 and UTD24
    "mis quarterly",
    "information systems research",
    # Cross-disciplinary but publishes major IS research
    "management science",
}

TIER_2_BASKET8 = {
    # Remaining 6 of the AIS Senior Scholars' Basket of 8
    "journal of management information systems",
    "journal of the association for information systems",
    "european journal of information systems",
    "information systems journal",
    "journal of information technology",
    "journal of strategic information systems",
}

TIER_3_STRONG = {
    # Other UTD24/FT50 that publish IS-adjacent research
    "organization science",
    "academy of management journal",
    "academy of management review",
    "administrative science quarterly",
    "strategic management journal",
    # ABDC A* / ABS 3-4 IS journals
    "information & management",
    "decision support systems",
    "communications of the acm",
    "international journal of information management",
    "information processing & management",
    "government information quarterly",
    "computers in human behavior",
    "journal of computer-mediated communication",
    "international journal of human-computer studies",
    "acm transactions on computer-human interaction",
    "new media & society",
    "electronic commerce research and applications",
    "journal of the american medical informatics association",
    "decision sciences",
    "information systems frontiers",
    "internet research",
    "information and organization",
    "behaviour & information technology",
    "electronic markets",
    "information technology & people",
    "journal of global information management",
    "journal of database management",
    "computer supported cooperative work",
    "the information society",
    "information communication & society",
    "social media + society",
    "communications of the association for information systems",
    "ethics and information technology",
    "journal of information systems",
    "journal of the american society for information science and technology",
}


def get_journal_tier(journal_name: str) -> str:
    """Return tier label for a journal based on academic consensus rankings."""
    name = journal_name.lower().strip()
    if name in TIER_1_FT_UTD:
        return "Tier1-FT/UTD"
    if name in TIER_2_BASKET8:
        return "Tier2-Basket8"
    if name in TIER_3_STRONG:
        return "Tier3-Strong"
    return "Other"


# ---------------------------------------------------------------------------
# Step 1: Download from S3
# ---------------------------------------------------------------------------
def download_snapshot():
    """Download OpenAlex works + sources from S3."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")
    bucket = "openalex"

    print("=" * 60)
    print("Downloading OpenAlex snapshot from AWS S3")
    print(f"Destination: {SNAPSHOT_DIR}")
    print("=" * 60)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _sync_s3_prefix(s3, bucket, "data/sources/", SOURCES_DIR)
    _sync_s3_prefix(s3, bucket, "data/works/", WORKS_DIR)
    print("\nDownload complete!")


def _sync_s3_prefix(s3, bucket: str, prefix: str, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Listing s3://{bucket}/{prefix} ---")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            files.append((obj["Key"], obj["Size"]))

    print(f"Found {len(files)} files")
    downloaded = 0
    skipped = 0
    total_bytes = sum(size for _, size in files)
    done_bytes = 0

    for i, (key, size) in enumerate(files):
        rel_path = key[len(prefix):]
        local_path = local_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size == size:
            skipped += 1
            done_bytes += size
            continue
        pct = done_bytes / total_bytes * 100 if total_bytes else 0
        print(f"  [{i+1}/{len(files)}] ({pct:.1f}%) {rel_path} ({size/1024/1024:.1f}MB)")
        s3.download_file(bucket, key, str(local_path))
        downloaded += 1
        done_bytes += size

    print(f"  Done: {downloaded} downloaded, {skipped} skipped")


# ---------------------------------------------------------------------------
# Step 2: Extract papers by subfield + tag with journal tier
# ---------------------------------------------------------------------------
def extract_papers():
    """Extract all IS-related papers from snapshot using subfield filtering.

    Streams results directly to CSV to avoid memory issues with large datasets.
    """
    import csv

    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filters = config.get("filters", {})
    year_min = filters.get("year_min", 2000)
    year_max = filters.get("year_max", 2026)

    target_subfields = {s.lower() for s in IS_SUBFIELDS}

    print("=" * 60, flush=True)
    print("Extracting IS-related papers from OpenAlex snapshot", flush=True)
    print("=" * 60, flush=True)
    print(f"  Filter by subfields: {len(IS_SUBFIELDS)} IS-related subfields", flush=True)
    print(f"  Year range: {year_min}-{year_max}", flush=True)
    print(f"  Streaming to CSV (low memory)", flush=True)
    print(flush=True)

    columns = [
        "openalex_id", "title", "abstract", "citation_count", "year",
        "journal", "doi", "authors", "streams", "field", "subfield",
        "topic", "journal_tier",
    ]

    gz_files = sorted(WORKS_DIR.rglob("*.gz"))
    print(f"  Total files to scan: {len(gz_files)}", flush=True)

    files_processed = 0
    total_scanned = 0
    papers_found = 0

    # Stream directly to CSV file
    with open(EXTRACTED_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for gz_file in gz_files:
            files_processed += 1
            if files_processed % 50 == 0:
                print(f"  Progress: {files_processed}/{len(gz_files)} files | "
                      f"{papers_found:,} IS papers | {total_scanned:,} scanned",
                      flush=True)

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

                        # Check subfield match
                        matched_subfield = ""
                        pt = work.get("primary_topic") or {}
                        pt_subfield = (pt.get("subfield") or {}).get("display_name", "")
                        if pt_subfield.lower() in target_subfields:
                            matched_subfield = pt_subfield
                        else:
                            for topic in work.get("topics", []):
                                sf = (topic.get("subfield") or {}).get("display_name", "")
                                if sf.lower() in target_subfields:
                                    matched_subfield = sf
                                    break

                        if not matched_subfield:
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

                        writer.writerow({
                            "openalex_id": work.get("id", ""),
                            "title": work.get("title", ""),
                            "abstract": abstract,
                            "citation_count": work.get("cited_by_count", 0),
                            "year": year,
                            "journal": journal,
                            "doi": work.get("doi", ""),
                            "authors": authors,
                            "streams": matched_subfield,
                            "field": (pt.get("field") or {}).get("display_name", ""),
                            "subfield": pt_subfield,
                            "topic": pt.get("display_name", ""),
                            "journal_tier": get_journal_tier(journal),
                        })
                        papers_found += 1

            except Exception:
                pass

    print(f"\nDone! Scanned {total_scanned:,} works, found {papers_found:,} IS-related papers.",
          flush=True)
    print(f"Saved to {EXTRACTED_CSV}", flush=True)

    # Print stats from the CSV (read back, lightweight)
    df = pd.read_csv(EXTRACTED_CSV, encoding="utf-8-sig")
    df = df.sort_values("citation_count", ascending=False)
    df.to_csv(EXTRACTED_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}", flush=True)
    print(f"Stats", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Total papers: {len(df):,}", flush=True)
    print(f"  Year range: {df['year'].min()}-{df['year'].max()}", flush=True)
    print(f"  Unique journals: {df['journal'].nunique()}", flush=True)

    print(f"\n  By journal tier:", flush=True)
    for tier, count in df["journal_tier"].value_counts().items():
        print(f"    {tier}: {count:,}", flush=True)

    print(f"\n  By subfield (top 15):", flush=True)
    for sf, count in df["subfield"].value_counts().head(15).items():
        print(f"    {sf}: {count:,}", flush=True)

    print(f"\n  Top 20 journals:", flush=True)
    for journal, count in df["journal"].value_counts().head(20).items():
        tier = get_journal_tier(journal)
        print(f"    [{tier}] {journal}: {count:,}", flush=True)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_logging()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python download_openalex.py download          # Download snapshot")
        print("  python download_openalex.py extract           # Extract IS papers")
        print("  python download_openalex.py download+extract  # Both")
        sys.exit(0)

    action = sys.argv[1].lower()

    if "download" in action:
        download_snapshot()

    if "extract" in action:
        if not WORKS_DIR.exists():
            print(f"Works not found at {WORKS_DIR}. Run download first.")
            sys.exit(1)
        extract_papers()
        print("\nNext: python embed.py && python cluster.py")


if __name__ == "__main__":
    main()
