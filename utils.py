"""
Shared utilities: logging, config loading, data schema, deduplication, helpers.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Schema
# ---------------------------------------------------------------------------
PAPER_COLUMNS_REQUIRED = [
    "openalex_id", "title", "abstract", "citation_count",
    "year", "journal", "doi", "authors", "streams",
]
PAPER_COLUMNS_OPTIONAL = ["openalex_url", "pdf_url"]

CLUSTER_COLUMNS_EXTRA = ["x", "y", "cluster_id", "cluster_label", "topic_words"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_configured = False


def setup_logging(name: str = "pipeline") -> logging.Logger:
    """Configure logging to file + console. Call once per process."""
    global _log_configured

    logger = logging.getLogger("landscape")

    if _log_configured:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(DATA_DIR / "pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    _log_configured = True
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"landscape.{module_name}")


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------
def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml, override OpenAlex credentials from .env."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy config_default.yaml to config.yaml and edit it."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load .env and override credentials
    load_dotenv(PROJECT_ROOT / ".env")

    env_key = os.getenv("OPENALEX_API_KEY", "")
    env_email = os.getenv("OPENALEX_EMAIL", "")
    if env_key:
        config.setdefault("openalex", {})["api_key"] = env_key
    if env_email:
        config.setdefault("openalex", {})["email"] = env_email

    return config


def validate_config(config: dict) -> list[str]:
    """Return list of warnings/errors about the config."""
    warnings = []
    oa = config.get("openalex", {})
    if not oa.get("api_key"):
        warnings.append(
            "OpenAlex API key not set. Register free at https://openalex.org/login "
            "and add OPENALEX_API_KEY to .env"
        )
    if not oa.get("email"):
        warnings.append(
            "OpenAlex email not set. Set OPENALEX_EMAIL in .env for polite pool access."
        )
    if not config.get("search_streams"):
        warnings.append("No search_streams defined in config.")
    return warnings


# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------
def make_cache_key(stream_name: str, keyword: str, journal: str = "") -> str:
    """Generate a deterministic filename-safe cache key."""
    raw = f"{stream_name}|{keyword}|{journal}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    return h


def get_cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def load_cache(cache_key: str, current_filters: dict) -> list | None:
    """Load cached results if they exist and filters match.

    Returns None if cache is missing or filters don't match (invalidated).
    """
    path = get_cache_path(cache_key)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Check if filter parameters match
    cached_filters = cached.get("filters", {})
    for key in ("citation_min", "year_min", "year_max", "doc_type"):
        if str(cached_filters.get(key)) != str(current_filters.get(key)):
            return None  # Invalidated — filters changed

    return cached.get("papers", [])


def save_cache(cache_key: str, papers: list, filters: dict) -> None:
    """Save query results + filter params to cache."""
    path = get_cache_path(cache_key)
    data = {
        "filters": filters,
        "papers": papers,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clear_all_cache() -> int:
    """Delete all cache files. Returns number deleted."""
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Deduplicate by openalex_id, merging stream tags."""
    seen = {}
    for p in papers:
        oid = p["openalex_id"]
        if oid in seen:
            # Merge streams
            existing_streams = set(seen[oid]["streams"].split("; "))
            new_streams = set(p.get("streams", "").split("; "))
            seen[oid]["streams"] = "; ".join(sorted(existing_streams | new_streams))
            # Keep higher citation count (in case of stale cache)
            seen[oid]["citation_count"] = max(
                seen[oid]["citation_count"], p.get("citation_count", 0)
            )
        else:
            seen[oid] = p
    return list(seen.values())


# ---------------------------------------------------------------------------
# Abstract Reconstruction (for raw API calls, pyalex does this automatically)
# ---------------------------------------------------------------------------
def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct plaintext abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(word for _, word in word_positions)


# ---------------------------------------------------------------------------
# Paper extraction from OpenAlex work object
# ---------------------------------------------------------------------------
def extract_paper_from_openalex(work: dict, stream_name: str = "") -> dict | None:
    """Extract structured paper dict from an OpenAlex work object.

    Returns None if the paper has no usable abstract.
    """
    abstract = work.get("abstract", "")
    if not abstract and work.get("abstract_inverted_index"):
        abstract = reconstruct_abstract(work["abstract_inverted_index"])

    title = work.get("title", "") or ""
    if not abstract and not title:
        return None

    # Extract authors
    authorships = work.get("authorships", [])
    authors = "; ".join(
        a.get("author", {}).get("display_name", "")
        for a in authorships[:10]  # Cap at 10 authors
    )

    # Extract journal
    primary_loc = work.get("primary_location") or {}
    source = primary_loc.get("source") or {}
    journal = source.get("display_name", "")

    # Extract DOI
    doi = work.get("doi", "") or ""

    return {
        "openalex_id": work.get("id", ""),
        "title": title,
        "abstract": abstract,
        "citation_count": work.get("cited_by_count", 0),
        "year": work.get("publication_year", 0),
        "journal": journal,
        "doi": doi,
        "authors": authors,
        "streams": stream_name,
        "openalex_url": work.get("id", ""),
    }


# ---------------------------------------------------------------------------
# DataFrame I/O with schema awareness
# ---------------------------------------------------------------------------
def save_papers_csv(papers: list[dict], path: str | Path | None = None) -> Path:
    """Save papers list to CSV with consistent column ordering."""
    if path is None:
        path = DATA_DIR / "papers.csv"
    path = Path(path)

    df = pd.DataFrame(papers)

    # Ensure required columns exist
    for col in PAPER_COLUMNS_REQUIRED:
        if col not in df.columns:
            df[col] = ""

    # Order columns: required first, then optional, then any extras
    all_known = PAPER_COLUMNS_REQUIRED + PAPER_COLUMNS_OPTIONAL
    ordered = [c for c in all_known if c in df.columns]
    extras = [c for c in df.columns if c not in all_known]
    df = df[ordered + extras]

    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_papers_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load papers CSV, selecting only needed columns (forward-compatible).

    If columns is None, loads all columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Papers file not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    if columns is not None:
        # Only select columns that actually exist
        available = [c for c in columns if c in df.columns]
        df = df[available]

    return df
