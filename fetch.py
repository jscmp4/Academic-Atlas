"""
Stage 2: FETCH — Retrieve papers from OpenAlex (primary) or Semantic Scholar (fallback).

Supports:
- Per-query caching with filter-based invalidation
- Breakpoint resume (each query cached independently)
- Data source abstraction (openalex / semantic_scholar)
- Deduplication across streams
"""

import time
from pathlib import Path

from tqdm import tqdm

from utils import (
    clear_all_cache,
    deduplicate_papers,
    extract_paper_from_openalex,
    get_logger,
    load_cache,
    load_config,
    make_cache_key,
    save_cache,
    save_papers_csv,
    setup_logging,
    DATA_DIR,
)

logger = get_logger("fetch")

# Rate limit: wait between API calls, retry with backoff on 429
API_DELAY = 0.5  # seconds between calls
MAX_RETRIES = 4


def _api_call_with_retry(func, *args, **kwargs):
    """Call an API function with retry + exponential backoff on 429."""
    for attempt in range(MAX_RETRIES):
        try:
            result = func(*args, **kwargs)
            time.sleep(API_DELAY)
            return result
        except Exception as e:
            if "429" in str(e):
                wait = 2 ** attempt * 2  # 2, 4, 8, 16 seconds
                logger.warning(f"  Rate limited (429), waiting {wait}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Max retries ({MAX_RETRIES}) exceeded due to rate limiting")


# ---------------------------------------------------------------------------
# OpenAlex journal ID resolution
# ---------------------------------------------------------------------------
def resolve_journal_ids(journal_config: dict) -> dict[str, str]:
    """Resolve journal names to OpenAlex source IDs.

    Returns dict of {journal_name: openalex_source_id}.
    Caches results to data/journal_ids.json.
    """
    import json

    cache_path = DATA_DIR / "journal_ids.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        logger.info(f"Loaded {len(cached)} cached journal IDs")
        return cached

    from pyalex import Sources

    journal_ids = {}
    all_journals = []
    for category, names in journal_config.items():
        all_journals.extend(names)

    logger.info(f"Resolving OpenAlex IDs for {len(all_journals)} journals...")

    for name in tqdm(all_journals, desc="Resolving journals"):
        try:
            results = _api_call_with_retry(lambda: Sources().search(name).get())
            if results:
                best = results[0]
                journal_ids[name] = best["id"]
                logger.debug(f"  {name} -> {best['id']} ({best['display_name']})")
            else:
                logger.warning(f"  No OpenAlex source found for: {name}")
        except Exception as e:
            logger.error(f"  Error resolving '{name}': {e}")

    # Cache for next run
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(journal_ids, f, ensure_ascii=False, indent=2)

    logger.info(f"Resolved {len(journal_ids)}/{len(all_journals)} journal IDs")
    return journal_ids


# ---------------------------------------------------------------------------
# OpenAlex fetcher
# ---------------------------------------------------------------------------
def fetch_from_openalex(config: dict) -> list[dict]:
    """Fetch papers from OpenAlex API with per-query caching."""
    import pyalex
    from pyalex import Works

    # Configure pyalex
    oa_config = config.get("openalex", {})
    api_key = oa_config.get("api_key", "")
    email = oa_config.get("email", "")

    if api_key:
        pyalex.config.api_key = api_key
    if email:
        pyalex.config.email = email

    if not api_key:
        logger.warning(
            "No OpenAlex API key set! API calls may fail. "
            "Register at https://openalex.org/login"
        )

    # Resolve journal IDs
    journal_config = config.get("journals", {})
    journal_ids = resolve_journal_ids(journal_config)

    # Global filters
    global_filters = config.get("filters", {})
    streams = config.get("search_streams", [])

    all_papers = []
    total_queries = sum(
        len(s.get("keywords", [])) * max(len(journal_ids), 1)
        for s in streams
    )
    logger.info(f"Starting OpenAlex fetch: {len(streams)} streams, ~{total_queries} queries")

    pbar = tqdm(total=total_queries, desc="Fetching papers")

    for stream in streams:
        stream_name = stream.get("name", "Unknown")
        keywords = stream.get("keywords", [])
        # Stream-level citation_min overrides global
        stream_citation_min = stream.get("citation_min", global_filters.get("citation_min", 0))

        current_filters = {
            "citation_min": stream_citation_min,
            "year_min": global_filters.get("year_min", 2000),
            "year_max": global_filters.get("year_max", 2026),
            "doc_type": global_filters.get("doc_type", "journal-article"),
        }

        for keyword in keywords:
            if journal_ids:
                # Search each journal separately
                for journal_name, journal_id in journal_ids.items():
                    cache_key = make_cache_key(stream_name, keyword, journal_name)
                    cached = load_cache(cache_key, current_filters)

                    if cached is not None:
                        all_papers.extend(cached)
                        pbar.update(1)
                        continue

                    papers = _fetch_openalex_query(
                        Works, keyword, journal_id, current_filters, stream_name
                    )
                    save_cache(cache_key, papers, current_filters)
                    all_papers.extend(papers)
                    pbar.update(1)
            else:
                # No journals configured — search without journal filter
                cache_key = make_cache_key(stream_name, keyword, "")
                cached = load_cache(cache_key, current_filters)

                if cached is not None:
                    all_papers.extend(cached)
                    pbar.update(1)
                    continue

                papers = _fetch_openalex_query(
                    Works, keyword, None, current_filters, stream_name
                )
                save_cache(cache_key, papers, current_filters)
                all_papers.extend(papers)
                pbar.update(1)

    pbar.close()
    return all_papers


def _clean_keyword_for_openalex(keyword: str) -> str:
    """Clean Boolean keyword syntax for OpenAlex search.

    OpenAlex title_and_abstract.search does NOT support AND/OR/parentheses.
    It treats all words as implicit AND.
    - Remove AND, OR, parentheses
    - Split into words, keep unique meaningful terms
    """
    import re
    # Remove parentheses
    cleaned = keyword.replace("(", "").replace(")", "")
    # Remove Boolean operators
    cleaned = re.sub(r'\bAND\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bOR\b', ' ', cleaned, flags=re.IGNORECASE)
    # Collapse whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _fetch_openalex_query(
    Works, keyword: str, journal_id: str | None, filters: dict, stream_name: str
) -> list[dict]:
    """Execute a single OpenAlex query and return extracted papers."""
    papers = []

    # Clean keyword for OpenAlex (no Boolean operators)
    search_term = _clean_keyword_for_openalex(keyword)

    try:
        filter_kwargs = {
            "title_and_abstract": {"search": search_term},
            "cited_by_count": f">{filters['citation_min']}",
            "publication_year": f">{filters['year_min'] - 1}",
            "type": filters["doc_type"],
        }
        if journal_id:
            filter_kwargs["primary_location"] = {"source": {"id": journal_id}}

        def _do_query():
            q = (
                Works()
                .filter(**filter_kwargs)
                .sort(cited_by_count="desc")
                .select([
                    "id", "doi", "title", "abstract_inverted_index",
                    "cited_by_count", "publication_year",
                    "primary_location", "authorships",
                ])
            )
            result = []
            for page in q.paginate(per_page=200, n_max=2000):
                for work in page:
                    paper = extract_paper_from_openalex(work, stream_name)
                    if paper and paper["abstract"]:
                        result.append(paper)
            return result

        papers = _api_call_with_retry(_do_query)

        logger.debug(
            f"  [{stream_name}] '{search_term[:50]}' "
            f"journal={journal_id or 'any'} -> {len(papers)} papers"
        )

    except Exception as e:
        logger.error(f"  OpenAlex query failed: {e}")

    return papers


# ---------------------------------------------------------------------------
# Semantic Scholar fetcher (fallback)
# ---------------------------------------------------------------------------
def fetch_from_semantic_scholar(config: dict) -> list[dict]:
    """Fetch papers from Semantic Scholar as fallback.

    Note: S2 has stricter rate limits and no journal-level filtering API.
    Strategy: keyword search only, journal filtering done locally (post-filter).
    """
    import requests

    s2_key = config.get("s2_api_key", "") or ""
    # Also check .env
    import os
    s2_key = s2_key or os.getenv("S2_API_KEY", "")

    headers = {}
    if s2_key:
        headers["x-api-key"] = s2_key

    global_filters = config.get("filters", {})
    streams = config.get("search_streams", [])

    # Build set of allowed journal names for post-filtering
    allowed_journals = set()
    for category, names in config.get("journals", {}).items():
        for name in names:
            allowed_journals.add(name.lower())

    all_papers = []
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,title,abstract,citationCount,year,journal,authors,externalIds"

    for stream in streams:
        stream_name = stream.get("name", "Unknown")
        keywords = stream.get("keywords", [])
        stream_citation_min = stream.get(
            "citation_min", global_filters.get("citation_min", 0)
        )
        current_filters = {
            "citation_min": stream_citation_min,
            "year_min": global_filters.get("year_min", 2000),
            "year_max": global_filters.get("year_max", 2026),
            "doc_type": global_filters.get("doc_type", ""),
        }

        for keyword in tqdm(keywords, desc=f"S2: {stream_name}"):
            cache_key = make_cache_key(stream_name, keyword, "s2")
            cached = load_cache(cache_key, current_filters)
            if cached is not None:
                all_papers.extend(cached)
                continue

            papers = []
            offset = 0
            limit = 100

            while offset < 1000:  # Cap at 1000 per query
                params = {
                    "query": keyword,
                    "offset": offset,
                    "limit": limit,
                    "fields": fields,
                    "year": f"{current_filters['year_min']}-{current_filters['year_max']}",
                    "minCitationCount": current_filters["citation_min"],
                }

                try:
                    resp = requests.get(base_url, params=params, headers=headers, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"  S2 query failed at offset {offset}: {e}")
                    break

                results = data.get("data", [])
                if not results:
                    break

                for r in results:
                    abstract = r.get("abstract", "") or ""
                    title = r.get("title", "") or ""
                    if not abstract:
                        continue

                    # Post-filter by journal name
                    journal_info = r.get("journal") or {}
                    journal_name = journal_info.get("name", "") or ""
                    if allowed_journals and journal_name.lower() not in allowed_journals:
                        continue

                    # Extract DOI
                    ext_ids = r.get("externalIds") or {}
                    doi = ext_ids.get("DOI", "") or ""
                    if doi and not doi.startswith("http"):
                        doi = f"https://doi.org/{doi}"

                    authors = "; ".join(
                        a.get("name", "") for a in (r.get("authors") or [])[:10]
                    )

                    papers.append({
                        "openalex_id": f"s2:{r.get('paperId', '')}",
                        "title": title,
                        "abstract": abstract,
                        "citation_count": r.get("citationCount", 0),
                        "year": r.get("year", 0),
                        "journal": journal_name,
                        "doi": doi,
                        "authors": authors,
                        "streams": stream_name,
                    })

                offset += limit
                total = data.get("total", 0)
                if offset >= total:
                    break

                # Respect rate limit: 1 req/sec without key, 10/sec with key
                time.sleep(1.0 if not s2_key else 0.15)

            save_cache(cache_key, papers, current_filters)
            all_papers.extend(papers)
            logger.debug(f"  [{stream_name}] S2 '{keyword[:40]}...' -> {len(papers)} papers")

    return all_papers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def fetch_papers(config: dict | None = None) -> list[dict]:
    """Fetch papers using configured data source. Returns deduplicated list."""
    if config is None:
        config = load_config()

    # Clear cache if requested
    if config.get("clear_cache", False):
        n = clear_all_cache()
        logger.info(f"Cleared {n} cache files")

    source = config.get("data_source", "openalex")
    logger.info(f"Fetching papers from: {source}")

    if source == "openalex":
        raw_papers = fetch_from_openalex(config)
    elif source == "semantic_scholar":
        raw_papers = fetch_from_semantic_scholar(config)
    else:
        raise ValueError(f"Unknown data_source: {source}")

    logger.info(f"Raw papers fetched: {len(raw_papers)}")

    # Deduplicate
    papers = deduplicate_papers(raw_papers)
    logger.info(f"After deduplication: {len(papers)} unique papers")

    # Filter out papers without abstracts
    papers = [p for p in papers if p.get("abstract", "").strip()]
    logger.info(f"After filtering no-abstract: {len(papers)} papers")

    # Save to CSV
    csv_path = save_papers_csv(papers)
    logger.info(f"Saved to {csv_path}")

    return papers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    setup_logging()
    config = load_config()

    warnings = []
    from utils import validate_config
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(w)

    papers = fetch_papers(config)
    print(f"\nDone! {len(papers)} papers saved to data/papers.csv")
