"""
LLM-powered analysis: intelligent search (tool_use) + cluster/landscape analysis.

Supports two modes for API key:
1. ANTHROPIC_API_KEY in .env (for CLI / server-side)
2. api_key parameter (passed from frontend)
"""

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from utils import PROJECT_ROOT, DATA_DIR, get_logger, setup_logging

logger = get_logger("analyze")

load_dotenv(PROJECT_ROOT / ".env")

# Lakehouse: use derived search.db if available, fall back to legacy openalex.db
_LAKEHOUSE_SEARCH_DB = Path("N:/academic-data/derived/search.db")
DB_PATH = _LAKEHOUSE_SEARCH_DB if _LAKEHOUSE_SEARCH_DB.exists() else DATA_DIR / "openalex.db"


def _get_client(api_key: str | None = None):
    """Get Anthropic client. Uses provided key or falls back to .env."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "Provide your Anthropic API key (get one at https://console.anthropic.com/)"
        )
    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# Tool definitions for Claude tool_use
# ---------------------------------------------------------------------------
SEARCH_TOOL = {
    "name": "search_papers",
    "description": (
        "Search the academic papers database (OpenAlex — all disciplines). "
        "Uses FTS5 full-text search on title and abstract. "
        "You can combine keyword search with structured filters. "
        "Call this tool multiple times with different queries to build a comprehensive result set. "
        "Each call returns up to `limit` papers sorted by relevance (BM25) then citations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "FTS5 search terms. Each string is an OR group; within a string, "
                    "words are ANDed. Example: ['machine learning', 'deep learning neural'] "
                    "matches papers containing 'machine' AND 'learning', OR 'deep' AND 'learning' AND 'neural'. "
                    "Use short phrases (2-4 words). Include synonyms and cross-disciplinary terms."
                ),
            },
            "year_min": {
                "type": "integer",
                "description": "Minimum publication year (default: no limit)",
            },
            "year_max": {
                "type": "integer",
                "description": "Maximum publication year (default: no limit)",
            },
            "citation_min": {
                "type": "integer",
                "description": "Minimum citation count (default: 0). Use low values (0-10) for recent/niche topics.",
            },
            "journal_tiers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["Tier1-FT/UTD", "Tier2-Basket8", "Tier3-Strong", "Other"],
                },
                "description": (
                    "Filter by journal tier. Tier1=MISQ/ISR/MS, Tier2=Basket8, "
                    "Tier3=Strong IS journals, Other=remaining. Omit for all tiers."
                ),
            },
            "subfields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by OpenAlex subfield name (e.g., 'Information Systems', 'Human-Computer Interaction'). Omit for all.",
            },
            "limit": {
                "type": "integer",
                "description": "Max papers to return (default: 50, max: 200)",
            },
        },
        "required": ["keywords"],
    },
}

GET_DB_STATS_TOOL = {
    "name": "get_db_stats",
    "description": (
        "Get database statistics: total papers, papers by tier, top subfields, year range. "
        "Call this first to understand what's available before searching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Tool execution: search papers via FTS5
# ---------------------------------------------------------------------------
def execute_search(
    keywords: list[str],
    year_min: int | None = None,
    year_max: int | None = None,
    citation_min: int = 0,
    journal_tiers: list[str] | None = None,
    subfields: list[str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """Execute a search against the SQLite FTS5 index."""
    limit = min(max(limit, 1), 200)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Build FTS5 match expression: OR between keyword groups
    # Within each keyword string, words are ANDed by FTS5 by default (space = AND)
    # Between keyword strings, we use OR
    fts_terms = []
    for kw in keywords:
        # Clean up the keyword: remove special FTS5 chars, keep words
        safe = kw.replace('"', '').replace("'", "").strip()
        if safe:
            # Wrap each multi-word group in parens so OR works correctly
            words = safe.split()
            if len(words) > 1:
                fts_terms.append(f'({" ".join(words)})')
            else:
                fts_terms.append(words[0])

    if not fts_terms:
        conn.close()
        return []

    fts_expr = " OR ".join(fts_terms)

    # Build WHERE clauses for structured filters
    where_parts = []
    params = []

    if year_min is not None:
        where_parts.append("p.year >= ?")
        params.append(year_min)
    if year_max is not None:
        where_parts.append("p.year <= ?")
        params.append(year_max)
    if citation_min > 0:
        where_parts.append("p.citation_count >= ?")
        params.append(citation_min)
    if journal_tiers:
        placeholders = ",".join("?" * len(journal_tiers))
        where_parts.append(f"p.journal_tier IN ({placeholders})")
        params.extend(journal_tiers)
    if subfields:
        placeholders = ",".join("?" * len(subfields))
        where_parts.append(f"p.subfield IN ({placeholders})")
        params.extend(subfields)

    where_clause = ""
    if where_parts:
        where_clause = "AND " + " AND ".join(where_parts)

    # Detect available columns (search.db lacks authors/topic; legacy has all)
    col_info = conn.execute("PRAGMA table_info(papers)").fetchall()
    available_cols = {row[1] for row in col_info}
    has_authors = "authors" in available_cols
    has_topic = "topic" in available_cols

    # Build SELECT with only available columns
    select_cols = [
        "p.openalex_id", "p.title", "p.abstract", "p.citation_count",
        "p.year", "p.journal", "p.doi" if "doi" in available_cols else "'' as doi",
    ]
    if has_authors:
        select_cols.append("p.authors")
    else:
        select_cols.append("'' as authors")
    select_cols.append("p.subfield" if "subfield" in available_cols else "'' as subfield")
    select_cols.append("p.field" if "field" in available_cols else "'' as field")
    if has_topic:
        select_cols.append("p.topic")
    else:
        select_cols.append("'' as topic")
    select_cols.append("p.journal_tier")

    query = f"""
        SELECT {', '.join(select_cols)}, rank
        FROM papers_fts fts
        JOIN papers p ON p.rowid = fts.rowid
        WHERE papers_fts MATCH ? {where_clause}
        ORDER BY rank, p.citation_count DESC
        LIMIT ?
    """

    try:
        rows = conn.execute(query, [fts_expr] + params + [limit]).fetchall()
    except Exception as e:
        logger.warning(f"FTS5 query failed: {e}")
        conn.close()
        return []

    results = []
    for row in rows:
        results.append({
            "openalex_id": row["openalex_id"],
            "title": row["title"],
            "abstract": row["abstract"][:300] + "..." if len(row["abstract"] or "") > 300 else (row["abstract"] or ""),
            "citation_count": row["citation_count"],
            "year": row["year"],
            "journal": row["journal"],
            "doi": row["doi"],
            "authors": row["authors"],
            "subfield": row["subfield"],
            "journal_tier": row["journal_tier"],
        })

    conn.close()
    logger.info(f"Search '{fts_expr[:80]}' → {len(results)} results")
    return results


def execute_get_db_stats() -> dict:
    """Get database statistics."""
    conn = sqlite3.connect(str(DB_PATH))

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    tiers = {}
    for row in conn.execute(
        "SELECT journal_tier, COUNT(*) FROM papers GROUP BY journal_tier ORDER BY COUNT(*) DESC"
    ):
        tiers[row[0]] = row[1]

    subfields = {}
    for row in conn.execute(
        "SELECT subfield, COUNT(*) FROM papers GROUP BY subfield ORDER BY COUNT(*) DESC LIMIT 20"
    ):
        subfields[row[0]] = row[1]

    year_range = conn.execute("SELECT MIN(year), MAX(year) FROM papers").fetchone()

    conn.close()
    return {
        "total_papers": total,
        "by_tier": tiers,
        "top_subfields": subfields,
        "year_range": {"min": year_range[0], "max": year_range[1]},
    }


# ---------------------------------------------------------------------------
# Intelligent search: Claude agentic loop with tool_use
# ---------------------------------------------------------------------------
SEARCH_SYSTEM = """You are an expert academic research assistant with access to a comprehensive local database of academic papers from OpenAlex (all disciplines).

Your job: given a user's research interest, search the database intelligently to find the most relevant papers for mapping their research landscape.

Strategy:
1. First, call get_db_stats to understand the database contents.
2. Think about what search terms would capture the research area comprehensively.
3. Search with multiple queries using different angles:
   - Core terminology (the main concept)
   - Synonyms and cross-disciplinary terms
   - Methodological terms if relevant
   - Recent buzzwords AND classic terminology
4. Start broad, then narrow if you get too many results.
5. Use low citation thresholds for recent topics (2020+), higher for established ones.
6. After collecting results, provide a brief summary of what you found.

Important:
- FTS5 search: words within a keyword string are ANDed. Separate keyword strings are ORed.
- You can call search_papers multiple times to cover different angles.
- Keep each search focused (2-4 keywords per call) for better precision.
- Cover both high-citation seminal works AND recent papers.
- The database covers all academic disciplines: sciences, engineering, social sciences, humanities, medicine, etc."""


def intelligent_search(
    research_idea: str,
    api_key: str | None = None,
    max_papers: int = 200,
    callback=None,
) -> dict:
    """
    Use Claude tool_use to intelligently search the local paper database.

    Args:
        research_idea: Natural language description of the research interest
        api_key: Anthropic API key (or None to use .env)
        max_papers: Maximum total papers to collect across all searches
        callback: Optional function(status_msg: str) for progress updates

    Returns:
        {
            "papers": [...],          # Deduplicated list of paper dicts
            "search_log": [...],      # List of search queries performed
            "summary": "...",         # Claude's summary of findings
            "total_found": int,
        }
    """
    client = _get_client(api_key)

    def notify(msg):
        logger.info(msg)
        if callback:
            callback(msg)

    notify("Starting intelligent search...")

    messages = [{"role": "user", "content": research_idea}]
    tools = [SEARCH_TOOL, GET_DB_STATS_TOOL]

    all_papers = {}  # keyed by openalex_id for dedup
    search_log = []
    summary = ""

    # Agentic loop: let Claude call tools until it's done
    for turn in range(15):  # Safety cap
        notify(f"Claude thinking... (turn {turn + 1})")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SEARCH_SYSTEM,
            tools=tools,
            messages=messages,
        )

        # Process response content blocks
        assistant_content = response.content
        has_tool_use = False

        for block in assistant_content:
            if block.type == "text" and block.text.strip():
                summary = block.text

        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            has_tool_use = True
            tool_results = []

            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input

                if tool_name == "search_papers":
                    kw = tool_input.get("keywords", [])
                    notify(f"Searching: {', '.join(kw)}")
                    results = execute_search(
                        keywords=kw,
                        year_min=tool_input.get("year_min"),
                        year_max=tool_input.get("year_max"),
                        citation_min=tool_input.get("citation_min", 0),
                        journal_tiers=tool_input.get("journal_tiers"),
                        subfields=tool_input.get("subfields"),
                        limit=tool_input.get("limit", 50),
                    )

                    # Add to collection (dedup by ID)
                    new_count = 0
                    for p in results:
                        pid = p["openalex_id"]
                        if pid not in all_papers:
                            all_papers[pid] = p
                            new_count += 1

                    search_log.append({
                        "keywords": kw,
                        "filters": {k: v for k, v in tool_input.items() if k != "keywords"},
                        "results": len(results),
                        "new": new_count,
                    })

                    # Return truncated results to Claude (save tokens)
                    brief = [
                        {"title": r["title"], "year": r["year"],
                         "citations": r["citation_count"], "journal": r["journal"],
                         "tier": r["journal_tier"]}
                        for r in results[:20]
                    ]
                    tool_result = json.dumps({
                        "found": len(results),
                        "new_unique": new_count,
                        "total_collected": len(all_papers),
                        "sample": brief,
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    })

                elif tool_name == "get_db_stats":
                    notify("Getting database stats...")
                    stats = execute_get_db_stats()
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(stats),
                    })

            # Add assistant message + tool results
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            # Check if we've hit the paper cap
            if len(all_papers) >= max_papers:
                notify(f"Reached {len(all_papers)} papers, wrapping up...")
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": f"We now have {len(all_papers)} unique papers collected. Please provide your summary of what was found.",
                    }],
                })
        else:
            # Claude is done (stop_reason == "end_turn")
            break

    notify(f"Search complete: {len(all_papers)} unique papers from {len(search_log)} queries")

    return {
        "papers": list(all_papers.values()),
        "search_log": search_log,
        "summary": summary,
        "total_found": len(all_papers),
    }


# ---------------------------------------------------------------------------
# Stage 5+: Analyze a cluster of papers
# ---------------------------------------------------------------------------
CLUSTER_ANALYSIS_SYSTEM = """You are a research literature analyst. You will receive a set of academic papers that belong to the same research cluster (identified by topic modeling).

Analyze these papers and provide a rich narrative in the following structure:

## Cluster Theme
A 2-3 sentence description of what this cluster is fundamentally about.

## Historical Development
Trace how this research topic evolved chronologically. Identify:
- **Foundational works** (earliest, most-cited papers that established the field)
- **Key turning points** where methodology or focus shifted
- **Recent trends** in the most recent papers

## Core Research Questions
What are the main questions researchers in this cluster are trying to answer?

## Methodological Approaches
What methods dominate? (surveys, experiments, computational methods, text mining, etc.)

## Key Findings & Debates
What are the major findings? Are there competing perspectives or debates?

## Research Gaps & Future Directions
Based on the papers, what questions remain unanswered? Where could new research go?

Write in clear, concise academic prose. Reference specific paper titles when making claims. Use the year and citation count to gauge influence and recency."""


def analyze_cluster(
    cluster_id: int,
    cluster_label: str,
    df_cluster: pd.DataFrame,
    api_key: str | None = None,
) -> str:
    """Use Claude to analyze papers in a cluster and generate narrative."""
    client = _get_client(api_key)

    df_sorted = df_cluster.sort_values("year")

    papers_text = []
    for _, row in df_sorted.iterrows():
        title = str(row.get("title", ""))
        year = int(row.get("year", 0))
        citations = int(row.get("citation_count", 0))
        journal = str(row.get("journal", ""))
        abstract = str(row.get("abstract", ""))

        if len(abstract) > 300:
            abstract = abstract[:300] + "..."

        papers_text.append(
            f"- **{title}** ({year}, {journal}, {citations} citations)\n"
            f"  Abstract: {abstract}"
        )

    papers_block = "\n\n".join(papers_text)

    prompt = f"""Analyze this research cluster:

**Cluster Label:** {cluster_label}
**Number of papers:** {len(df_cluster)}
**Year range:** {df_sorted['year'].min()} - {df_sorted['year'].max()}
**Total citations in cluster:** {df_cluster['citation_count'].sum():,}

### Papers in this cluster (sorted chronologically):

{papers_block}

Please provide a comprehensive analysis following the structure in your instructions."""

    logger.info(
        f"Analyzing cluster #{cluster_id} ({cluster_label}): "
        f"{len(df_cluster)} papers"
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=CLUSTER_ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    analysis = message.content[0].text
    logger.info(f"Cluster #{cluster_id} analysis complete ({len(analysis)} chars)")
    return analysis


# ---------------------------------------------------------------------------
# Analyze full landscape (all clusters)
# ---------------------------------------------------------------------------
LANDSCAPE_SYSTEM = """You are a research landscape analyst. You will receive summaries of multiple research clusters discovered through topic modeling of academic papers.

Provide a high-level synthesis:

## Landscape Overview
What is the overall research landscape? How do these clusters relate to each other?

## Cross-Cluster Connections
Which clusters are closely related? What bridges them?

## Evolution of the Field
How has the overall field developed based on the cluster composition?

## Underexplored Areas
What gaps exist between clusters? What topics are missing?

Write concisely in academic prose."""


def analyze_landscape(df: pd.DataFrame, api_key: str | None = None) -> str:
    """Analyze the full research landscape across all clusters."""
    client = _get_client(api_key)

    summaries = []
    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            continue
        mask = df["cluster_id"] == cid
        cluster_df = df[mask]
        label = cluster_df["cluster_label"].iloc[0]
        n = len(cluster_df)
        year_range = f"{cluster_df['year'].min()}-{cluster_df['year'].max()}"
        total_cites = cluster_df["citation_count"].sum()
        top_papers = cluster_df.nlargest(3, "citation_count")
        top_titles = "; ".join(
            f"\"{r['title'][:60]}\" ({r['year']}, {r['citation_count']} cites)"
            for _, r in top_papers.iterrows()
        )
        summaries.append(
            f"**Cluster #{cid}: {label}** ({n} papers, {year_range}, {total_cites:,} total cites)\n"
            f"  Top papers: {top_titles}"
        )

    prompt = f"""Research landscape with {df['cluster_id'].nunique()} clusters and {len(df)} total papers:

{'\\n\\n'.join(summaries)}

Provide a high-level landscape analysis."""

    logger.info(f"Analyzing full landscape: {df['cluster_id'].nunique()} clusters")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        system=LANDSCAPE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text
