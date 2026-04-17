"""
Stage 1: INTERPRET — Optional Claude API integration for generating search config.

Takes a natural-language research idea and produces a structured config.yaml.
Requires ANTHROPIC_API_KEY in .env.

Usage:
    python interpret.py "online community user behavior research"
    python interpret.py  # Interactive prompt
"""

import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils import PROJECT_ROOT, get_logger, setup_logging

logger = get_logger("interpret")

SYSTEM_PROMPT = """You are a research methodology expert helping a scholar map an academic research landscape.

Given a natural-language description of a research interest, produce a structured JSON configuration for searching academic papers. You must:

1. **Interpret the topic**: Clarify and refine the vague research idea into a precise scope.
2. **Generate search streams**: Create 2-4 distinct research streams, each with 3-5 keyword queries using Boolean operators (AND, OR). Cover cross-disciplinary synonyms:
   - IS field terms (e.g., "online community")
   - CS field terms (e.g., "virtual community")
   - Social science terms (e.g., "digital community")
3. **Suggest journals**: Group by category (IS_core, IS_extended, management, CS_HCI, etc.)
4. **Set citation thresholds**: Newer fields should have lower thresholds.
5. **Note concept evolution**: How terminology has changed over time.

Output ONLY valid JSON matching this schema:
{
  "topic": "Refined topic description",
  "search_streams": [
    {
      "name": "Stream Name",
      "keywords": ["keyword1 AND keyword2", "keyword3 OR keyword4"],
      "citation_min": 50,
      "rationale": "Why these keywords"
    }
  ],
  "journals": {
    "IS_core": ["Journal Name 1", "Journal Name 2"],
    "IS_extended": [...],
    "management": [...],
    "CS_HCI": [...]
  },
  "filters": {
    "citation_min": 50,
    "year_min": 2010,
    "year_max": 2026,
    "doc_type": "journal-article"
  },
  "concept_evolution_notes": [
    "Note about how terminology evolved"
  ]
}"""


def interpret_with_claude(research_idea: str) -> dict:
    """Call Claude API to interpret a research idea into structured search config."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set in .env. "
            "Get an API key at https://console.anthropic.com/"
        )

    client = anthropic.Anthropic(api_key=api_key)

    logger.info(f"Sending research idea to Claude: '{research_idea[:100]}...'")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": research_idea},
        ],
    )

    # Extract JSON from response
    response_text = message.content[0].text

    # Try to parse JSON (Claude may wrap it in markdown code blocks)
    json_str = response_text
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]

    result = json.loads(json_str.strip())
    logger.info(f"Claude generated config with {len(result.get('search_streams', []))} streams")

    return result


def merge_into_config(interpreted: dict, config_path: Path | None = None) -> Path:
    """Merge Claude's interpretation into config.yaml."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    # Load existing config for structure/defaults
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Update with interpreted values
    if "topic" in interpreted:
        config["topic"] = interpreted["topic"]

    if "search_streams" in interpreted:
        config["search_streams"] = []
        for stream in interpreted["search_streams"]:
            config["search_streams"].append({
                "name": stream.get("name", "Unknown"),
                "keywords": stream.get("keywords", []),
                "citation_min": stream.get("citation_min", config.get("filters", {}).get("citation_min", 50)),
            })

    if "journals" in interpreted:
        config["journals"] = interpreted["journals"]

    if "filters" in interpreted:
        config["filters"] = interpreted["filters"]

    # Save
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"Updated config saved to {config_path}")
    return config_path


def main():
    setup_logging()

    # Get research idea
    if len(sys.argv) > 1:
        research_idea = " ".join(sys.argv[1:])
    else:
        print("Enter your research idea (natural language):")
        print("Example: 'How do people study user behavior in online communities?'")
        print()
        research_idea = input("> ").strip()

    if not research_idea:
        print("No input provided. Exiting.")
        sys.exit(1)

    print(f"\nInterpreting: '{research_idea}'")
    print("Calling Claude API...\n")

    try:
        result = interpret_with_claude(research_idea)
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install with: pip install anthropic")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nClaude API call failed: {e}")
        print("You can manually edit config.yaml instead.")
        sys.exit(1)

    # Display result
    print("=" * 60)
    print("Claude's Interpretation:")
    print("=" * 60)
    print(f"\nTopic: {result.get('topic', 'N/A')}")

    for stream in result.get("search_streams", []):
        print(f"\nStream: {stream.get('name')}")
        print(f"  Citation min: {stream.get('citation_min', 'default')}")
        for kw in stream.get("keywords", []):
            print(f"  - {kw}")
        if stream.get("rationale"):
            print(f"  Rationale: {stream['rationale']}")

    print(f"\nJournals: {json.dumps(result.get('journals', {}), indent=2)}")

    for note in result.get("concept_evolution_notes", []):
        print(f"\nNote: {note}")

    print("\n" + "=" * 60)

    # Ask to save
    response = input("\nSave to config.yaml? [Y/n]: ").strip().lower()
    if response in ("", "y", "yes"):
        path = merge_into_config(result)
        print(f"\nSaved to {path}")
        print("Review and edit config.yaml, then run: python main.py")
    else:
        # Save as JSON for reference
        ref_path = PROJECT_ROOT / "data" / "interpreted_config.json"
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved raw interpretation to {ref_path}")
        print("You can manually edit config.yaml using this as reference.")


if __name__ == "__main__":
    main()
