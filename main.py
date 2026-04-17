"""
Research Landscape Mapping — Pipeline Orchestrator.

Usage:
    python main.py              # Run full pipeline (fetch → embed → cluster)
    python main.py --skip-fetch # Skip fetch, re-run embed + cluster
    python main.py --app-only   # Just launch the Dash web app
    python main.py --clear-cache # Clear fetch cache and re-run everything
"""

import argparse
import sys

from utils import load_config, setup_logging, validate_config, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Research Landscape Mapping Pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip paper fetching")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--app-only", action="store_true", help="Only launch Dash web app")
    parser.add_argument("--clear-cache", action="store_true", help="Clear fetch cache first")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logger = setup_logging()

    # App-only mode
    if args.app_only:
        logger.info("Launching Dash web app...")
        from app import app, datasets
        if not datasets:
            print("No *_clustered.csv found in data/. Run pipeline first.")
            sys.exit(1)
        app.run(debug=True, port=8050)
        return

    # Load config
    config = load_config(args.config)
    warnings = validate_config(config)
    for w in warnings:
        logger.warning(w)

    # Check for critical missing config
    if not config.get("openalex", {}).get("api_key") and config.get("data_source") == "openalex":
        print("\n" + "=" * 60)
        print("OpenAlex API key required!")
        print("1. Register free at https://openalex.org/login")
        print("2. Add your key to .env: OPENALEX_API_KEY=your_key_here")
        print("=" * 60 + "\n")
        response = input("Continue without API key? (may fail) [y/N]: ").strip().lower()
        if response != "y":
            sys.exit(1)

    # Clear cache if requested
    if args.clear_cache:
        config["clear_cache"] = True

    # ---- Stage 2: FETCH ----
    papers_path = DATA_DIR / "papers.csv"
    if not args.skip_fetch:
        logger.info("=" * 60)
        logger.info("Stage 2: FETCH — Retrieving papers")
        logger.info("=" * 60)
        from fetch import fetch_papers
        papers = fetch_papers(config)
        logger.info(f"Fetch complete: {len(papers)} papers")
        if len(papers) == 0:
            logger.error("No papers fetched! Check your config.yaml keywords, journals, and filters.")
            print("\nNo papers found. Possible causes:")
            print("  - Keywords too specific or misspelled")
            print("  - Citation threshold too high")
            print("  - Journal names not matching OpenAlex")
            print("  - API key issue")
            print("\nCheck data/pipeline.log for details.")
            sys.exit(1)
    else:
        if not papers_path.exists():
            logger.error("Cannot skip fetch: data/papers.csv not found")
            sys.exit(1)
        logger.info("Skipping fetch (using existing data/papers.csv)")

    # ---- Stage 3: EMBED ----
    embeddings_path = DATA_DIR / "embeddings.npy"
    if not args.skip_embed:
        logger.info("=" * 60)
        logger.info("Stage 3: EMBED — Generating embeddings")
        logger.info("=" * 60)
        from embed import generate_embeddings
        embeddings, paper_ids = generate_embeddings(config)
        logger.info(f"Embed complete: {embeddings.shape}")
    else:
        if not embeddings_path.exists():
            logger.error("Cannot skip embed: data/embeddings.npy not found")
            sys.exit(1)
        logger.info("Skipping embed (using existing data/embeddings.npy)")

    # ---- Stage 4: CLUSTER ----
    logger.info("=" * 60)
    logger.info("Stage 4: CLUSTER — Dimensionality reduction + clustering")
    logger.info("=" * 60)
    from cluster import cluster_papers
    df = cluster_papers(config)
    logger.info(f"Cluster complete: {len(df)} papers in data/papers_clustered.csv")

    # ---- Done ----
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Papers: {len(df):,}")
    print(f"  Clusters: {df['cluster_id'].nunique()}")
    print(f"  Output: data/papers_clustered.csv")
    print("")
    print("Launch the interactive web app:")
    print("  python app.py")
    print("  → Open http://localhost:8050 in your browser")
    print("=" * 60)


if __name__ == "__main__":
    main()
