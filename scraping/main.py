"""3D Print Orientation Research Scraper - Main Orchestrator."""

import argparse
import logging
import os
import sys
import traceback

from dotenv import load_dotenv

from config import LOG_FILE, RAW_DIR


def setup_logging():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler - detailed
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(fh)

    # Console handler - info only
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root_logger.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Print Orientation Research Scraper"
    )
    parser.add_argument(
        "--skip-youtube", action="store_true",
        help="Skip the YouTube scraper (slowest)",
    )
    parser.add_argument(
        "--max-posts", type=int, default=None,
        help="Limit Reddit posts per query (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be scraped without fetching data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    setup_logging()
    logger = logging.getLogger(__name__)

    os.makedirs(RAW_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Pipeline started")
    if args.dry_run:
        logger.info("DRY RUN MODE - no data will be written")

    # Reddit (skip if no credentials)
    if os.getenv("REDDIT_CLIENT_ID"):
        try:
            logger.info("Starting Reddit scraper...")
            from scrapers import scrape_reddit
            count = scrape_reddit(max_posts=args.max_posts, dry_run=args.dry_run)
            logger.info(f"Reddit: {count} new posts")
        except Exception:
            logger.error(f"Reddit scraper failed:\n{traceback.format_exc()}")
    else:
        logger.info("Skipping Reddit scraper (REDDIT_CLIENT_ID not set)")

    # GitHub
    try:
        logger.info("Starting GitHub scraper...")
        from scrapers import scrape_github
        count = scrape_github(dry_run=args.dry_run)
        logger.info(f"GitHub: {count} new issues")
    except Exception:
        logger.error(f"GitHub scraper failed:\n{traceback.format_exc()}")

    # YouTube
    if not args.skip_youtube:
        try:
            logger.info("Starting YouTube scraper...")
            from scrapers import scrape_youtube
            count = scrape_youtube(dry_run=args.dry_run)
            logger.info(f"YouTube: {count} new videos")
        except Exception:
            logger.error(f"YouTube scraper failed:\n{traceback.format_exc()}")
    else:
        logger.info("Skipping YouTube scraper")

    logger.info("Pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
