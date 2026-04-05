"""Reddit scraper using PRAW."""

import json
import logging
import os
from pathlib import Path

import praw
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    RAW_DIR,
    REDDIT_COMMENTS_PER_POST,
    REDDIT_POSTS_PER_QUERY,
    REDDIT_QUERIES,
    SUBREDDITS,
)

logger = logging.getLogger(__name__)

OUTPUT_FILE = os.path.join(RAW_DIR, "reddit.jsonl")


def _init_praw():
    load_dotenv()
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.getenv("REDDIT_USER_AGENT", "3dscraping/1.0"),
    )


def _load_existing_ids(filepath):
    ids = set()
    if not os.path.exists(filepath):
        return ids
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                ids.add(record["post_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _fetch_comments(submission, limit):
    submission.comment_sort = "best"
    submission.comments.replace_more(limit=0)

    comments = []
    for comment in submission.comments:
        if comment.body in ("[deleted]", "[removed]"):
            continue
        comments.append({
            "body": comment.body,
            "score": comment.score,
            "author": str(comment.author) if comment.author else "[deleted]",
        })
        # Also grab second-level replies
        for reply in comment.replies:
            if hasattr(reply, "body") and reply.body not in ("[deleted]", "[removed]"):
                comments.append({
                    "body": reply.body,
                    "score": reply.score,
                    "author": str(reply.author) if reply.author else "[deleted]",
                })

    comments.sort(key=lambda c: c["score"], reverse=True)
    return comments[:limit]


def _serialize_post(submission, comments):
    return {
        "source": "reddit",
        "subreddit": str(submission.subreddit),
        "post_id": submission.id,
        "title": submission.title,
        "body": submission.selftext,
        "score": submission.score,
        "num_comments": submission.num_comments,
        "url": f"https://reddit.com{submission.permalink}",
        "created_utc": submission.created_utc,
        "comments": comments,
    }


def scrape_reddit(max_posts=None, dry_run=False):
    reddit = _init_praw()
    existing_ids = _load_existing_ids(OUTPUT_FILE)
    posts_per_query = max_posts or REDDIT_POSTS_PER_QUERY
    new_records = []

    logger.info(
        f"Reddit scraper: {len(SUBREDDITS)} subreddits x {len(REDDIT_QUERIES)} queries, "
        f"limit={posts_per_query}, existing={len(existing_ids)}"
    )

    for sub_name in tqdm(SUBREDDITS, desc="Subreddits"):
        subreddit = reddit.subreddit(sub_name)
        for query in tqdm(REDDIT_QUERIES, desc=f"r/{sub_name}", leave=False):
            try:
                results = subreddit.search(query, limit=posts_per_query)
                for submission in results:
                    if submission.id in existing_ids:
                        continue
                    existing_ids.add(submission.id)

                    if dry_run:
                        tqdm.write(f"  [DRY] r/{sub_name} | {submission.id} | {submission.title[:80]}")
                        new_records.append(None)
                        continue

                    try:
                        comments = _fetch_comments(submission, REDDIT_COMMENTS_PER_POST)
                        record = _serialize_post(submission, comments)
                        new_records.append(record)
                    except Exception as e:
                        logger.warning(f"Failed to fetch comments for {submission.id}: {e}")
                        record = _serialize_post(submission, [])
                        new_records.append(record)

            except Exception as e:
                logger.warning(f"Failed search r/{sub_name} query='{query}': {e}")
                continue

    if not dry_run and new_records:
        Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "a") as f:
            for record in new_records:
                f.write(json.dumps(record) + "\n")

    count = len(new_records)
    logger.info(f"Reddit scraper: {count} new posts collected")
    return count
