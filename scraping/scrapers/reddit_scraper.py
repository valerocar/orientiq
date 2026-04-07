"""Reddit scraper using the public JSON API (no credentials required)."""

import json
import logging
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    RAW_DIR,
    REDDIT_COMMENTS_PER_POST,
    REDDIT_POSTS_PER_QUERY,
    REDDIT_QUERIES,
    SUBREDDITS,
)

logger = logging.getLogger(__name__)

BASE = "https://www.reddit.com"
HEADERS = {"User-Agent": "orientiq-research/1.0"}
OUTPUT_FILE = os.path.join(RAW_DIR, "reddit.jsonl")


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


def _search(subreddit, query, limit):
    url = f"{BASE}/r/{subreddit}/search.json"
    params = {"q": query, "limit": limit, "sort": "relevance", "t": "all", "restrict_sr": 1}
    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["data"]["children"]


def _fetch_comments(post_id, limit):
    url = f"{BASE}/comments/{post_id}.json"
    r = requests.get(url, headers=HEADERS, params={"limit": limit, "sort": "best"}, timeout=15)
    r.raise_for_status()
    comments_data = r.json()[1]["data"]["children"]
    comments = []
    for c in comments_data:
        if c.get("kind") != "t1":
            continue
        d = c["data"]
        if d.get("body") in (None, "[deleted]", "[removed]"):
            continue
        comments.append({
            "body": d["body"],
            "score": d.get("score", 0),
            "author": d.get("author", "[deleted]"),
        })
        replies = d.get("replies")
        if not isinstance(replies, dict):
            continue
        for reply in replies.get("data", {}).get("children", []):
            if reply.get("kind") != "t1":
                continue
            rd = reply["data"]
            if rd.get("body") not in (None, "[deleted]", "[removed]"):
                comments.append({
                    "body": rd["body"],
                    "score": rd.get("score", 0),
                    "author": rd.get("author", "[deleted]"),
                })
    return sorted(comments, key=lambda c: c["score"], reverse=True)[:limit]


def scrape_reddit(max_posts=None, dry_run=False):
    existing_ids = _load_existing_ids(OUTPUT_FILE)
    posts_per_query = max_posts or REDDIT_POSTS_PER_QUERY
    new_records = []

    logger.info(
        f"Reddit scraper: {len(SUBREDDITS)} subreddits x {len(REDDIT_QUERIES)} queries, "
        f"limit={posts_per_query}, existing={len(existing_ids)}"
    )

    for sub_name in tqdm(SUBREDDITS, desc="Subreddits"):
        for query in tqdm(REDDIT_QUERIES, desc=f"r/{sub_name}", leave=False):
            try:
                children = _search(sub_name, query, posts_per_query)
                time.sleep(2)  # respect public API rate limit
                for child in children:
                    post = child["data"]
                    post_id = post["id"]
                    if post_id in existing_ids:
                        continue
                    existing_ids.add(post_id)

                    if dry_run:
                        tqdm.write(f"  [DRY] r/{sub_name} | {post_id} | {post['title'][:80]}")
                        new_records.append(None)
                        continue

                    try:
                        comments = _fetch_comments(post_id, REDDIT_COMMENTS_PER_POST)
                        time.sleep(2)
                    except Exception as e:
                        logger.warning(f"Failed to fetch comments for {post_id}: {e}")
                        comments = []

                    record = {
                        "source": "reddit",
                        "subreddit": post.get("subreddit", sub_name),
                        "post_id": post_id,
                        "title": post.get("title", ""),
                        "body": post.get("selftext", ""),
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "url": f"https://reddit.com{post.get('permalink', '')}",
                        "created_utc": post.get("created_utc", 0),
                        "comments": comments,
                    }
                    new_records.append(record)

            except Exception as e:
                wait = 10 if "429" in str(e) else 2
                logger.warning(f"Failed search r/{sub_name} query='{query}': {e} (waiting {wait}s)")
                time.sleep(wait)
                continue

    if not dry_run and new_records:
        Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "a") as f:
            for record in new_records:
                f.write(json.dumps(record) + "\n")

    count = len(new_records)
    logger.info(f"Reddit scraper: {count} new posts collected")
    return count
