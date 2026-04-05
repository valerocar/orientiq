"""GitHub Issues scraper using the REST API."""

import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from config import (
    GITHUB_LABELS,
    GITHUB_REPOS,
    GITHUB_SEARCH_TERMS,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

OUTPUT_FILE = os.path.join(RAW_DIR, "github.jsonl")

API_BASE = "https://api.github.com"

# Track search requests for 30/min secondary rate limit
_search_timestamps = deque(maxlen=30)


def _init_session():
    load_dotenv()
    token = os.environ["GITHUB_TOKEN"]
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    })
    return session


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
                ids.add(f"{record['repo']}#{record['issue_number']}")
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _rate_limit_wait(response, is_search=False):
    remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
    reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))

    if remaining < 5 and reset_ts:
        wait = max(reset_ts - time.time() + 1, 0)
        if wait > 0:
            logger.info(f"GitHub rate limit low ({remaining} left), waiting {wait:.0f}s")
            time.sleep(wait)

    if is_search:
        now = time.time()
        if len(_search_timestamps) == 30 and (now - _search_timestamps[0]) < 60:
            wait = 60 - (now - _search_timestamps[0]) + 0.5
            logger.info(f"GitHub search rate limit, waiting {wait:.1f}s")
            time.sleep(wait)
        _search_timestamps.append(time.time())


def _request_with_retry(session, method, url, max_retries=3, is_search=False, **kwargs):
    for attempt in range(max_retries + 1):
        resp = session.request(method, url, **kwargs)
        _rate_limit_wait(resp, is_search=is_search)

        if resp.status_code in (403, 429) and attempt < max_retries:
            delay = (2 ** attempt)
            logger.warning(f"GitHub {resp.status_code} on {url}, retry in {delay}s")
            time.sleep(delay)
            continue

        resp.raise_for_status()
        return resp

    resp.raise_for_status()
    return resp


def _search_issues(session, repo, term):
    """Search for issues in a repo matching a term."""
    query = f"{term} repo:{repo} type:issue"
    issues = []
    page = 1

    max_results = 500  # GitHub caps at 1000; stay well under to avoid 422s
    while True:
        resp = _request_with_retry(
            session, "GET", f"{API_BASE}/search/issues",
            params={"q": query, "per_page": 100, "page": page},
            is_search=True,
        )
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        issues.extend(items)
        if len(issues) >= max_results or len(issues) >= data.get("total_count", 0) or len(items) < 100:
            break
        page += 1

    return issues


def _search_by_label(session, repo, label):
    """Fetch issues with a specific label."""
    issues = []
    page = 1

    while True:
        try:
            resp = _request_with_retry(
                session, "GET", f"{API_BASE}/repos/{repo}/issues",
                params={"labels": label, "state": "all", "per_page": 100, "page": page},
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.warning(f"Label '{label}' not found in {repo}")
                return []
            raise
        items = resp.json()
        if not items:
            break
        issues.extend(items)
        if len(items) < 100:
            break
        page += 1

    return issues


def _fetch_comments(session, repo, issue_number):
    """Fetch all comments for an issue."""
    comments = []
    page = 1

    while True:
        resp = _request_with_retry(
            session, "GET",
            f"{API_BASE}/repos/{repo}/issues/{issue_number}/comments",
            params={"per_page": 100, "page": page},
        )
        items = resp.json()
        if not items:
            break
        for c in items:
            reactions = c.get("reactions", {})
            comments.append({
                "body": c.get("body", ""),
                "reactions_total": (
                    reactions.get("+1", 0)
                    + reactions.get("hooray", 0)
                    + reactions.get("heart", 0)
                ),
                "author": c.get("user", {}).get("login", "unknown"),
            })
        if len(items) < 100:
            break
        page += 1

    return comments


def _compute_reactions(issue):
    reactions = issue.get("reactions", {})
    return (
        reactions.get("+1", 0)
        + reactions.get("hooray", 0)
        + reactions.get("heart", 0)
    )


def _serialize_issue(issue, comments, repo):
    return {
        "source": "github",
        "repo": repo,
        "issue_number": issue["number"],
        "title": issue.get("title", ""),
        "body": issue.get("body", "") or "",
        "state": issue.get("state", "unknown"),
        "labels": [l["name"] for l in issue.get("labels", [])],
        "reactions_total": _compute_reactions(issue),
        "comments_count": issue.get("comments", 0),
        "url": issue.get("html_url", ""),
        "created_at": issue.get("created_at", ""),
        "comments": comments,
    }


def scrape_github(dry_run=False):
    session = _init_session()
    existing_ids = _load_existing_ids(OUTPUT_FILE)
    new_records = []

    # Collect unique issues from both search strategies
    issue_map = {}  # "repo#number" -> issue data

    logger.info(
        f"GitHub scraper: {len(GITHUB_REPOS)} repos, "
        f"{len(GITHUB_SEARCH_TERMS)} terms, {len(GITHUB_LABELS)} labels, "
        f"existing={len(existing_ids)}"
    )

    # Term-based search
    for repo in tqdm(GITHUB_REPOS, desc="GitHub repos (term search)"):
        for term in tqdm(GITHUB_SEARCH_TERMS, desc=f"{repo}", leave=False):
            try:
                issues = _search_issues(session, repo, term)
                for issue in issues:
                    # Skip pull requests returned by search
                    if "pull_request" in issue:
                        continue
                    key = f"{repo}#{issue['number']}"
                    if key not in existing_ids and key not in issue_map:
                        issue_map[key] = (repo, issue)
            except Exception as e:
                logger.warning(f"Failed search {repo} term={term}: {e}")
                continue

    # Label-based search
    for repo in tqdm(GITHUB_REPOS, desc="GitHub repos (label search)"):
        for label in tqdm(GITHUB_LABELS, desc=f"{repo}", leave=False):
            try:
                issues = _search_by_label(session, repo, label)
                for issue in issues:
                    if "pull_request" in issue:
                        continue
                    key = f"{repo}#{issue['number']}"
                    if key not in existing_ids and key not in issue_map:
                        issue_map[key] = (repo, issue)
            except Exception as e:
                logger.warning(f"Failed label search {repo} label={label}: {e}")
                continue

    logger.info(f"GitHub: {len(issue_map)} unique new issues found, fetching comments...")

    # Fetch comments and serialize — write incrementally so progress survives interrupts
    Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    count = 0
    for key, (repo, issue) in tqdm(issue_map.items(), desc="Fetching comments"):
        if dry_run:
            tqdm.write(f"  [DRY] {key} | {issue.get('title', '')[:80]}")
            count += 1
            continue

        try:
            comments = _fetch_comments(session, repo, issue["number"])
            record = _serialize_issue(issue, comments, repo)
        except Exception as e:
            logger.warning(f"Failed to fetch comments for {key}: {e}")
            record = _serialize_issue(issue, [], repo)

        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        existing_ids.add(key)
        count += 1

    logger.info(f"GitHub scraper: {count} new issues collected")
    return count
