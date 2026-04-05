# 3D Print Orientation Pain-Point Research — Data Scraper

## Objective

Build an automated scraping tool that collects community discussions about pain points that print farm operators experience when orienting 3D models for printing. The output is raw data (JSONL) from three platforms, ready for manual analysis.

## Architecture Overview

```
[Scrapers] → [Raw Corpus Store (JSONL)]
```

Three data sources, three output files.

---

## Data Sources

### 1. Reddit (PRAW)

**Target subreddits:**
- r/3Dprinting
- r/FixMyPrint
- r/3dprintingbusiness
- r/prusa3d
- r/BambuLab
- r/resinprinting
- r/functionalprint
- r/slicing

**Search queries (use Reddit search within each subreddit):**
- `orientation`
- `auto-orient`
- `auto orient`
- `support minimization`
- `support removal`
- `support waste`
- `bed packing`
- `nesting`
- `batch printing`
- `print farm workflow`
- `model placement`
- `build plate layout`
- `overhang`
- `surface finish orientation`
- `bridge orientation`
- `print failure orientation`

**What to capture per post:**
```python
{
    "source": "reddit",
    "subreddit": str,
    "post_id": str,
    "title": str,
    "body": str,           # selftext
    "score": int,
    "num_comments": int,
    "url": str,
    "created_utc": float,
    "comments": [           # top-level + second-level only
        {
            "body": str,
            "score": int,
            "author": str
        }
    ]
}
```

**Implementation notes:**
- Use PRAW (Python Reddit API Wrapper)
- Requires Reddit API credentials (client_id, client_secret, user_agent) — store in `.env`
- Rate limit: respect PRAW's built-in rate limiting
- Fetch up to 100 posts per query per subreddit (use `limit=100`)
- For each post, fetch top 20 comments sorted by score (skip deleted/removed)
- Deduplicate posts by `post_id` across queries
- Save raw corpus as JSONL: `data/raw/reddit.jsonl`

### 2. GitHub Issues (GitHub API)

**Target repositories:**
- `prusa3d/PrusaSlicer`
- `SoftFever/OrcaSlicer`
- `Ultimaker/Cura` (check current org, may be `UltiMaker/Cura`)
- `supermerill/SuperSlicer`

**Search approach:**
Use GitHub Search API to find issues containing orientation-related terms:
```
repo:{owner}/{repo} "auto orient" OR "print orientation" OR "support generation" OR "nesting" OR "arrange" OR "bed packing" OR "model placement" OR "overhang angle"
```

Also search specifically for labels if they exist:
- `feature-request`
- `enhancement`
- `auto-orient`
- `arrange`

**What to capture per issue:**
```python
{
    "source": "github",
    "repo": str,
    "issue_number": int,
    "title": str,
    "body": str,
    "state": str,           # open/closed
    "labels": [str],
    "reactions_total": int,  # thumbs up + hooray + heart
    "comments_count": int,
    "url": str,
    "created_at": str,
    "comments": [            # all comments
        {
            "body": str,
            "reactions_total": int,
            "author": str
        }
    ]
}
```

**Implementation notes:**
- Use `requests` with GitHub REST API v3
- Requires GitHub personal access token — store in `.env`
- Rate limit: 30 search requests/minute, 5000 general requests/hour
- Add sleep/backoff logic
- Save raw corpus as JSONL: `data/raw/github.jsonl`

### 3. YouTube Transcripts (yt-dlp)

**Search queries on YouTube:**
- `print farm workflow`
- `print farm orientation`
- `3d print orientation tips`
- `auto orient 3d print`
- `print farm efficiency`
- `batch 3d printing workflow`
- `3d print nesting`
- `slicer auto arrange`
- `print farm day in the life`

**Target channels (prioritize if found):**
- Made with Layers
- 3D Printing Nerd
- Uncle Jessy
- Thomas Sanladerer
- CNC Kitchen
- Slant 3D (actual print farm)
- 3D Musketeers
- Makers Muse

**What to capture per video:**
```python
{
    "source": "youtube",
    "video_id": str,
    "title": str,
    "channel": str,
    "description": str,
    "transcript": str,      # full auto-generated transcript
    "view_count": int,
    "url": str,
    "published_at": str
}
```

**Implementation notes:**
- Use `yt-dlp` to search and extract metadata + subtitles
- Also use `youtube-transcript-api` as primary transcript source, yt-dlp as fallback
- Limit: ~20 videos per search query, ~200 videos total max
- Skip videos shorter than 3 minutes or longer than 60 minutes
- Save raw corpus as JSONL: `data/raw/youtube.jsonl`

---

## Project Structure

```
3dscraping/
├── .env                          # API keys (REDDIT_*, GITHUB_TOKEN)
├── .env.example                  # Template
├── requirements.txt              # praw, requests, yt-dlp, youtube-transcript-api,
│                                 #   python-dotenv, tqdm
├── config.py                     # All search queries, subreddits, repos, channels
├── scrapers/
│   ├── reddit_scraper.py
│   ├── github_scraper.py
│   └── youtube_scraper.py
├── data/
│   └── raw/                      # JSONL from scrapers
└── main.py                       # Orchestrator
```

---

## CLI Interface

```bash
# Run all scrapers
python main.py

# Options
python main.py --skip-youtube          # Skip YouTube (slowest scraper)
python main.py --max-posts 50          # Limit Reddit posts per query
python main.py --dry-run               # Show what would be scraped, don't execute
```

---

## Key Implementation Notes

1. **Idempotency**: Each scraper checks for existing data and skips already-fetched posts/issues/videos. Use source IDs as keys.

2. **Progress tracking**: Use `tqdm` for progress bars. Log each stage to `data/pipeline.log`.

3. **Error resilience**: Wrap all API calls in try/except with exponential backoff. A single failed video transcript shouldn't kill the pipeline.

4. **Rate limits**: Reddit (60 req/min via PRAW), GitHub (30 search/min, 5000 req/hr), YouTube (no strict limit but be polite with 1-2s delays).

5. **Storage**: Everything is local files. No database needed for this scale (~1000-2000 documents max).

6. **Reproducibility**: Save all raw data so analysis can be done separately with any tool of choice.
