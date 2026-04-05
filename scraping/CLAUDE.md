# OrientIQ — Project Context for Claude

## What This Project Is

OrientIQ is a standalone 3D print orientation optimizer targeting print farm operators. It automatically finds the optimal orientation for 3D models, minimizes support material, and packs build plates for batch printing.

This repo contains two things:
1. **The research pipeline** — scrapers that collected market data to validate the idea
2. **The landing page** — orientiq.io waitlist page

## Project Structure

```
3dscraping/
├── scrapers/               # Data collection (Reddit, GitHub, YouTube)
├── data/
│   ├── raw/                # Scraped JSONL data (not in git)
│   └── output/             # Analysis results and synthesis report
├── orientiq/               # Landing page (deployed to orientiq.io)
│   └── index.html
├── main.py                 # Scraper orchestrator
├── analyze.py              # Pain point extraction and ranking
├── config.py               # Search queries, subreddits, repos, channels
├── requirements.txt        # Python dependencies
└── print-orientation-research-pipeline.md  # Original spec (scraping only)
```

## Research Findings Summary

Analyzed 3,729 GitHub issues + 120 YouTube videos. Top pain points:

| Rank | Category | Issues | Signal |
|------|----------|--------|--------|
| 1 | Support optimization | 851 | 3,734 |
| 2 | Automation trust | 593 | 2,676 |
| 3 | Orientation decision | 471 | 2,054 |
| 4 | Packing & nesting | 400 | 1,667 |
| 5 | Surface quality | 211 | 952 |

Full report: `data/output/synthesis_report.md`

## Target Market

**Primary:** Small-to-medium print farm operators (5-50 printers)
**Secondary:** Professional 3D printing service bureaus

## Current Status

- [x] Market research complete
- [x] Landing page live at orientiq.io
- [ ] Email collection wired up (Google Sheets + Apps Script — next step)
- [ ] Waitlist signups (target: 200)
- [ ] Product development

## Tech Stack

- **Scraper:** Python (PRAW, requests, yt-dlp, youtube-transcript-api)
- **Landing page:** Plain HTML/CSS/JS on GitHub Pages (valerocar/orientiq)
- **Domain:** orientiq.io (Squarespace registrar, DNS → GitHub Pages)

## Running the Scraper

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in GITHUB_TOKEN (REDDIT optional)
python main.py        # full scrape
python main.py --dry-run        # preview only
python main.py --skip-youtube   # skip YouTube
python main.py --max-posts 5    # quick test
```

## Running the Analysis

```bash
python analyze.py
# outputs to data/output/
```
