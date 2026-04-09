"""
Enhanced pain point analysis from scraped 3D printing data.
Extracts: WTP signals, per-slicer breakdown, feature vs bug classification,
engagement scoring, timeline trends, YouTube view-weighting, workaround patterns.

Outputs: data/output/deep_analysis.json
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime

from analyze import (
    CATEGORIES,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    categorize,
    is_relevant,
    severity,
    user_type,
)

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/output"

SLICER_REPOS = {
    "prusa3d/PrusaSlicer": "PrusaSlicer",
    "SoftFever/OrcaSlicer": "OrcaSlicer",
    "Ultimaker/Cura": "Cura",
    "supermerill/SuperSlicer": "SuperSlicer",
}

WTP_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\$\d+",
        r"would pay",
        r"shut up and take my money",
        r"take my money",
        r"pay for (?:this|it|that)",
        r"worth paying",
        r"premium feature",
        r"subscription",
        r"i['\u2019]?d pay",
    ]
]

WORKAROUND_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"i usually\b.{5,120}",
        r"my workaround\b.{5,120}",
        r"what i do is\b.{5,120}",
        r"i ended up\b.{5,120}",
        r"currently i\b.{5,120}",
        r"as a workaround\b.{5,120}",
        r"i work around\b.{5,120}",
        r"i just\b.{5,80}\binstead\b",
    ]
]

FEATURE_LABELS = {"feature request", "enhancement", "new feature", "type: new feature"}
BUG_LABELS = {"bug", "type: bug", "prerelease :bug:", "upstream bug"}

FEATURE_KEYWORDS = [
    "feature", "enhancement", "request", "wish", "would be nice",
    "would love", "please add", "suggestion",
]
BUG_KEYWORDS = [
    "bug", "broken", "fix", "error", "crash", "regression",
    "not working", "fails", "unexpected",
]


def load_jsonl(filepath):
    records = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return records


def build_full_text(issue):
    title = issue.get("title", "")
    body = issue.get("body", "") or ""
    comments_text = " ".join(c.get("body", "") for c in issue.get("comments", []))
    return f"{title} {body} {comments_text}"


def parse_quarter(date_str, source="github"):
    try:
        if source == "youtube":
            year = int(date_str[:4])
            month = int(date_str[4:6])
        else:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            year = dt.year
            month = dt.month
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    except (ValueError, TypeError):
        return None


def classify_feature_vs_bug(issue):
    labels = {l.lower() for l in issue.get("labels", [])}
    if labels & FEATURE_LABELS:
        return "feature_request"
    if labels & BUG_LABELS:
        return "bug_report"

    text = build_full_text(issue).lower()
    feat_score = sum(1 for kw in FEATURE_KEYWORDS if kw in text)
    bug_score = sum(1 for kw in BUG_KEYWORDS if kw in text)

    if feat_score > bug_score:
        return "feature_request"
    if bug_score > feat_score:
        return "bug_report"
    return "other"


def compute_engagement(issue):
    reactions = issue.get("reactions_total", 0) or 0
    comments = issue.get("comments_count", 0) or 0
    return reactions + comments * 2


def extract_sentences(text):
    return re.split(r"(?<=[.!?])\s+|\n+", text)


def extract_wtp_signals(text, url, title):
    signals = []
    sentences = extract_sentences(text)
    for sent in sentences:
        for pat in WTP_PATTERNS:
            if pat.search(sent):
                quote = sent.strip()[:200]
                if len(quote) > 15:
                    signals.append({
                        "quote": quote,
                        "source_url": url,
                        "source_title": title,
                    })
                break
    return signals


def extract_workarounds(text, url, title):
    results = []
    for pat in WORKAROUND_PATTERNS:
        for match in pat.finditer(text):
            quote = match.group(0).strip()[:200]
            if len(quote) > 20:
                results.append({
                    "quote": quote,
                    "source_url": url,
                    "source_title": title,
                })
    return results


def dedup_workarounds(workarounds):
    groups = defaultdict(list)
    for w in workarounds:
        key = re.sub(r"\s+", " ", w["quote"].lower().strip())[:80]
        groups[key].append(w)

    deduped = []
    for key, items in groups.items():
        best = max(items, key=lambda x: len(x["quote"]))
        deduped.append({
            "quote": best["quote"],
            "source_url": best["source_url"],
            "source_title": best["source_title"],
            "count": len(items),
        })
    deduped.sort(key=lambda x: -x["count"])
    return deduped


def analyze_all():
    # Load raw data
    github_issues = load_jsonl(os.path.join(RAW_DIR, "github.jsonl"))
    youtube_videos = load_jsonl(os.path.join(RAW_DIR, "youtube.jsonl"))

    # Accumulators
    categories = defaultdict(lambda: {
        "count": 0,
        "severity": {"high": 0, "medium": 0, "low": 0},
        "by_slicer": defaultdict(int),
        "engagement_total": 0,
        "youtube_views": 0,
    })
    timeline = defaultdict(lambda: defaultdict(int))
    feature_vs_bug = {"feature_request": 0, "bug_report": 0, "other": 0}
    all_engagement = []
    all_wtp = []
    all_workarounds = []

    github_total = len(github_issues)
    github_relevant = 0
    youtube_total = len(youtube_videos)
    youtube_relevant = 0

    # Process GitHub
    for issue in github_issues:
        full_text = build_full_text(issue)
        if not is_relevant(full_text):
            continue
        github_relevant += 1

        cat = categorize(full_text)
        if not cat:
            continue

        sev = severity(full_text)
        slicer = SLICER_REPOS.get(issue.get("repo", ""), "Unknown")
        engagement = compute_engagement(issue)
        quarter = parse_quarter(issue.get("created_at", ""))
        fvb = classify_feature_vs_bug(issue)

        categories[cat]["count"] += 1
        categories[cat]["severity"][sev] += 1
        categories[cat]["by_slicer"][slicer] += 1
        categories[cat]["engagement_total"] += engagement

        if quarter:
            timeline[quarter][cat] += 1

        feature_vs_bug[fvb] += 1

        all_engagement.append({
            "title": issue.get("title", "")[:80],
            "url": issue.get("url", ""),
            "repo": slicer,
            "reactions": issue.get("reactions_total", 0) or 0,
            "comments": issue.get("comments_count", 0) or 0,
            "engagement": engagement,
            "category": cat,
        })

        url = issue.get("url", "")
        title = issue.get("title", "")
        all_wtp.extend(extract_wtp_signals(full_text, url, title))
        all_workarounds.extend(extract_workarounds(full_text, url, title))

    # Process YouTube
    for video in youtube_videos:
        title = video.get("title", "")
        description = video.get("description", "") or ""
        transcript = video.get("transcript", "") or ""
        full_text = f"{title} {description} {transcript}"

        if not is_relevant(full_text):
            continue
        youtube_relevant += 1

        cat = categorize(full_text)
        if not cat:
            continue

        views = video.get("view_count", 0) or 0
        categories[cat]["youtube_views"] += views

        quarter = parse_quarter(video.get("published_at", ""), source="youtube")
        if quarter:
            timeline[quarter][cat] += 1

        url = video.get("url", "")
        all_wtp.extend(extract_wtp_signals(full_text, url, title))
        all_workarounds.extend(extract_workarounds(full_text, url, title))

    # Build signal scores
    for cat, data in categories.items():
        data["signal_score"] = (
            data["count"] * 3
            + data["severity"]["high"] * 2
            + data["severity"]["medium"] * 1
        )
        # Convert defaultdict to dict for JSON serialization
        data["by_slicer"] = dict(data["by_slicer"])

    # Sort engagement, get top 10
    all_engagement.sort(key=lambda x: -x["engagement"])
    top_engagement = all_engagement[:10]

    # Dedup workarounds
    workarounds_deduped = dedup_workarounds(all_workarounds)

    # Dedup WTP signals by quote prefix
    seen_wtp = set()
    wtp_deduped = []
    for w in all_wtp:
        key = w["quote"].lower()[:60]
        if key not in seen_wtp:
            seen_wtp.add(key)
            wtp_deduped.append(w)

    # Sort timeline keys
    sorted_timeline = dict(sorted(timeline.items()))
    sorted_timeline = {k: dict(v) for k, v in sorted_timeline.items()}

    # YouTube view-weighted summary
    youtube_view_weighted = {
        cat: data["youtube_views"]
        for cat, data in categories.items()
        if data["youtube_views"] > 0
    }

    # Build output
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "github_issues_total": github_total,
            "youtube_videos_total": youtube_total,
            "github_relevant": github_relevant,
            "youtube_relevant": youtube_relevant,
        },
        "categories": {
            cat: dict(data)
            for cat, data in sorted(categories.items(), key=lambda x: -x[1]["signal_score"])
        },
        "timeline": sorted_timeline,
        "feature_vs_bug": feature_vs_bug,
        "top_engagement_issues": top_engagement,
        "wtp_signals": wtp_deduped,
        "workaround_patterns": workarounds_deduped[:30],
        "youtube_view_weighted": dict(sorted(youtube_view_weighted.items(), key=lambda x: -x[1])),
    }

    return output


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running enhanced analysis...")
    output = analyze_all()

    output_path = os.path.join(OUTPUT_DIR, "deep_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    meta = output["metadata"]
    print(f"\nGitHub: {meta['github_issues_total']} total → {meta['github_relevant']} relevant")
    print(f"YouTube: {meta['youtube_videos_total']} total → {meta['youtube_relevant']} relevant")
    print(f"\nCategories ({len(output['categories'])}):")
    for cat, data in output["categories"].items():
        print(f"  {cat}: {data['count']} issues, signal={data['signal_score']}")
    print(f"\nFeature vs Bug: {output['feature_vs_bug']}")
    print(f"WTP signals found: {len(output['wtp_signals'])}")
    print(f"Workaround patterns: {len(output['workaround_patterns'])}")
    print(f"Timeline quarters: {len(output['timeline'])}")
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
