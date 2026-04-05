"""
Pain point extraction from scraped 3D printing data.
Uses keyword matching + heuristics to categorize issues.
No LLM needed — fast local processing.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

RAW_DIR = "data/raw"
OUTPUT_DIR = "data/output"
PROGRESS_FILE = "data/progress.txt"

# Keywords by category
CATEGORIES = {
    "orientation_decision": [
        "orient", "orientation", "rotate", "rotation", "tilt", "angle",
        "auto-orient", "auto orient", "autoorient", "face down", "face up",
        "which way", "direction", "axis", "flat", "upright", "standing",
    ],
    "support_optimization": [
        "support", "supports", "support structure", "support removal",
        "support waste", "support material", "tree support", "organic support",
        "support interface", "support density", "overhang", "bridge",
        "bridging", "drooping", "sagging",
    ],
    "packing_nesting": [
        "pack", "packing", "nest", "nesting", "arrange", "arrangement",
        "bed layout", "build plate", "plate layout", "batch", "multiple parts",
        "fill plate", "plate packing", "bed packing", "placement", "fit more",
    ],
    "automation_trust": [
        "auto", "automatic", "automatically", "algorithm", "unreliable",
        "doesn't work", "wrong result", "bad result", "trust", "manual",
        "have to do it myself", "not good enough", "poor quality",
        "inaccurate", "incorrect",
    ],
    "workflow_speed": [
        "slow", "time consuming", "hours", "takes too long", "tedious",
        "repetitive", "manual work", "workflow", "productivity", "efficiency",
        "farm", "print farm", "batch", "scale", "volume",
    ],
    "surface_quality": [
        "surface", "finish", "quality", "layer lines", "visible", "aesthetic",
        "cosmetic", "smooth", "rough", "appearance", "marks", "striation",
    ],
    "failure_recovery": [
        "fail", "failure", "failed", "crash", "warping", "warp", "detach",
        "adhesion", "spaghetti", "reprint", "waste", "ruined", "error",
    ],
    "multi_material": [
        "multi material", "multi-material", "mmu", "ams", "purge",
        "color change", "filament change", "dual", "two color",
    ],
    "resin_specific": [
        "resin", "msla", "sla", "dlp", "uv", "fep", "suction", "hollow",
        "drain", "exposure",
    ],
}

SEVERITY_HIGH = [
    "hours", "days", "nightmare", "terrible", "awful", "horrible", "hate",
    "broken", "useless", "waste", "frustrat", "impossible", "critical",
    "urgent", "please fix", "please add", "desperately", "badly needed",
    "money", "cost", "expensive", "$",
]

SEVERITY_MEDIUM = [
    "annoying", "inconvenient", "difficult", "hard", "problem", "issue",
    "request", "would be nice", "wish", "could", "should", "improve",
    "better", "missing", "lack",
]

USER_TYPES = {
    "large_farm": ["print farm", "print service", "service bureau", "hundreds", "thousands", "mass production", "production run"],
    "small_farm": ["small farm", "few printers", "multiple printers", "side business"],
    "professional_service": ["professional", "commercial", "client", "customer", "business"],
    "hobbyist": ["hobby", "hobbyist", "personal", "home", "enthusiast"],
}


def categorize(text):
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def severity(text):
    text_lower = text.lower()
    if any(kw in text_lower for kw in SEVERITY_HIGH):
        return "high"
    if any(kw in text_lower for kw in SEVERITY_MEDIUM):
        return "medium"
    return "low"


def user_type(text):
    text_lower = text.lower()
    for utype, keywords in USER_TYPES.items():
        if any(kw in text_lower for kw in keywords):
            return utype
    return "unknown"


def extract_quote(text, max_words=20):
    # Find the most relevant sentence
    sentences = re.split(r'[.!?]\s+', text)
    best = ""
    best_score = 0
    all_keywords = [kw for kws in CATEGORIES.values() for kw in kws]
    for sent in sentences:
        score = sum(1 for kw in all_keywords if kw in sent.lower())
        if score > best_score and 5 < len(sent.split()) < 60:
            best_score = score
            best = sent
    words = best.split()
    return " ".join(words[:max_words]) if words else ""


def is_relevant(text):
    text_lower = text.lower()
    all_keywords = [kw for kws in CATEGORIES.values() for kw in kws]
    return sum(1 for kw in all_keywords if kw in text_lower) >= 2


def process_github():
    filepath = os.path.join(RAW_DIR, "github.jsonl")
    output_file = os.path.join(OUTPUT_DIR, "pain_points_github.jsonl")
    pain_points = []
    total = 0
    relevant = 0

    with open(filepath, "r") as f:
        lines = f.readlines()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as out:
        for i, line in enumerate(lines):
            try:
                issue = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            total += 1

            # Build full text
            title = issue.get("title", "")
            body = issue.get("body", "") or ""
            comments_text = " ".join(c.get("body", "") for c in issue.get("comments", []))
            full_text = f"{title} {body} {comments_text}"

            if not is_relevant(full_text):
                continue

            relevant += 1
            cat = categorize(full_text)
            if not cat:
                continue

            record = {
                "description": title,
                "category": cat,
                "severity": severity(full_text),
                "user_type": user_type(full_text),
                "source_quote": extract_quote(full_text),
                "workaround": None,
                "source_url": issue.get("url", ""),
                "source_score": issue.get("reactions_total", 0),
                "repo": issue.get("repo", ""),
                "source": "github",
            }
            out.write(json.dumps(record) + "\n")
            pain_points.append(record)

            if (i + 1) % 100 == 0:
                with open(PROGRESS_FILE, "a") as pf:
                    pf.write(f"GitHub: {i+1}/{len(lines)} processed, {relevant} relevant\n")

    return total, relevant, len(pain_points)


def process_youtube():
    filepath = os.path.join(RAW_DIR, "youtube.jsonl")
    output_file = os.path.join(OUTPUT_DIR, "pain_points_youtube.jsonl")
    pain_points = []
    total = 0
    relevant = 0

    with open(filepath, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as out:
        for i, line in enumerate(lines):
            try:
                video = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            total += 1
            title = video.get("title", "")
            description = video.get("description", "") or ""
            transcript = video.get("transcript", "") or ""
            full_text = f"{title} {description} {transcript}"

            if not is_relevant(full_text):
                continue

            relevant += 1
            cat = categorize(full_text)
            if not cat:
                continue

            record = {
                "description": title,
                "category": cat,
                "severity": severity(full_text),
                "user_type": user_type(full_text),
                "source_quote": extract_quote(full_text),
                "workaround": None,
                "source_url": video.get("url", ""),
                "source_score": video.get("view_count", 0),
                "channel": video.get("channel", ""),
                "source": "youtube",
            }
            out.write(json.dumps(record) + "\n")
            pain_points.append(record)

    with open(PROGRESS_FILE, "a") as pf:
        pf.write(f"YouTube: {total} videos, {relevant} relevant\n")

    return total, relevant, len(pain_points)


def merge_and_rank():
    """Load all pain points, deduplicate by title similarity, rank by score."""
    all_points = []

    for fname in ["pain_points_github.jsonl", "pain_points_youtube.jsonl"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            for line in f:
                try:
                    all_points.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    # Group by category
    by_category = {}
    for p in all_points:
        cat = p.get("category", "other")
        by_category.setdefault(cat, []).append(p)

    # Build ranked summary
    summary = []
    for cat, points in sorted(by_category.items(), key=lambda x: -len(x[1])):
        high = sum(1 for p in points if p.get("severity") == "high")
        medium = sum(1 for p in points if p.get("severity") == "medium")
        top_sources = sorted(points, key=lambda p: p.get("source_score", 0) or 0, reverse=True)[:3]

        summary.append({
            "category": cat,
            "count": len(points),
            "high_severity_count": high,
            "medium_severity_count": medium,
            "signal_score": len(points) * 3 + high * 2 + medium * 1,
            "top_sources": [
                {"url": p.get("source_url"), "description": p.get("description"), "score": p.get("source_score")}
                for p in top_sources
            ],
            "repos": list(set(p.get("repo", "") for p in points if p.get("repo"))),
        })

    summary.sort(key=lambda x: -x["signal_score"])
    for i, s in enumerate(summary):
        s["rank"] = i + 1

    output = {
        "metadata": {
            "total_pain_points": len(all_points),
            "total_relevant": len(all_points),
            "sources_breakdown": {
                "github": sum(1 for p in all_points if p.get("source") == "github"),
                "youtube": sum(1 for p in all_points if p.get("source") == "youtube"),
            },
            "generated_at": datetime.now().isoformat(),
        },
        "ranked_categories": summary,
    }

    output_path = os.path.join(OUTPUT_DIR, "ranked_pain_points.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"\n=== Analysis started {datetime.now().isoformat()} ===\n")

    print("Processing GitHub issues...")
    g_total, g_relevant, g_points = process_github()
    print(f"  GitHub: {g_total} issues → {g_relevant} relevant → {g_points} pain points")

    print("Processing YouTube transcripts...")
    y_total, y_relevant, y_points = process_youtube()
    print(f"  YouTube: {y_total} videos → {y_relevant} relevant → {y_points} pain points")

    print("Merging and ranking...")
    output = merge_and_rank()
    meta = output["metadata"]
    print(f"  Total pain points: {meta['total_pain_points']}")
    print(f"  GitHub: {meta['sources_breakdown']['github']}, YouTube: {meta['sources_breakdown']['youtube']}")
    print(f"\nTop categories:")
    for cat in output["ranked_categories"][:5]:
        print(f"  [{cat['rank']}] {cat['category']}: {cat['count']} issues (signal: {cat['signal_score']})")

    print(f"\nResults written to {OUTPUT_DIR}/")
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"=== Analysis complete {datetime.now().isoformat()} ===\n")


if __name__ == "__main__":
    main()
