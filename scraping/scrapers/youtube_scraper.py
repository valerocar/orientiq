"""YouTube scraper using yt-dlp and youtube-transcript-api."""

import json
import logging
import os
import time
from pathlib import Path

from tqdm import tqdm

from config import (
    RAW_DIR,
    YOUTUBE_MAX_DURATION,
    YOUTUBE_MIN_DURATION,
    YOUTUBE_PRIORITY_CHANNELS,
    YOUTUBE_QUERIES,
    YOUTUBE_RESULTS_PER_QUERY,
)

logger = logging.getLogger(__name__)

OUTPUT_FILE = os.path.join(RAW_DIR, "youtube.jsonl")


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
                ids.add(record["video_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _search_videos(query, max_results):
    import yt_dlp

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            result = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        except Exception as e:
            logger.warning(f"yt-dlp search failed for '{query}': {e}")
            return []

    entries = result.get("entries", []) if result else []
    return [e for e in entries if e is not None]


def _filter_video(entry):
    duration = entry.get("duration")
    if duration is None:
        return False
    return YOUTUBE_MIN_DURATION <= duration <= YOUTUBE_MAX_DURATION


def _get_full_metadata(video_id):
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            download=False,
        )
    return info


def _get_transcript(video_id):
    # Primary: youtube-transcript-api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(segment["text"] for segment in transcript_list)
    except Exception as e:
        logger.debug(f"youtube-transcript-api failed for {video_id}: {e}")

    # Fallback: yt-dlp subtitle extraction
    try:
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "json3",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False,
            )
            # Check for subtitles in the info
            subs = info.get("automatic_captions", {}).get("en") or info.get("subtitles", {}).get("en")
            if subs:
                # yt-dlp may include subtitle data directly
                for sub_entry in subs:
                    if sub_entry.get("ext") == "json3" and "url" in sub_entry:
                        import requests
                        resp = requests.get(sub_entry["url"], timeout=15)
                        if resp.ok:
                            sub_data = resp.json()
                            events = sub_data.get("events", [])
                            texts = []
                            for event in events:
                                segs = event.get("segs", [])
                                for seg in segs:
                                    text = seg.get("utf8", "").strip()
                                    if text and text != "\n":
                                        texts.append(text)
                            if texts:
                                return " ".join(texts)
    except Exception as e:
        logger.debug(f"yt-dlp subtitle fallback failed for {video_id}: {e}")

    return None


def _prioritize_results(videos):
    priority_lower = {ch.lower() for ch in YOUTUBE_PRIORITY_CHANNELS}

    def sort_key(v):
        channel = (v.get("channel") or v.get("uploader") or "").lower()
        return 0 if channel in priority_lower else 1

    return sorted(videos, key=sort_key)


def scrape_youtube(dry_run=False):
    existing_ids = _load_existing_ids(OUTPUT_FILE)
    new_records = []

    # Collect unique videos across all queries
    video_map = {}  # video_id -> flat entry

    logger.info(
        f"YouTube scraper: {len(YOUTUBE_QUERIES)} queries, "
        f"max {YOUTUBE_RESULTS_PER_QUERY}/query, existing={len(existing_ids)}"
    )

    for query in tqdm(YOUTUBE_QUERIES, desc="YouTube search"):
        try:
            entries = _search_videos(query, YOUTUBE_RESULTS_PER_QUERY)
            for entry in entries:
                vid = entry.get("id") or entry.get("url", "").split("v=")[-1]
                if not vid or vid in existing_ids or vid in video_map:
                    continue
                if not _filter_video(entry):
                    continue
                video_map[vid] = entry
        except Exception as e:
            logger.warning(f"Failed YouTube search for '{query}': {e}")
            continue

    logger.info(f"YouTube: {len(video_map)} unique new videos found")

    # Prioritize by channel
    video_list = _prioritize_results(list(video_map.values()))

    # Fetch full metadata and transcripts
    for entry in tqdm(video_list, desc="Fetching metadata & transcripts"):
        vid = entry.get("id") or entry.get("url", "").split("v=")[-1]

        if dry_run:
            title = entry.get("title", "unknown")[:80]
            channel = entry.get("channel") or entry.get("uploader") or "unknown"
            tqdm.write(f"  [DRY] {vid} | {channel} | {title}")
            new_records.append(None)
            continue

        try:
            # Get full metadata
            try:
                info = _get_full_metadata(vid)
            except Exception as e:
                logger.warning(f"Failed to get metadata for {vid}: {e}")
                info = entry  # fall back to flat entry data

            # Get transcript
            transcript = _get_transcript(vid)

            record = {
                "source": "youtube",
                "video_id": vid,
                "title": info.get("title", entry.get("title", "")),
                "channel": info.get("channel") or info.get("uploader") or entry.get("channel", ""),
                "description": info.get("description", "") or "",
                "transcript": transcript,
                "view_count": info.get("view_count") or entry.get("view_count"),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "published_at": info.get("upload_date", ""),
            }
            new_records.append(record)

        except Exception as e:
            logger.warning(f"Failed to process video {vid}: {e}")
            continue

        time.sleep(1.5)  # Politeness delay

    if not dry_run and new_records:
        Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "a") as f:
            for record in new_records:
                f.write(json.dumps(record) + "\n")

    count = len(new_records)
    logger.info(f"YouTube scraper: {count} new videos collected")
    return count
