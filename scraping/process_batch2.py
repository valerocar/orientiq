#!/usr/bin/env python3
"""
Process GitHub issues lines 1866-3729 for pain points about:
- Model orientation
- Support structures
- Bed/build plate packing/nesting
- Auto-orient
- Surface finish from orientation
- Print failures from orientation
- Time on manual orientation
- Batch printing workflow
"""

import json
import re
import sys
from pathlib import Path

INPUT_FILE = "/Users/valeroc/Dropbox/saas/3dscraping/data/raw/github.jsonl"
OUTPUT_FILE = "/Users/valeroc/Dropbox/saas/3dscraping/data/output/pain_points_github_2.jsonl"
PROGRESS_FILE = "/Users/valeroc/Dropbox/saas/3dscraping/data/progress.txt"

START_LINE = 1866  # 1-indexed
END_LINE = 3729

# Keywords for relevance detection
ORIENTATION_KEYWORDS = [
    'orient', 'rotation', 'rotate', 'tilt', 'angle', 'flip', 'upside', 'face down',
    'face up', 'laying', 'standing', 'auto-orient', 'auto orient', 'autorotate',
    'auto-rotate', 'placement', 'position', 'place on bed', 'lay flat', 'lay it flat',
    'optimal orientation', 'best orientation', 'print orientation', 'model orientation',
    'part orientation', 'object orientation', 'overhang', 'bridge', 'bridging',
    'bottom surface', 'top surface', 'z-axis', 'print direction'
]

SUPPORT_KEYWORDS = [
    'support', 'supports', 'support structure', 'support material', 'tree support',
    'organic support', 'support removal', 'remove support', 'detach support',
    'support interface', 'support density', 'support generation', 'auto support',
    'support placement', 'support optimization', 'support contact', 'support cleanup',
    'support scarring', 'support marks', 'support leftover', 'support adhesion',
    'support threshold', 'overhang angle', 'overhang threshold'
]

NESTING_KEYWORDS = [
    'nest', 'nesting', 'pack', 'packing', 'arrange', 'arrangement', 'layout',
    'plate', 'build plate', 'bed', 'batch', 'batch print', 'multiple parts',
    'fill plate', 'plate fill', 'auto arrange', 'auto-arrange', 'auto layout',
    'plate layout', 'bin packing', 'fit more', 'maximize plate', 'plate efficiency',
    'plate utilization', 'copy to plate', 'multi-plate', 'multi plate'
]

SURFACE_KEYWORDS = [
    'surface finish', 'surface quality', 'layer lines', 'layer artifact',
    'stair stepping', 'staircase', 'visible layers', 'smooth surface',
    'print quality', 'cosmetic surface', 'finish quality', 'seam', 'scarring',
    'blemish', 'artifact', 'aesthetic', 'appearance'
]

WORKFLOW_KEYWORDS = [
    'time consuming', 'tedious', 'manual', 'workflow', 'automate', 'automation',
    'batch', 'queue', 'farm', 'print farm', 'production', 'efficiency', 'speed up',
    'repeat', 'repetitive', 'every time', 'each time', 'one by one', 'hours',
    'wasted time', 'time wasted', 'slow', 'annoying', 'frustrating', 'pain'
]

ALL_KEYWORDS = (ORIENTATION_KEYWORDS + SUPPORT_KEYWORDS + NESTING_KEYWORDS +
                SURFACE_KEYWORDS + WORKFLOW_KEYWORDS)

def is_relevant(text):
    """Check if text contains any relevant keywords."""
    text_lower = text.lower()
    for kw in ALL_KEYWORDS:
        if kw in text_lower:
            return True
    return False

def get_all_text(issue):
    """Get combined text from issue title, body, and comments."""
    parts = []
    if issue.get('title'):
        parts.append(issue['title'])
    if issue.get('body'):
        parts.append(issue['body'])
    for comment in issue.get('comments', []):
        if comment.get('body'):
            parts.append(comment['body'])
    return ' '.join(parts)

def categorize_pain_point(text):
    """Determine category of pain point."""
    text_lower = text.lower()

    if any(kw in text_lower for kw in NESTING_KEYWORDS):
        return 'packing_nesting'
    if any(kw in text_lower for kw in ['auto-orient', 'auto orient', 'autorotate', 'auto-rotate', 'auto orient', 'optimal orientation']):
        return 'automation_trust'
    if any(kw in text_lower for kw in SURFACE_KEYWORDS[:8]):
        return 'surface_quality'
    if any(kw in text_lower for kw in ['support removal', 'remove support', 'detach support', 'support scarring', 'support marks', 'support leftover']):
        return 'support_optimization'
    if any(kw in text_lower for kw in ['support generation', 'auto support', 'support placement', 'support optimization', 'support threshold', 'overhang angle']):
        return 'support_optimization'
    if any(kw in text_lower for kw in SUPPORT_KEYWORDS):
        return 'support_optimization'
    if any(kw in text_lower for kw in ORIENTATION_KEYWORDS):
        return 'orientation_decision'
    if any(kw in text_lower for kw in ['time consuming', 'tedious', 'manual', 'workflow', 'batch', 'farm']):
        return 'workflow_speed'
    return 'other'

def get_severity(text, reactions):
    """Estimate severity from language and reactions."""
    text_lower = text.lower()
    high_indicators = [
        'critical', 'impossible', 'broken', 'completely', 'always fails', 'never works',
        'hours', 'wasted', 'ruined', 'destroyed', 'unusable', 'terrible', 'horrible',
        'frustrating', 'infuriating', 'disaster', 'nightmare', 'awful', 'pain point',
        'major issue', 'huge problem', 'significant', 'failure', 'failed print',
        'wasted material', 'wasted filament', 'much time', 'lot of time'
    ]
    medium_indicators = [
        'annoying', 'tedious', 'slow', 'difficult', 'hard to', 'problem', 'issue',
        'workaround', 'manual', 'inefficient', 'suboptimal', 'not ideal', 'could be better',
        'would be nice', 'improvement', 'enhancement', 'better if', 'should be',
        'missing feature', 'lack of', 'no option', 'unable to'
    ]

    if reactions and reactions >= 10:
        return 'high'
    if any(ind in text_lower for ind in high_indicators):
        return 'high'
    if reactions and reactions >= 3:
        return 'medium'
    if any(ind in text_lower for ind in medium_indicators):
        return 'medium'
    return 'low'

def get_user_type(text):
    """Estimate user type from context clues."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in ['print farm', 'farm', 'production run', 'many printers', 'multiple printers', 'batch production', 'business', 'clients', 'customers', 'commercial']):
        if any(kw in text_lower for kw in ['large', 'many', 'hundreds', 'thousands', 'fleet']):
            return 'large_farm'
        return 'small_farm'
    if any(kw in text_lower for kw in ['professional', 'service bureau', 'printing service', 'print service', 'for clients', 'customer parts']):
        return 'professional_service'
    if any(kw in text_lower for kw in ['hobby', 'hobbyist', 'home printer', 'personal', 'fun project', 'home use']):
        return 'hobbyist'
    return 'unknown'

def extract_quote(text, keyword, max_words=20):
    """Extract a brief quote around a keyword."""
    text_lower = text.lower()
    idx = text_lower.find(keyword.lower())
    if idx == -1:
        return text[:100].strip()

    # Get surrounding context
    start = max(0, idx - 50)
    end = min(len(text), idx + len(keyword) + 100)
    snippet = text[start:end].strip()

    # Trim to roughly max_words words
    words = snippet.split()
    if len(words) > max_words:
        snippet = ' '.join(words[:max_words])

    return snippet.replace('\n', ' ').strip()

def find_workaround(text):
    """Extract any workaround mentioned."""
    text_lower = text.lower()
    workaround_patterns = [
        r'workaround[:\s]+([^.!?\n]+)',
        r'as a workaround[,\s]+([^.!?\n]+)',
        r'instead[,\s]+(?:i |you can |we )?([^.!?\n]+)',
        r'manually ([^.!?\n]+)',
        r'i (rotate|orient|flip|arrange|place|move)[^.!?\n]+',
        r'by (rotating|orienting|flipping|arranging|placing|moving)[^.!?\n]+'
    ]
    for pattern in workaround_patterns:
        match = re.search(pattern, text_lower)
        if match:
            snippet = text[match.start():match.end()].strip()
            words = snippet.split()
            if len(words) > 15:
                snippet = ' '.join(words[:15])
            return snippet
    return None

def extract_pain_points(issue):
    """Extract pain points from a single issue."""
    pain_points = []

    title = issue.get('title', '')
    body = issue.get('body', '') or ''
    comments = issue.get('comments', []) or []
    url = issue.get('url', '')
    repo = issue.get('repo', '')
    reactions = issue.get('reactions_total', 0) or 0

    all_text = get_all_text(issue)

    if not is_relevant(all_text):
        return pain_points

    # Check for specific pain point patterns and extract them

    # --- SUPPORT PAIN POINTS ---

    # Support generation issues
    if any(kw in all_text.lower() for kw in ['support generation', 'generate support', 'support placement', 'support algorithm', 'auto support', 'support detection']):
        if any(kw in all_text.lower() for kw in ['wrong', 'incorrect', 'bad', 'poor', 'unnecessary', 'missing', 'not generating', 'fails', 'problem', 'issue', 'bug', 'broken', 'inaccurate', 'too much', 'too many', 'not enough', 'lacks', 'improve', 'better', 'request', 'wish']):
            quote_kw = next((kw for kw in ['support generation', 'generate support', 'auto support'] if kw in all_text.lower()), 'support')
            pp = {
                "description": f"Support generation issues: {title}",
                "category": "support_optimization",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, quote_kw),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # Support removal difficulty
    if any(kw in all_text.lower() for kw in ['support removal', 'remove support', 'removing support', 'detach support', 'pull off support', 'break off support']):
        pp = {
            "description": f"Support removal difficulty: {title}",
            "category": "support_optimization",
            "severity": get_severity(all_text, reactions),
            "user_type": get_user_type(all_text),
            "source_quote": extract_quote(all_text, 'support removal' if 'support removal' in all_text.lower() else 'remove support'),
            "workaround": find_workaround(all_text),
            "source_url": url,
            "source_score": reactions,
            "repo": repo
        }
        pain_points.append(pp)

    # Support quality/scarring
    if any(kw in all_text.lower() for kw in ['support scar', 'support mark', 'support artifact', 'support interface', 'support contact', 'support adhesion']):
        if any(kw in all_text.lower() for kw in ['scar', 'mark', 'damage', 'ruin', 'ugly', 'poor', 'bad', 'improve', 'better', 'issue', 'problem']):
            pp = {
                "description": f"Support leaving marks/scars on surface: {title}",
                "category": "surface_quality",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, 'support'),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # Tree support issues
    if 'tree support' in all_text.lower() or 'organic support' in all_text.lower():
        if any(kw in all_text.lower() for kw in ['problem', 'issue', 'bug', 'fail', 'broken', 'improve', 'better', 'wrong', 'incorrect', 'request', 'wish', 'poor', 'bad', 'slow', 'too slow', 'crash']):
            pp = {
                "description": f"Tree/organic support issues: {title}",
                "category": "support_optimization",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, 'tree support' if 'tree support' in all_text.lower() else 'organic support'),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # Overhang/support threshold issues
    if any(kw in all_text.lower() for kw in ['overhang angle', 'overhang threshold', 'support threshold', 'support angle']):
        pp = {
            "description": f"Overhang/support threshold configuration issues: {title}",
            "category": "support_optimization",
            "severity": get_severity(all_text, reactions),
            "user_type": get_user_type(all_text),
            "source_quote": extract_quote(all_text, 'overhang' if 'overhang' in all_text.lower() else 'support threshold'),
            "workaround": find_workaround(all_text),
            "source_url": url,
            "source_score": reactions,
            "repo": repo
        }
        pain_points.append(pp)

    # --- ORIENTATION PAIN POINTS ---

    # Auto-orient feature requests/issues
    if any(kw in all_text.lower() for kw in ['auto-orient', 'auto orient', 'autorotate', 'auto-rotate', 'auto rotation', 'automatic orientation', 'automatic orient']):
        pp = {
            "description": f"Auto-orient feature request or issue: {title}",
            "category": "automation_trust",
            "severity": get_severity(all_text, reactions),
            "user_type": get_user_type(all_text),
            "source_quote": extract_quote(all_text, 'auto'),
            "workaround": find_workaround(all_text),
            "source_url": url,
            "source_score": reactions,
            "repo": repo
        }
        pain_points.append(pp)

    # Manual orientation time/difficulty
    if any(kw in all_text.lower() for kw in ['orient', 'rotation', 'rotate', 'flip', 'tilt', 'place on bed', 'lay flat']):
        if any(kw in all_text.lower() for kw in ['time consuming', 'tedious', 'manually', 'manual', 'every time', 'each time', 'one by one', 'annoying', 'frustrating', 'difficult', 'hard to', 'hours']):
            pp = {
                "description": f"Manual orientation process is time-consuming: {title}",
                "category": "workflow_speed",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, 'orient' if 'orient' in all_text.lower() else 'rotate'),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # Orientation affecting print quality
    if any(kw in all_text.lower() for kw in ['orient', 'rotation', 'rotate']):
        if any(kw in all_text.lower() for kw in ['quality', 'surface', 'finish', 'layer line', 'layer artifact', 'visible', 'smooth', 'rough', 'appearance', 'aesthetic']):
            pp = {
                "description": f"Orientation affects surface quality/finish: {title}",
                "category": "surface_quality",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, 'orient' if 'orient' in all_text.lower() else 'rotate'),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # Print failure due to orientation
    if any(kw in all_text.lower() for kw in ['orient', 'rotation', 'rotate', 'tilt', 'angle']):
        if any(kw in all_text.lower() for kw in ['fail', 'failed', 'failure', 'collapse', 'fell', 'topple', 'knocked', 'warped', 'warp']):
            pp = {
                "description": f"Print failure caused by orientation choice: {title}",
                "category": "failure_recovery",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, 'fail' if 'fail' in all_text.lower() else 'orient'),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # --- NESTING/PACKING PAIN POINTS ---

    # Auto arrange / nesting
    if any(kw in all_text.lower() for kw in ['auto arrange', 'auto-arrange', 'arrange', 'nesting', 'nest', 'pack', 'packing', 'plate layout', 'bin packing']):
        if any(kw in all_text.lower() for kw in ['problem', 'issue', 'request', 'improve', 'better', 'wrong', 'incorrect', 'fails', 'slow', 'not working', 'feature', 'wish', 'want', 'need', 'missing', 'lacks', 'poor', 'bad', 'inefficient', 'overlap', 'collision']):
            pp = {
                "description": f"Bed arrangement/nesting issues: {title}",
                "category": "packing_nesting",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, 'arrange' if 'arrange' in all_text.lower() else 'nest'),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # Batch printing workflow
    if any(kw in all_text.lower() for kw in ['batch print', 'batch printing', 'print farm', 'farm printing', 'multiple plates', 'multi-plate', 'sequential printing', 'production workflow']):
        pp = {
            "description": f"Batch/farm printing workflow friction: {title}",
            "category": "workflow_speed",
            "severity": get_severity(all_text, reactions),
            "user_type": get_user_type(all_text),
            "source_quote": extract_quote(all_text, 'batch' if 'batch' in all_text.lower() else 'farm'),
            "workaround": find_workaround(all_text),
            "source_url": url,
            "source_score": reactions,
            "repo": repo
        }
        pain_points.append(pp)

    # Copy to plate / duplicate issues
    if any(kw in all_text.lower() for kw in ['copy to plate', 'duplicate', 'copies', 'multiple copies', 'fill bed', 'fill plate', 'replicate', 'instance']):
        if any(kw in all_text.lower() for kw in ['problem', 'issue', 'request', 'improve', 'better', 'wrong', 'fails', 'feature', 'wish', 'want', 'need', 'missing', 'lacks', 'inefficient', 'overlap', 'collision', 'not working']):
            pp = {
                "description": f"Copying/duplicating parts to fill print bed: {title}",
                "category": "packing_nesting",
                "severity": get_severity(all_text, reactions),
                "user_type": get_user_type(all_text),
                "source_quote": extract_quote(all_text, 'copy' if 'copy' in all_text.lower() else 'duplicate'),
                "workaround": find_workaround(all_text),
                "source_url": url,
                "source_score": reactions,
                "repo": repo
            }
            pain_points.append(pp)

    # Remove duplicates based on description+url
    seen = set()
    unique_pp = []
    for pp in pain_points:
        key = (pp['description'][:50], pp['source_url'])
        if key not in seen:
            seen.add(key)
            unique_pp.append(pp)

    return unique_pp


def main():
    pain_points_all = []
    processed = 0
    total = END_LINE - START_LINE + 1

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if line_num < START_LINE:
                continue
            if line_num > END_LINE:
                break

            line = line.strip()
            if not line:
                continue

            try:
                issue = json.loads(line)
                pps = extract_pain_points(issue)
                pain_points_all.extend(pps)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}", file=sys.stderr)

            processed += 1

            # Update progress every 100 issues
            if processed % 100 == 0:
                progress_msg = f"GitHub batch 2: {processed}/{total} issues processed, {len(pain_points_all)} pain points found\n"
                with open(PROGRESS_FILE, 'w') as pf:
                    pf.write(f"Analysis started at Sat Apr  4 18:39:04 CST 2026\n")
                    pf.write(progress_msg)
                print(progress_msg.strip())

    # Write output
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for pp in pain_points_all:
            out.write(json.dumps(pp, ensure_ascii=False) + '\n')

    # Final progress update
    final_msg = f"GitHub batch 2: {processed}/{total} issues processed, {len(pain_points_all)} pain points found\n"
    with open(PROGRESS_FILE, 'w') as pf:
        pf.write(f"Analysis started at Sat Apr  4 18:39:04 CST 2026\n")
        pf.write(final_msg)

    print(f"\nDone! Processed {processed} issues, found {len(pain_points_all)} pain points.")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
