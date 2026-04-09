"""
Generate a standalone HTML report from deep_analysis.json.
Uses Plotly for interactive charts, dark theme matching OrientIQ brand.

Outputs: data/output/deep_analysis_report.html
"""

import json
import os
from datetime import datetime

import plotly.graph_objects as go
import plotly.io as pio

INPUT_PATH = "data/output/deep_analysis.json"
OUTPUT_PATH = "data/output/deep_analysis_report.html"

ACCENT = "#6c63ff"
ACCENT_LIGHT = "#8b85ff"
PALETTE = [
    "#6c63ff", "#ff6584", "#43e8d8", "#ffc857", "#a29bfe",
    "#fd79a8", "#00cec9", "#fdcb6e", "#e17055",
]

DARK_LAYOUT = dict(
    paper_bgcolor="#0a0a0f",
    plot_bgcolor="#0a0a0f",
    font=dict(color="#e0e0e0", family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size=13),
    xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    margin=dict(l=60, r=30, t=50, b=60),
    hoverlabel=dict(bgcolor="#1e1e2e", font_color="#e0e0e0"),
)

SLICER_COLORS = {
    "PrusaSlicer": "#ff6584",
    "OrcaSlicer": "#43e8d8",
    "Cura": "#ffc857",
    "SuperSlicer": "#a29bfe",
}


def pretty_category(name):
    return name.replace("_", " ").title()


def chart_to_div(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# --- Charts ---

def chart_category_bars(data):
    cats = data["categories"]
    names = [pretty_category(c) for c in cats]
    scores = [cats[c]["signal_score"] for c in cats]

    fig = go.Figure(go.Bar(
        y=names[::-1], x=scores[::-1],
        orientation="h", marker_color=ACCENT,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title="Pain Points by Category (Signal Score)",
        **DARK_LAYOUT,
        height=400,
    )
    return chart_to_div(fig)


def chart_severity_stacked(data):
    cats = data["categories"]
    names = list(cats.keys())[::-1]
    pretty = [pretty_category(c) for c in names]

    high = [cats[c]["severity"]["high"] for c in names]
    medium = [cats[c]["severity"]["medium"] for c in names]
    low = [cats[c]["severity"]["low"] for c in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=pretty, x=high, name="High", orientation="h", marker_color="#ff6584"))
    fig.add_trace(go.Bar(y=pretty, x=medium, name="Medium", orientation="h", marker_color="#ffc857"))
    fig.add_trace(go.Bar(y=pretty, x=low, name="Low", orientation="h", marker_color="#555"))
    fig.update_layout(
        title="Severity Distribution by Category",
        barmode="stack",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        **DARK_LAYOUT,
        height=400,
    )
    return chart_to_div(fig)


def chart_slicer_breakdown(data):
    cats = data["categories"]
    cat_names = list(cats.keys())[:5]  # top 5 only
    pretty = [pretty_category(c) for c in cat_names]
    slicers = ["PrusaSlicer", "OrcaSlicer", "Cura", "SuperSlicer"]

    fig = go.Figure()
    for slicer in slicers:
        values = [cats[c]["by_slicer"].get(slicer, 0) for c in cat_names]
        fig.add_trace(go.Bar(
            x=pretty, y=values, name=slicer,
            marker_color=SLICER_COLORS[slicer],
        ))
    fig.update_layout(
        title="Issues by Slicer (Top 5 Categories)",
        barmode="group",
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        **DARK_LAYOUT,
        height=420,
    )
    return chart_to_div(fig)


def chart_timeline(data):
    timeline = data["timeline"]
    quarters = sorted(timeline.keys())
    # Filter to 2020+ to avoid sparse early data
    quarters = [q for q in quarters if q >= "2020"]

    # Top 5 categories by total count
    top_cats = list(data["categories"].keys())[:5]

    fig = go.Figure()
    for i, cat in enumerate(top_cats):
        values = [timeline.get(q, {}).get(cat, 0) for q in quarters]
        fig.add_trace(go.Scatter(
            x=quarters, y=values,
            mode="lines+markers", name=pretty_category(cat),
            line=dict(color=PALETTE[i], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title="Issues Over Time (by Quarter, Top 5 Categories)",
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        **DARK_LAYOUT,
        height=400,
    )
    return chart_to_div(fig)


def chart_feature_vs_bug(data):
    fvb = data["feature_vs_bug"]
    labels = ["Feature Request", "Bug Report", "Other"]
    values = [fvb["feature_request"], fvb["bug_report"], fvb["other"]]
    colors = [ACCENT, "#ff6584", "#555"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo="label+percent",
        textfont=dict(color="#e0e0e0"),
    ))
    fig.update_layout(
        title="Feature Requests vs Bug Reports",
        **DARK_LAYOUT,
        height=380,
        showlegend=False,
    )
    return chart_to_div(fig)


def chart_engagement(data):
    cats = data["categories"]
    names = [pretty_category(c) for c in cats]
    engagement = [cats[c]["engagement_total"] for c in cats]

    fig = go.Figure(go.Bar(
        y=names[::-1], x=engagement[::-1],
        orientation="h", marker_color="#43e8d8",
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title="Total Engagement by Category (Reactions + Comments)",
        **DARK_LAYOUT,
        height=400,
    )
    return chart_to_div(fig)


def chart_youtube_views(data):
    yt = data.get("youtube_view_weighted", {})
    if not yt:
        return "<p style='color:#888;'>No YouTube view data available.</p>"

    names = [pretty_category(c) for c in yt]
    views = list(yt.values())

    fig = go.Figure(go.Bar(
        y=names[::-1], x=views[::-1],
        orientation="h", marker_color="#ffc857",
        hovertemplate="%{y}: %{x:,.0f} views<extra></extra>",
    ))
    fig.update_layout(
        title="YouTube View-Weighted Categories (Market Reach)",
        **DARK_LAYOUT,
        height=380,
    )
    return chart_to_div(fig)


# --- Tables ---

def table_top_engagement(data):
    issues = data.get("top_engagement_issues", [])
    if not issues:
        return "<p style='color:#888;'>No engagement data available.</p>"

    rows = ""
    for i, iss in enumerate(issues, 1):
        title = iss["title"][:70]
        url = iss["url"]
        rows += f"""<tr>
            <td>{i}</td>
            <td><a href="{url}" target="_blank">{title}</a></td>
            <td>{iss['repo']}</td>
            <td>{iss['reactions']}</td>
            <td>{iss['comments']}</td>
            <td>{iss['engagement']}</td>
            <td>{pretty_category(iss['category'])}</td>
        </tr>"""

    return f"""
    <h2>Top 10 Highest-Engagement Issues</h2>
    <table>
        <tr><th>#</th><th>Title</th><th>Slicer</th><th>Reactions</th><th>Comments</th><th>Score</th><th>Category</th></tr>
        {rows}
    </table>"""


def table_wtp(data):
    signals = data.get("wtp_signals", [])
    if not signals:
        return """<h2>Willingness-to-Pay Signals</h2>
        <p style='color:#888;'>No explicit willingness-to-pay quotes found in the data.</p>"""

    rows = ""
    for s in signals[:20]:
        quote = s["quote"][:150]
        url = s["source_url"]
        title = s["source_title"][:50]
        rows += f"""<tr>
            <td>"{quote}"</td>
            <td><a href="{url}" target="_blank">{title}</a></td>
        </tr>"""

    return f"""
    <h2>Willingness-to-Pay Signals ({len(signals)} found)</h2>
    <table>
        <tr><th>Quote</th><th>Source</th></tr>
        {rows}
    </table>"""


def table_workarounds(data):
    workarounds = data.get("workaround_patterns", [])
    if not workarounds:
        return """<h2>Current Workarounds</h2>
        <p style='color:#888;'>No workaround patterns found.</p>"""

    rows = ""
    for w in workarounds[:15]:
        quote = w["quote"][:150]
        url = w["source_url"]
        title = w["source_title"][:50]
        rows += f"""<tr>
            <td>"{quote}"</td>
            <td>{w['count']}</td>
            <td><a href="{url}" target="_blank">{title}</a></td>
        </tr>"""

    return f"""
    <h2>Current Workarounds ({len(workarounds)} patterns)</h2>
    <table>
        <tr><th>Workaround</th><th>Freq</th><th>Example Source</th></tr>
        {rows}
    </table>"""


# --- Assembly ---

def stat_card(number, label):
    return f"""<div class="stat-card">
        <div class="stat-number">{number}</div>
        <div class="stat-label">{label}</div>
    </div>"""


def generate_html(data):
    meta = data["metadata"]
    cats = data["categories"]
    top_cat = next(iter(cats))
    top_issue = data["top_engagement_issues"][0] if data["top_engagement_issues"] else None

    total_relevant = meta["github_relevant"] + meta["youtube_relevant"]

    summary_cards = "".join([
        stat_card(f"{meta['github_issues_total'] + meta['youtube_videos_total']:,}", "Sources Analyzed"),
        stat_card(f"{total_relevant:,}", "Relevant Pain Points"),
        stat_card(pretty_category(top_cat), "Top Category"),
        stat_card(f"{cats[top_cat]['signal_score']:,}", "Top Signal Score"),
        stat_card(f"{len(data['wtp_signals'])}", "WTP Signals"),
        stat_card(f"{len(data['timeline'])}", "Quarters of Data"),
    ])

    charts = [
        chart_category_bars(data),
        chart_severity_stacked(data),
        chart_slicer_breakdown(data),
        chart_timeline(data),
    ]

    charts_row2 = [
        chart_feature_vs_bug(data),
        chart_engagement(data),
    ]

    charts_row3 = [
        chart_youtube_views(data),
    ]

    tables = [
        table_top_engagement(data),
        table_wtp(data),
        table_workarounds(data),
    ]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OrientIQ — Deep Market Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        background: #0a0a0f;
        color: #e0e0e0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        line-height: 1.6;
        padding: 40px;
        max-width: 1200px;
        margin: 0 auto;
    }}
    h1 {{
        color: #fff;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }}
    h1 span {{ color: {ACCENT}; }}
    .subtitle {{
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }}
    h2 {{
        color: {ACCENT};
        font-size: 1.3rem;
        font-weight: 700;
        margin: 3rem 0 1rem;
        padding-top: 1rem;
        border-top: 1px solid #1e1e2e;
    }}
    .summary {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 2rem 0 3rem;
    }}
    .stat-card {{
        background: #13131a;
        border: 1px solid #1e1e2e;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        min-width: 160px;
        text-align: center;
        flex: 1;
    }}
    .stat-number {{
        font-size: 1.6rem;
        color: {ACCENT};
        font-weight: 800;
    }}
    .stat-label {{
        color: #888;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }}
    .chart-container {{
        margin: 2rem 0;
    }}
    .chart-row {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0 2rem;
        font-size: 0.9rem;
    }}
    th {{
        background: #13131a;
        color: {ACCENT};
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    td {{
        padding: 0.6rem 1rem;
        border-bottom: 1px solid #1e1e2e;
        color: #ccc;
    }}
    tr:hover td {{ background: rgba(108, 99, 255, 0.05); }}
    a {{ color: {ACCENT}; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    footer {{
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1e1e2e;
        color: #555;
        font-size: 0.8rem;
        text-align: center;
    }}
    @media (max-width: 768px) {{
        body {{ padding: 20px; }}
        .chart-row {{ grid-template-columns: 1fr; }}
        .stat-card {{ min-width: 120px; }}
    }}
</style>
</head>
<body>

<h1>Orient<span>IQ</span> — Deep Market Analysis</h1>
<p class="subtitle">Enhanced analysis of {meta['github_issues_total']:,} GitHub issues + {meta['youtube_videos_total']} YouTube videos from major 3D printing slicer communities.</p>

<div class="summary">
    {summary_cards}
</div>

<div class="chart-container">{charts[0]}</div>
<div class="chart-container">{charts[1]}</div>
<div class="chart-container">{charts[2]}</div>
<div class="chart-container">{charts[3]}</div>

<div class="chart-row">
    <div class="chart-container">{charts_row2[0]}</div>
    <div class="chart-container">{charts_row2[1]}</div>
</div>

<div class="chart-container">{charts_row3[0]}</div>

{tables[0]}
{tables[1]}
{tables[2]}

<footer>
    Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | OrientIQ Market Research
</footer>

</body>
</html>"""

    return html


def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    html = generate_html(data)

    with open(OUTPUT_PATH, "w") as f:
        f.write(html)

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"Report written to {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
