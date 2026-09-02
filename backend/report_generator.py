"""
Week 8 -- Automated PDF report generation
----------------------------------------------
Takes the same combined analysis dict analyze_screenplay() produces and
turns it into a polished PDF, styled like a real professional script
coverage report (the industry-standard format a studio reader produces):
a header summary, a recommendation verdict, character breakdown,
sentiment/structure summary, and a genre + viability assessment.

WHY THIS SHAPE, NOT JUST A DUMP OF THE JSON: a real coverage report leads
with a verdict a producer can act on in 10 seconds (Pass / Consider /
Recommend), then supports it with detail. Dumping raw numbers first
would bury the one thing a reader actually wants first.

Run standalone (for testing without the API):
    python backend/report_generator.py <analysis.json> <output.pdf>
"""

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- required for server-side PDF generation
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
)

NAVY = colors.HexColor("#1F3864")
LIGHT_BLUE = colors.HexColor("#EAF1FB")
GREEN = colors.HexColor("#D9EAD3")
AMBER = colors.HexColor("#FCE8B2")
RED = colors.HexColor("#F4CCCC")


def _build_sentiment_chart(analysis: dict):
    """Draws the sentiment arc (raw + smoothed) with every predicted story
    beat marked at its actual scene position. Labels for beats that land
    close together (e.g. All Is Lost and Dark Night of the Soul sometimes
    land on the exact same scene -- a known limitation, not hidden here)
    are staggered onto alternating height tiers so they don't overlap,
    while the vertical line always marks the TRUE scene position
    regardless of which tier the label text sits at."""
    sent = analysis.get("sentiment_arc", {})
    scores = sent.get("scene_scores")
    smoothed = sent.get("smoothed_scores")
    if not scores:
        return None

    n = len(scores)
    x = list(range(n))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, scores, color="#c9c9c9", linewidth=0.7, label="Raw per-scene score")
    if smoothed:
        ax.plot(x, smoothed, color="#1F3864", linewidth=2, label="Smoothed arc")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)

    beats = analysis.get("story_structure", {}).get("predicted_beats", [])
    beats_sorted = sorted(beats, key=lambda b: b["scene_index"])
    min_gap = max(2, n * 0.03)
    nudge = n * 0.022
    last_x = -1e9
    cluster_step = 0
    for b in beats_sorted:
        idx = b["scene_index"]
        if idx >= n:
            continue
        cluster_step = cluster_step + 1 if (idx - last_x) < min_gap else 0
        last_x = idx

        ax.axvline(idx, color="#a33d2e", alpha=0.3, linewidth=1)
        label_x = idx + cluster_step * nudge
        if cluster_step > 0:
            # Thin connector so it's clear this label's TRUE position is
            # the vertical line, not wherever the text ended up nudged to
            ax.plot([idx, label_x], [1.0, 1.03], transform=ax.get_xaxis_transform(),
                     color="#a33d2e", alpha=0.4, linewidth=0.6, clip_on=False)
        ax.annotate(
            b["beat"], xy=(label_x, 1.05), xycoords=("data", "axes fraction"),
            rotation=90, fontsize=6.5, ha="center", va="bottom", color="#4a4a4a",
        )

    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(-1, n)
    ax.set_xlabel("Scene index", fontsize=9)
    ax.set_ylabel("Sentiment (-1 negative -> +1 positive)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    fig.suptitle("Sentiment Arc with Predicted Story Beats", fontsize=11, fontweight="bold", y=1.15)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _recommendation(analysis: dict) -> tuple[str, colors.Color, str]:
    """Derives a simple Pass/Consider/Recommend verdict from the
    viability prediction -- the single most useful thing a real coverage
    report leads with. Thresholds are a reasonable, disclosed starting
    point (rating out of 10), not a validated industry standard."""
    rating = analysis.get("viability", {}).get("predicted_imdb_rating")
    if rating is None:
        return "UNRATED", colors.grey, "No viability estimate was available for this script."
    if rating >= 7.0:
        return "RECOMMEND", GREEN, (
            f"The predicted audience reception ({rating}/10) is strong. "
            f"Worth prioritising for further review."
        )
    if rating >= 5.5:
        return "CONSIDER", AMBER, (
            f"The predicted audience reception ({rating}/10) is middling. "
            f"May be worth a second read depending on genre fit and budget."
        )
    return "PASS", RED, (
        f"The predicted audience reception ({rating}/10) is weak. "
        f"Note: this estimate is text-only and has a documented low ceiling -- "
        f"treat as one input among many, not a final verdict."
    )


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontSize=22, leading=26, textColor=NAVY,
                               spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("SectionHeading", fontSize=14, leading=18, textColor=NAVY,
                               spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle("Body", fontSize=10, leading=14))
    styles.add(ParagraphStyle("Caveat", fontSize=8.5, leading=12, textColor=colors.HexColor("#666666"),
                               fontName="Helvetica-Oblique"))
    return styles


def generate_report(analysis: dict, output_path: str):
    if not analysis.get("success"):
        raise ValueError("Cannot generate a report from a failed analysis.")

    styles = build_styles()
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                             leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = []

    # ---- Header ----
    story.append(Paragraph(analysis["title"], styles["ReportTitle"]))
    story.append(Paragraph("Automated Script Coverage Report", styles["Body"]))
    story.append(Spacer(1, 12))

    # ---- Recommendation verdict (leads the report, as a real one would) ----
    verdict, verdict_color, verdict_text = _recommendation(analysis)
    verdict_table = Table(
        [[Paragraph(f"<b>{verdict}</b>", ParagraphStyle("V", fontSize=15, textColor=NAVY, leading=17)),
          Paragraph(verdict_text, styles["Body"])]],
        colWidths=[1.7 * inch, 4.8 * inch],
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Automated estimate only -- script text alone has a documented, limited ability to "
        "predict audience reception (see Viability Assessment below). Treat as one input, not a final call.",
        styles["Caveat"],
    ))

    # ---- Overview ----
    story.append(Paragraph("Overview", styles["SectionHeading"]))
    ov = analysis["overview"]
    overview_table = Table([
        ["Scenes", "Characters", "Dialogue Lines", "Predicted Genre(s)"],
        [str(ov["scene_count"]), str(ov["character_count"]), str(ov["dialogue_count"]),
         ", ".join(analysis["predicted_genres"]) or "None predicted"],
    ], colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 2 * inch])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(overview_table)

    if analysis.get("parser_notes"):
        story.append(Spacer(1, 6))
        for note in analysis["parser_notes"]:
            story.append(Paragraph(f"Parser note: {note}", styles["Caveat"]))

    # ---- Characters ----
    story.append(Paragraph("Character Breakdown", styles["SectionHeading"]))
    chars = analysis["characters"]
    story.append(Paragraph(
        f"<b>{len(chars['likely_real_names'])}</b> identified characters, "
        f"<b>{len(chars['likely_role_labels'])}</b> generic role labels filtered out "
        f"(e.g. background/functional characters), "
        f"<b>{len(chars['uncertain'])}</b> uncertain cases flagged for manual review.",
        styles["Body"],
    ))
    if chars["likely_real_names"]:
        story.append(Paragraph(", ".join(chars["likely_real_names"][:15]), styles["Body"]))

    rel = analysis["character_relationships"]
    story.append(Spacer(1, 8))
    protagonist_line = ""
    if rel.get("likely_protagonist"):
        protagonist_line = (
            f" Likely protagonist: <b>{rel['likely_protagonist']}</b> "
            f"(most scenes shared with other characters -- a reasonable proxy for "
            f"narrative centrality, not a confirmed role)."
        )
    story.append(Paragraph(
        f"<b>Character network:</b> {rel['character_count_in_network']} characters, "
        f"{rel['relationship_count']} relationships.{protagonist_line}",
        styles["Body"],
    ))
    if rel["most_central_characters"]:
        central_rows = [["Character", "Scenes Shared (weighted)", "Bridge Score (betweenness)"]]
        for c in rel["most_central_characters"][:5]:
            name = c["name"]
            if name == rel.get("likely_protagonist"):
                name = f"{name}  \u2605"  # star marker, explained in the line above
            central_rows.append([name, str(c["weighted_degree"]), f"{c['betweenness_centrality']:.3f}"])
        central_table = Table(central_rows, colWidths=[2.2 * inch, 2.4 * inch, 2.4 * inch])
        central_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(Spacer(1, 6))
        story.append(central_table)

    # ---- Sentiment & Structure ----
    story.append(Paragraph("Emotional Arc & Story Structure", styles["SectionHeading"]))
    sent = analysis["sentiment_arc"]
    story.append(Paragraph(
        f"Overall tone: <b>{sent.get('sentiment_label', 'N/A')}</b> "
        f"(raw score: {sent['average_sentiment']:+.3f}, model: {sent['model_source']}). "
        f"Most positive scene: <i>{sent['most_positive_scene']}</i>. "
        f"Most negative scene: <i>{sent['most_negative_scene']}</i>. "
        f"{sent['turning_point_count']} emotional turning points detected.",
        styles["Body"],
    ))
    if sent.get("sentiment_label_caveat"):
        story.append(Spacer(1, 3))
        story.append(Paragraph(sent["sentiment_label_caveat"], styles["Caveat"]))

    chart_buf = _build_sentiment_chart(analysis)
    if chart_buf:
        story.append(Spacer(1, 10))
        chart_width = 6.4 * inch
        chart_height = chart_width * (4.2 / 9)  # matches the figsize aspect ratio
        story.append(Image(chart_buf, width=chart_width, height=chart_height))
        story.append(Paragraph(
            "Red vertical lines mark each predicted story beat at its scene position -- "
            "see the table below for exact scene numbers and prediction method.",
            styles["Caveat"],
        ))

    story.append(Spacer(1, 8))
    beat_rows = [["Story Beat", "Scene #", "Method"]]
    for b in analysis["story_structure"]["predicted_beats"]:
        beat_rows.append([b["beat"], str(b["scene_index"]), b["method"]])
    beat_table = Table(beat_rows, colWidths=[2.2 * inch, 0.8 * inch, 4 * inch])
    beat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ]))
    story.append(beat_table)

    # ---- Viability detail ----
    story.append(Paragraph("Viability Assessment", styles["SectionHeading"]))
    via = analysis["viability"]
    story.append(Paragraph(
        f"Predicted IMDb rating: <b>{via['predicted_imdb_rating']}/10</b> "
        f"(confidence: {via.get('confidence', 'unknown')})",
        styles["Body"],
    ))
    if via.get("caveat"):
        story.append(Spacer(1, 4))
        story.append(Paragraph(via["caveat"], styles["Caveat"]))

    doc.build(story)


def main():
    if len(sys.argv) < 3:
        print("Usage: python backend/report_generator.py <analysis.json> <output.pdf>")
        sys.exit(1)
    analysis = json.loads(Path(sys.argv[1]).read_text())
    generate_report(analysis, sys.argv[2])
    print(f"Report saved to {sys.argv[2]}")


if __name__ == "__main__":
    main()