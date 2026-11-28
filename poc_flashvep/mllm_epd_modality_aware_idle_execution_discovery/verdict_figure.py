"""Render reviewed candidate verdicts into a standalone SVG matrix."""

import argparse
from html import escape
import json
from pathlib import Path
import textwrap


COLUMNS = (
    ("Candidate", "candidate", 80, 10),
    ("Execution boundary", "mechanism", 286, 33),
    ("Verdict", "verdict", 288, 32),
    ("Evidence", "evidence", 248, 27),
    ("Observed constraint or remaining gate", "basis", 470, 58),
)


def verdict_color(value):
    if value.startswith(("NO_GO", "FAIL")):
        return "#983f22"
    if value.startswith(("GO_", "PASS")):
        return "#17624c"
    if value.startswith(("CONDITIONAL", "INCONCLUSIVE")):
        return "#6a4b91"
    return "#485e77"


def wrap(value, width):
    return textwrap.wrap(value, width, break_long_words=True, break_on_hyphens=False)


def render(receipt):
    rows = receipt["candidates"]
    expected = "ABCDEFG"
    if len(rows) != len(expected) or {row.get("candidate") for row in rows} != set(expected):
        raise ValueError("Supply one reviewed disposition for each candidate A–G")
    prepared = []
    for row in sorted(rows, key=lambda entry: entry["candidate"]):
        lines = []
        for _, key, _, width in COLUMNS:
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Candidate {row['candidate']} requires a nonempty {key} string")
            lines.append(wrap(value, width))
        prepared.append((row, lines, 19 * max(map(len, lines)) + 21))

    width = sum(column[2] for column in COLUMNS) + 48
    height = 131 + sum(row[2] for row in prepared)
    title = "Candidate dispositions for fixed-role EPD"
    summary = receipt["scope"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 160:
        raise ValueError("Expected a nonempty figure scope of at most 160 characters")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"',
        f' viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(summary)}</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<g font-family="DejaVu Sans, Arial, sans-serif" fill="#13243a">',
        f'<text x="24" y="42" font-size="28" font-weight="700">{escape(title)}</text>',
        f'<text x="24" y="67" font-size="13" fill="#496078">{escape(summary)}</text>',
        f'<rect x="24" y="85" width="{width - 48}" height="36" rx="6" fill="#1a3957"/>',
    ]
    x = 24
    for label, _, column_width, _ in COLUMNS:
        svg.append(f'<text x="{x + 12}" y="108" fill="#fff" font-size="13" font-weight="700">'
                   f'{escape(label)}</text>')
        x += column_width

    y = 121
    for index, (row, lines, row_height) in enumerate(prepared):
        background = "#f0f4f8" if index % 2 == 0 else "#ffffff"
        svg.append(f'<rect x="24" y="{y}" width="{width - 48}" height="{row_height}" fill="{background}"/>')
        x = 24
        for (_, key, column_width, _), cell in zip(COLUMNS, lines, strict=True):
            fill = verdict_color(row[key]) if key == "verdict" else "#13243a"
            weight = 700 if key in ("candidate", "verdict") else 400
            for offset, line in enumerate(cell):
                svg.append(f'<text x="{x + 12}" y="{y + 23 + 19 * offset}" font-size="14" '
                           f'font-weight="{weight}" fill="{fill}">{escape(line)}</text>')
            x += column_width
        y += row_height
    svg.extend(["</g>", "</svg>", ""])
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Reviewed JSON with scope and A–G candidate objects")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    svg = render(json.loads(args.input.read_bytes()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg)


if __name__ == "__main__":
    main()
