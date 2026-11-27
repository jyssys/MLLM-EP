"""Render the H200 screening matrix from the recorded candidate decisions."""

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).parent
COLUMNS = ("Direction", "Physical evidence", "Screening decision", "Candidate online")
DECISION_COLORS = {
    "pending": "#e7eef7",
    "no_go": "#f6eade",
    "go": "#e3efeb",
}


def render(data, output):
    rows = data["directions"]
    cells = [
        [row["direction"], f"{row['evidence']}\n{fill(row['finding'], 63)}",
         f"{row['status']}\n{fill(row['reason'], 30)}", row["candidate_online"]]
        for row in rows
    ]
    backgrounds = [
        ["#eff4f8", "#f8fafc", DECISION_COLORS.get(row["decision_color"], "#f8fafc"), "#eff4f8"]
        for row in rows
    ]
    fig, ax = plt.subplots(figsize=(14.4, 5.3))
    ax.set_axis_off()
    fig.text(.03, .925, data["title"], fontweight="bold", fontsize=20, color="#182f43")
    fig.text(.03, .865, data["subtitle"], fontsize=11, color="#445a6f")
    table = ax.table(cellText=cells, cellColours=backgrounds, colLabels=COLUMNS,
                     colColours=["#253f56"] * len(COLUMNS), colWidths=(.205, .42, .245, .13),
                     cellLoc="left", colLoc="left", bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.5)
        cell.PAD = .045
        cell.set_height(.65 if row == 0 else 1.0)
        cell.get_text().set_color("white" if row == 0 else "#182f43")
        if row == 0 or col == 0:
            cell.get_text().set_fontweight("bold")
    ax.set_position((.03, .155, .94, .657))
    fig.text(.03, .075, data["note"], fontsize=10, color="#445a6f")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "results" / "candidate_verdict_matrix.json")
    parser.add_argument("--out", type=Path, default=HERE / "figures" / "candidate_verdict_matrix.png")
    args = parser.parse_args()
    render(json.loads(args.input.read_text()), args.out)


if __name__ == "__main__":
    main()
