"""Render the assessed direction/evidence matrix from the campaign decisions."""

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.decisions.read_text())
    fields = ('direction', 'online_evidence', 'verdict')
    rows = [[fill(row[key].replace('_', ' ') if key == 'verdict' else row[key], width)
             for key, width in zip(fields, (24, 52, 23), strict=True)] for row in data['directions']]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=('Direction', 'Serving evidence', 'Decision'),
                     colWidths=(.24, .51, .25), cellLoc='left', colLoc='left', bbox=[0, .02, 1, .86])
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#fff')
        if row == 0:
            cell.set_facecolor('#245a73')
            cell.set_text_props(color='white', weight='bold')
        elif column == 2:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2e6dd' if 'NO_GO' in data['directions'][row - 1]['verdict'] else '#e1e8ec')
        else:
            cell.set_facecolor('#f1f4f5')
    ax.set_title('Dense MLLM EPD memory-use decisions', loc='left', fontsize=15, pad=3)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
