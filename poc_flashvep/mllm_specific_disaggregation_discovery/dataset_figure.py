"""Make Figure 1 from the grouped freeze and measured Native200 results."""

import argparse
import csv
from pathlib import Path

from .dataset import file_hash, read, require


BENCHMARKS = ("Main100", "MMMU", "MathVista", "MMReason")
COLORS = ("#2d6080", "#b04e42", "#5f774d", "#775493")
TOKENS = ("visual_tokens", "text_tokens", "special_tokens")


def figure_tables(manifest_path, native_dir):
    manifest = read(manifest_path)
    summary = read(native_dir / "native_summary.json")
    require(summary["confirmation_manifest_SHA256"] == file_hash(manifest_path),
            "Native summary was prepared against another confirmation manifest")
    observed = read(native_dir / "native_requests.json")
    by_id = {row["request_id"]: row for row in observed}
    frozen = manifest["requests"]
    require(len(observed) == len(by_id) == len(frozen) == 200
            and by_id.keys() == {row["request_id"] for row in frozen}, "Native roster disagrees with the freeze")
    rows = []
    for row in frozen:
        native = by_id[row["request_id"]]
        require(row["source"] == native["source"] and row["total_input_tokens"] == native["input_tokens"],
                f"Native geometry differs for {row['request_id']}")
        rows.append({
            "request_id": row["request_id"], "source": row["source"], "benchmark": row["study_source"],
            "split": row["study_split"], "tier1": row["tier1"], "image_count": row["image_count"],
            "input_tokens": row["total_input_tokens"], **{key: row[key] for key in TOKENS},
            "native_status": native["status"], "native_output_tokens": native["output_tokens"],
        })
    scores = [{
        "source": source, "task": source.rsplit("/", 1)[-1], "n_scored": row["n_scored"],
        "requests": row["requests"], "score": row["score"], "metric": row["metric"],
        "complete_task_scorer": row["complete_task_scorer"],
        "evidence": "LOCAL_SCORE" if row["complete_task_scorer"] else "DIAGNOSTIC",
    } for source, row in summary["by_source"].items()]
    return rows, scores


def save_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw(out, rows, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 3, figsize=(13.3, 7.7), layout="constrained")
    x = range(1, 5)
    groups = [[row for row in rows if row["benchmark"] == key] for key in BENCHMARKS]
    count_ax, token_ax, input_ax, output_ax, image_ax, quality_ax = axes.flat

    for ax in (count_ax, token_ax, input_ax, output_ax, image_ax):
        ax.set_xticks(list(x), BENCHMARKS)
        ax.grid(axis="y", alpha=.20)
        ax.set_axisbelow(True)
    count_ax.set_title("A  Frozen cohort counts", loc="left")
    counts = {split: [sum(row["split"] == split for row in group) for group in groups]
              for split in ("discovery", "validation")}
    count_ax.bar(x, counts["discovery"], color="#2d6080", label="Discovery")
    count_ax.bar(x, counts["validation"], bottom=counts["discovery"], color="#a5c8d5", label="Validation")
    for point, group in zip(x, groups, strict=True):
        tier1 = sum(row["tier1"] for row in group)
        count_ax.text(point, len(group) + 2, f"Tier1 {tier1}", ha="center", fontsize=8)
    count_ax.set_ylim(0, 120)
    count_ax.set_ylabel("Requests")
    count_ax.legend(loc="upper right", frameon=False, fontsize=8)

    token_ax.set_title("B  Input token composition", loc="left")
    bottoms = [0.] * len(groups)
    for key, color in zip(TOKENS, ("#2d6080", "#dfa548", "#bbc2c5"), strict=True):
        values = [sum(row[key] for row in group) / sum(row["input_tokens"] for row in group) for group in groups]
        token_ax.bar(x, values, bottom=bottoms, color=color, label=key.removesuffix("_tokens").title())
        bottoms = [a + b for a, b in zip(bottoms, values, strict=True)]
    token_ax.yaxis.set_major_formatter(PercentFormatter(1))
    token_ax.set_ylim(0, 1)
    token_ax.set_ylabel("Share of summed input")
    token_ax.legend(loc="lower left", facecolor="white", framealpha=.96, fontsize=8)

    for ax, key, title, ylabel, use_log in (
        (input_ax, "input_tokens", "C  Model input lengths", "Input tokens", True),
        (output_ax, "native_output_tokens", "D  Native Natural EOS output lengths", "Output tokens", True),
        (image_ax, "image_count", "E  Image count per request", "Images", True),
    ):
        values = [[row[key] for row in group if row[key] is not None and row[key] > 0] for group in groups]
        box = ax.boxplot(values, positions=list(x), patch_artist=True, widths=.62,
                         medianprops={"color": "#202324", "linewidth": 1.1},
                         flierprops={"marker": ".", "markersize": 2, "alpha": .7})
        for patch, color in zip(box["boxes"], COLORS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(.70)
        if use_log:
            ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(x), BENCHMARKS)
    completed = sum(row["native_status"] == "MEASURED_ONLINE" for row in rows)
    output_ax.text(.98, .98, f"{completed}/200 completed", transform=output_ax.transAxes,
                   ha="right", va="top", fontsize=8)

    quality_ax.set_title("F  Native task point scores", loc="left")
    for index, row in enumerate(scores):
        if row["score"] is None:
            quality_ax.text(.01, index, "unscored", va="center", fontsize=8)
            continue
        diagnostic = row["evidence"] == "DIAGNOSTIC"
        quality_ax.barh(index, row["score"], color="#e4b080" if diagnostic else "#507995",
                        hatch="//" if diagnostic else "", edgecolor="#545454", linewidth=.5)
        quality_ax.text(row["score"] + .015, index, f"{100 * row['score']:.1f}% · {row['n_scored']}/{row['requests']}",
                        va="center", fontsize=8)
    quality_ax.set_yticks(range(len(scores)), [
        row["task"] + (" †" if row["evidence"] == "DIAGNOSTIC" else "") for row in scores
    ])
    quality_ax.invert_yaxis()
    quality_ax.set_xlabel("Accuracy, ANLS or task-specific diagnostic")
    quality_ax.set_xlim(0, 1.06)
    quality_ax.xaxis.set_major_formatter(PercentFormatter(1))
    quality_ax.grid(axis="x", alpha=.20)
    quality_ax.set_axisbelow(True)
    quality_ax.legend(handles=[
        Patch(facecolor="#507995", edgecolor="#545454", label="Local task scorer"),
        Patch(facecolor="#e4b080", edgecolor="#545454", hatch="//", label="† Local diagnostic"),
    ], loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Discovery200 · Frozen dataset and measured Native outputs", fontsize=15, fontweight="bold")
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"figure1_discovery200_dataset_profile.{suffix}", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, scores = figure_tables(args.manifest, args.native)
    args.output.mkdir(parents=True, exist_ok=True)
    save_table(args.output / "figure1_dataset_profile.csv", rows)
    save_table(args.output / "figure1_native_quality.csv", scores)
    draw(args.output, rows, scores)


if __name__ == "__main__":
    main()
