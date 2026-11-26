"""Regenerate measured campaign figures from exported JSON and CSV."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#1e5d82", "#ce7839", "#438362", "#91516e")


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def grouped_bars(folder, name, labels, series, title, ylabel, *, note=None):
    """Plot one bar per available measurement. Missing arms have empty positions."""
    fig, ax = plt.subplots(figsize=(max(6.0, 1.2 * len(labels)), 3.9))
    x = np.arange(len(labels))
    width = .78 / len(series)
    for index, (series_label, data) in enumerate(series.items()):
        mask = np.array([value is not None for value in data])
        position = x + width * (index - (len(series) - 1) / 2)
        ax.bar(position[mask], [value for value in data if value is not None],
               width=width, color=COLORS[index % len(COLORS)], label=series_label)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=.20)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    if note:
        fig.text(.08, -.025, note, color="#424246", fontsize=8)
    save(fig, folder, name)


def plot_correctness(data, folder):
    grouped_bars(
        folder, "k1_correctness",
        [row["label"] for row in data],
        {
            "Byte-identical final KV layer segments": [row["final_kv_byte_equal_percent"] for row in data],
            "Exact Greedy": [row["greedy_exact_percent"] for row in data],
        },
        "Strict correctness by controlled execution", "Agreement (%)",
        note="KV uses layer segments. Greedy uses requests. Mixed fixtures require actual D suffix execution.",
    )


def plot_funnel(data, folder):
    names = list(data["stages"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    y = np.arange(len(names))
    series = data["counts"]
    width = .78 / len(series)
    for index, (label, values) in enumerate(series.items()):
        mask = np.array([value is not None for value in values])
        ax.barh((y + width * (index - (len(series) - 1) / 2))[mask],
                [value for value in values if value is not None], height=width,
                label=label, color=COLORS[index % len(COLORS)])
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Distinct requests")
    ax.set_title("Admission funnel at observed prefill boundaries", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=.20)
    if data.get("scope"):
        fig.text(.08, -.025, data["scope"], color="#424246", fontsize=8)
    save(fig, folder, "k2_cut_aligned_slack")


def plot_forecast(data, folder):
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    extent = 0
    for index, (label, samples) in enumerate(data.items()):
        if not samples:
            continue
        actual, predicted = zip(*samples, strict=True)
        extent = max(extent, max(actual), max(predicted))
        ax.scatter(actual, predicted, color=COLORS[index % len(COLORS)], s=15, alpha=.55, label=label)
    ax.plot((0, extent), (0, extent), "--", color="#5a5959", lw=.9)
    ax.set(xlabel="Observed time to next P-ready (ms)", ylabel="Online prediction (ms)")
    ax.set_title("Forecast versus post-run observation", loc="left")
    ax.legend(frameon=False)
    save(fig, folder, "k3_budget_prediction")


def plot_partition(data, folder):
    grouped_bars(
        folder, "k6_partition_transport", [row["label"] for row in data],
        {"TTFT p95": [row["ttft_p95_s"] for row in data]},
        "Transport and partition in active online C3", "TTFT p95 (s)",
        note="Only correctness-qualified, completed online arms are plotted.",
    )


def plot_cdfs(data, folder):
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.1))
    for index, (label, samples) in enumerate(data.items()):
        for ax, (key, unit) in zip(axes, (("ttft_ms", "TTFT"), ("itl_ms", "Token ITL")), strict=True):
            values = np.sort(samples[key])
            if len(values):
                ax.step(values, np.arange(1, len(values) + 1) / len(values),
                        where="post", color=COLORS[index % len(COLORS)], label=label)
            ax.set(xlabel=f"{unit} (ms)", ylabel="Empirical CDF")
            ax.grid(alpha=.20)
    axes[0].legend(frameon=False)
    save(fig, folder, "k7_online_serving")


def plot_timeline(data, folder, filename="k8_request_timeline"):
    fig, ax = plt.subplots(figsize=(8.7, 3.2))
    lanes = list(data["events"])
    for index, lane in enumerate(lanes):
        for begin, end in data["events"][lane]:
            if end == begin:
                ax.scatter([begin], [index], marker="|", s=110, color=COLORS[index % len(COLORS)])
            else:
                ax.broken_barh([(begin, end - begin)], (index - .31, .62),
                               facecolors=COLORS[index % len(COLORS)])
    ax.set_yticks(range(len(lanes)), lanes)
    ax.invert_yaxis()
    ax.set_xlabel("Time from the observed cut (ms)")
    ax.set_title(data["label"], loc="left")
    ax.grid(axis="x", alpha=.2)
    save(fig, folder, filename)


DRAW = {
    "correctness": plot_correctness,
    "funnel": plot_funnel,
    "predictor": plot_forecast,
    "partition_online": plot_partition,
    "online": plot_cdfs,
    "timeline": plot_timeline,
    "active_timeline": lambda data, folder: plot_timeline(data, folder, "k8_active_export_timeline"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    items = json.loads(options.inputs.read_text())
    options.output.mkdir(parents=True, exist_ok=True)
    availability = {}
    for key, plot in DRAW.items():
        item = items.get(key, {})
        availability[key] = item.get("evidence", "NOT_MEASURED")
        if item.get("values"):
            plot(item["values"], options.output)
    (options.output / "availability.json").write_text(json.dumps(availability, indent=2) + "\n")


if __name__ == "__main__":
    main()
