#!/usr/bin/env python3
"""Generate required structural/timing plots from a virtual-EP projection."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from virtual_ep.schema import TraceBundle


def _mean_by(rows: list[dict], key: str, value: str):
    groups = defaultdict(list)
    for row in rows:
        if row[value] not in ("", None, "None"):
            groups[int(float(row[key]))].append(float(row[value]))
    x = sorted(groups)
    return x, [float(np.mean(groups[index])) for index in x]


def _line(path: Path, x, series, xlabel: str, ylabel: str):
    plt.figure(figsize=(7.2, 4.2))
    for label, values in series.items():
        plt.plot(x, values, marker=".", linewidth=1.2, label=label)
    if len(series) > 1:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trace = TraceBundle.load(args.trace)
    with args.invocations.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    iterations = [
        {name: trace.iterations[name][index].item() for name in trace.iterations}
        for index in range(len(trace.iterations["nfe"]))
    ]
    nfe, masked = _mean_by(iterations, "nfe", "masked_after")
    _line(
        args.output / "remaining_mask_vs_iteration.png", nfe,
        {"remaining MASK": masked}, "NFE / refinement iteration", "mean rows",
    )
    nfe, accepted = _mean_by(iterations, "nfe", "accepted")
    _line(
        args.output / "accepted_tokens_vs_iteration.png", nfe,
        {"accepted": accepted}, "NFE / refinement iteration", "mean tokens",
    )
    plot_specs = (
        ("max_mean_load", "ep4_max_mean_rank_load_vs_iteration.png", "max/mean rank load"),
        ("remote_dispatch_bytes", "ep4_remote_bytes_vs_iteration.png", "logical bytes"),
        ("mean_token_fanout", "ep4_fanout_vs_iteration.png", "destination ranks/token"),
    )
    for field, filename, ylabel in plot_specs:
        x, values = _mean_by(rows, "nfe", field)
        _line(args.output / filename, x, {field: values}, "NFE / refinement iteration", ylabel)
    timing = {}
    timing_x = None
    for field, label in (
        ("dispatch_ms", "EP dispatch"),
        ("critical_expert_ms", "routed expert compute"),
        ("combine_ms", "EP combine"),
    ):
        x, values = _mean_by(rows, "nfe", field)
        timing_x = x if timing_x is None else timing_x
        if x != timing_x:
            raise RuntimeError("timing fields have inconsistent iteration axes")
        timing[label] = values
    if timing_x:
        _line(
            args.output / "ep4_dispatch_expert_combine_vs_iteration.png",
            timing_x, timing, "NFE / refinement iteration", "predicted ms",
        )

    layers = sorted({int(row["layer_id"]) for row in rows})
    nfes = sorted({int(row["nfe"]) for row in rows})
    layer_index = {value: index for index, value in enumerate(layers)}
    nfe_index = {value: index for index, value in enumerate(nfes)}
    values = [[[] for _ in nfes] for _ in layers]
    for row in rows:
        values[layer_index[int(row["layer_id"])]][nfe_index[int(row["nfe"])]].append(
            float(row["rank_load_cv"])
        )
    heatmap = np.full((len(layers), len(nfes)), np.nan)
    for layer in range(len(layers)):
        for nfe_value in range(len(nfes)):
            if values[layer][nfe_value]:
                heatmap[layer, nfe_value] = np.mean(values[layer][nfe_value])
    plt.figure(figsize=(11, 5))
    image = plt.imshow(heatmap, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(image, label="rank-load CV")
    plt.yticks(range(len(layers)), layers)
    if nfes:
        ticks = np.linspace(0, len(nfes) - 1, min(10, len(nfes))).astype(int)
        plt.xticks(ticks, [nfes[index] for index in ticks], rotation=45)
    plt.xlabel("NFE / refinement iteration")
    plt.ylabel("MoE layer")
    plt.tight_layout()
    plt.savefig(args.output / "ep4_layer_iteration_imbalance_heatmap.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
