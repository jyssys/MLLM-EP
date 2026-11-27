"""Regenerate Native EPD motivation figures from compact measurement exports."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scan(scan, target):
    qps = [row["offered_qps"] for row in scan["runs"]]
    rows = [row["metrics"] for row in scan["runs"]]
    panels = (
        ("TTFT_p95_ms", "TTFT p95 (seconds)", .001, 5),
        ("TPOT_p95_ms", "TPOT p95 (ms/token)", 1, 33),
        ("ITL_p99_ms", "ITL p99 (ms)", 1, None),
        ("SLO_goodput", "Joint SLO goodput (tokens/s)", 1, None),
    )
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.0), layout="constrained")
    for axis, (name, label, scale, goal) in zip(axes, panels, strict=True):
        axis.plot(qps, [row[name] * scale for row in rows], "o-", color="#145374")
        if goal is not None:
            axis.axhline(goal, ls="--", color="#a84726", alpha=.7)
        axis.set(xlabel="Offered requests/s", title=label, xticks=qps)
        axis.grid(alpha=.2)
    fig.suptitle("NativeStrong EPD · 100 natural-output HTTP/SSE requests per point")
    write(fig, target / "native_qps_screen.png")


def plot_profile(profile, timeline, target, suffix=""):
    role_names = ("encoder", "prefill", "decode")
    labels = ("E all kernels", "P all kernels", "D all kernels",
              "D attention kernels", "E idle and no Encoder HTTP")
    keys = (*role_names, "D_attention", "E_no_kernel_or_HTTP")
    x = np.array([float(row["relative_monotonic_ms"]) / 1000 for row in timeline])
    values = np.array([[float(row[key]) for row in timeline] for key in keys])
    fig, axis = plt.subplots(figsize=(10, 3.5), layout="constrained")
    heat = axis.pcolormesh(np.r_[x, x[-1] + .1], np.arange(len(keys) + 1) - .5, values,
                           cmap="Blues", vmin=0, vmax=1)
    axis.set(xlabel="Seconds from the central arrival window", yticks=range(len(labels)), yticklabels=labels,
             title="Native E/P/D aligned timeline, 100 ms bins")
    axis.invert_yaxis()
    fig.colorbar(heat, ax=axis, label="Fraction of each bin")
    fig.suptitle("Clean/profile/clean evidence · " + profile["evidence"])
    write(fig, target / f"native_epd_timeline{suffix}.png")

    intervals = profile["E_and_P_opportunity"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 3.3), layout="constrained")
    hues = ("#19495f", "#277f69", "#ae5f47")
    friendly = ("E physical idle", "E idle + no HTTP", "P idle + no HTTP")
    for index, (label, row) in enumerate(intervals.items()):
        left.bar(index, 100 * row["overlap_fraction_of_D_attention"], color=hues[index])
        oracle = row["retrospective_attention_kernel_start_fit"]
        x_us = sorted(oracle, key=int)
        right.plot(list(map(int, x_us)), [100 * oracle[key]["eligible_fraction"] for key in x_us],
                   "o-", color=hues[index], label=friendly[index])
    left.set(title="Measured kernel-time intersection", ylabel="% of D attention wall time", ylim=(0, 100),
             xticks=range(len(friendly)), xticklabels=friendly)
    left.tick_params(axis="x", labelrotation=15)
    right.set(title="Observed future slack · offline oracle", ylabel="% of attention kernel starts",
              xlabel="Remaining gap required per start (µs)", ylim=(0, 100))
    right.legend(fontsize=8)
    fig.suptitle("Native attention opportunity · " + profile["evidence"])
    write(fig, target / f"native_encoder_opportunity{suffix}.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.4), layout="constrained", sharey=True)
    stages = ("worker_allocator_initial", "worker_allocator_after_load")
    for axis, stage, title in zip(axes, stages, ("Worker ready", "After qps 2.5 workload"), strict=True):
        measurements = profile[stage]
        parameters = np.array([measurements[role]["model_parameter_logical_bytes"] for role in role_names])
        kv = np.array([measurements[role]["unique_KV_storage_bytes"] for role in role_names])
        allocated = np.array([measurements[role]["allocated_bytes"] for role in role_names])
        reserved = np.array([measurements[role]["reserved_bytes"] for role in role_names])
        columns = {"Parameter payload": parameters, "Live KV": kv,
                   "Other live allocator": allocated - parameters - kv, "Reserved slack": reserved - allocated}
        bottom = np.zeros(len(role_names))
        for name, raw in columns.items():
            delta = raw / 2**30
            axis.bar(role_names, delta, bottom=bottom, label=name)
            bottom += delta
        high_water = np.array([measurements[role]["maximum_allocated_bytes"] for role in role_names]) / 2**30
        axis.scatter(role_names, high_water, marker="_", s=180, color="#211f1f", label="Allocator high water")
        axis.set(title=title, ylim=(0, 70))
    axes[0].set_ylabel("GiB")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Native EPD CUDA worker memory")
    write(fig, target / f"native_epd_memory{suffix}.png")


def plot_capacity(study, target):
    probes = [row for row in study["encoder_persistent_arena_probes"] if row["finished"]]
    e_native = study["native_peak_encoder_device_used_bytes"]
    if e_native is None or not probes:
        return
    prefill = study["post_warm_worker_memory"]["prefill"]
    p_warm = prefill["device_total_bytes"] - prefill["device_free_bytes"]
    resident = [row["allocator_receipt"]["size_bytes"] for row in probes]
    observed = [row["peak_encoder_device_used_bytes"] for row in probes]
    device_capacity = prefill["device_total_bytes"] / 2**30
    used_without_arena = np.array([e_native, p_warm, *[x - y for x, y in zip(observed, resident, strict=True)]]) / 2**30
    arena = np.array([0, 0, *resident]) / 2**30
    labels = ["E no arena\n(sampled peak)", "P no arena\n(warm snapshot)",
              *[f"E with {x / 2**30:g} GiB\n(sampled peak)" for x in resident]]
    fig, axis = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    axis.bar(labels, used_without_arena, label="Device used outside reserve", color="#175b77")
    axis.bar(labels, arena, bottom=used_without_arena, label="Physically touched reserve", color="#63a287")
    axis.axhline(device_capacity, color="#453830", ls="--", label="Reported device capacity")
    axis.set(ylabel="GiB", ylim=(0, device_capacity * 1.1), title="Native EPD Encoder memory co-residency")
    axis.legend(fontsize=8, loc="upper left")
    write(fig, target / "native_encoder_capacity.png")


def plot_capacity_by_role(encoder, prefill, target):
    labels = ("Encoder", "Prefill")
    stages = (encoder, prefill)
    roles = ("encoder", "prefill")
    warm = [study["post_warm_worker_memory"][role]["device_free_bytes"] / 2**30
            for study, role in zip(stages, roles, strict=True)]
    measured = [next(row for row in study[f"{role}_persistent_arena_probes"] if row["finished"])
                for study, role in zip(stages, roles, strict=True)]
    arenas = [row["allocator_receipt"]["size_bytes"] / 2**30 for row in measured]
    free = [row[f"{role}_worker_after_load"]["device_free_bytes"] / 2**30
            for row, role in zip(measured, roles, strict=True)]
    guard = measured[0]["allocator_receipt"]["physical_free_margin_gib"]
    fig, axis = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
    y = np.arange(len(labels))
    axis.barh(y - .17, warm, .31, color="#326d88", label="Free after Native warm cohort")
    axis.barh(y + .17, arenas, .31, color="#22806e", label="Arena held across 100 requests")
    offset = max(warm) * .01
    for idx, (size, leftover) in enumerate(zip(arenas, free, strict=True)):
        axis.text(warm[idx] - offset, idx - .17, f"{warm[idx]:.1f}",
                  va="center", ha="right", color="white", fontsize=9)
        axis.text(2 * offset, idx + .17, f"{size:.0f}", va="center", color="white", fontsize=9)
        axis.text(size - offset, idx + .17, f"{leftover:.1f} GiB free afterward",
                  va="center", ha="right", color="white", fontsize=9)
    axis.axvline(80 + guard, color="#ae5f47", ls="--", label="80 GiB request + 4 GiB guard")
    axis.set(title="Stock Native EPD · QPS2.5 physical co-residency", xlabel="GiB", yticks=y,
             yticklabels=labels, xlim=(0, max(warm) * 1.15), ylim=(-.55, 1.55))
    axis.invert_yaxis()
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.28), fontsize=8, ncol=2)
    write(fig, target / "native_encoder_prefill_capacity.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_scan(json.loads((args.results / "native_scan.json").read_text()), args.out_dir)
    name = args.results / "native_profile.json"
    if name.is_file():
        with (args.results / "native_profile_timeline.csv").open() as stream:
            timeline = list(csv.DictReader(stream))
        plot_profile(json.loads(name.read_text()), timeline, args.out_dir)
    reduced = args.results / "native_profile_low_observer.json"
    if reduced.is_file():
        with (args.results / "native_profile_low_observer_timeline.csv").open() as stream:
            timeline = list(csv.DictReader(stream))
        plot_profile(json.loads(reduced.read_text()), timeline, args.out_dir, suffix="_low_observer")
    capacity = args.results / "native_capacity_retry.json"
    if capacity.is_file():
        plot_capacity(json.loads(capacity.read_text()), args.out_dir)
    encoder = args.results / "native_encoder_capacity80.json"
    prefill = args.results / "native_prefill_capacity.json"
    if encoder.is_file() and prefill.is_file():
        plot_capacity_by_role(json.loads(encoder.read_text()), json.loads(prefill.read_text()), args.out_dir)


if __name__ == "__main__":
    main()
