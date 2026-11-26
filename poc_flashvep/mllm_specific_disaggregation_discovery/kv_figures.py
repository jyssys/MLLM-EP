"""Render KV capacity, decode cost, two-GPU correctness, and analytical break-even."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {"local": "#35618d", "balanced": "#369585", "prefix": "#9b6c3d",
           "visual": "#b74968", "equal_count_prefix": "#8e74b3"}
LABELS = {"local": "Full local", "balanced": "Balanced generic", "prefix": "Fixed prompt",
          "visual": "Visual only", "equal_count_prefix": "Same-size generic"}


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=170)
    plt.close(fig)


def figure7(rows, out):
    groups = defaultdict(list)
    for row in rows:
        groups[row["study_source"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    names = list(groups)
    floor = np.zeros(len(names))
    for key, label, color in (
        ("visual_rows", "Visual", "#35618d"), ("text_rows", "Prompt text", "#369585"),
        ("special_rows", "Special", "#9b6c3d"), ("possible_generated_kv_rows", "Generated", "#b74968"),
        ("expected_D_final_waste_rows", "Expected block padding", "#b7bcc2"),
    ):
        values = [np.mean([float(row[key]) * int(row["bytes_per_row"]) / (1 << 20)
                           for row in groups[name]]) for name in names]
        axes[0].bar(names, values, bottom=floor, label=label, color=color)
        floor += values
    axes[0].set_ylabel("Mean logical or padded KV MiB")
    axes[0].tick_params(axis="x", labelrotation=24)
    axes[0].legend(fontsize=7, frameon=False)
    for name in names:
        items = groups[name]
        axes[1].scatter([int(item["prefill_rows"]) for item in items],
                        [int(item["observed_exported_prompt_pages"]) for item in items], label=name, s=9, alpha=.7)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Prompt tokens")
    axes[1].set_ylabel("Observed exported Block128 IDs")
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, "figure7_visual_kv_memory")


def figure8(rows, batch, out):
    local = sorted((row for row in rows if row["layout"] == "local"), key=lambda row: int(row["prompt_rows"]))
    names = [f'{int(row["prompt_rows"]):,}' for row in local]
    b4_components = {}
    if batch:
        names.append("Ragged B4")
        b4_components = defaultdict(float)
        for layer in batch["layouts"]["local"]["diagnostic_profile"]["layers"][0]:
            for name, value in layer.items():
                b4_components[name] += value
    fig, axis = plt.subplots(figsize=(7.7, 3.8))
    floor = np.zeros(len(names))
    for key, label, color in (
        ("qkv_norm_rope", "QKV, norm, MRoPE", "#35618d"), ("write_kv", "Write KV", "#6c87a8"),
        ("attention_total", "FA3 attention", "#369585"),
        ("output_projection_residual", "Output projection, residual", "#9b6c3d"),
        ("mlp_norm_residual", "MLP, norm, residual", "#b74968"),
    ):
        values = [float(row[f"profile_{key}_ms_per_step"]) for row in local]
        if batch:
            values.append(b4_components[key])
        axis.bar(names, values, bottom=floor, label=label, color=color)
        floor += values
    axis.set_xlabel("Actual prompt rows (batch one at left)")
    axis.set_ylabel("Full28 profiled GPU interval, ms/step")
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out, "figure8_decode_cost")


def figure9(cases, batch, out):
    cells = [(f'B1 {case["input_tokens"]:,}', case["layer_1_attention"]) for case in cases]
    if batch:
        cells.append(("Ragged B4", batch))
    names = [label for label, _ in cells]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    for index, (layout, color) in enumerate(PALETTE.items()):
        arms = [cell["layouts"][layout] for _, cell in cells]
        axes[0].bar(x + (index - 2) * .16, [1e3 * median(arm["CUDA_ms"]) for arm in arms],
                    width=.155, color=color, label=LABELS[layout])
        axes[1].bar(x + (index - 2) * .16, [100 * arm["fp32_reference_error"]["relative_l2"] for arm in arms],
                    width=.155, color=color)
    for axis in axes:
        axis.set_xticks(x, names, rotation=25)
    axes[0].set_ylabel("Attention + transfer + merge, µs")
    axes[1].set_ylabel("Relative L2 vs FP32, %")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, "figure9_distributed_attention")


def figure10(rows, out):
    visual = [row for row in rows if row["layout"] == "visual"]
    fig, axis = plt.subplots(figsize=(7.8, 4.2))
    steps = np.geomspace(1, 2048, 220)
    for row in sorted(visual, key=lambda row: int(row["prompt_rows"])):
        initial = float(row["initial_saved_ms_vs_full_copy_after_packing"])
        rate = float(row["wall_delta_ms_per_full28_step"])
        native_steps = max(1, int(row["native_output_tokens"]) - 1)
        line, = axis.plot(steps, initial - steps * rate, label=f'{int(row["prompt_rows"]):,} prompt')
        axis.scatter([native_steps], [initial - native_steps * rate], color=line.get_color(), s=25)
    axis.axhline(0, color="#555555", lw=.8)
    axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=1)
    axis.set_xlabel("Generated tokens consumed by decode")
    axis.set_ylabel("Modeled net ms saved vs full copy")
    axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, out, "figure10_visual_memory_break_even")


def figure11(rows, out):
    labels = list(dict.fromkeys(row["request_id"] for row in rows))
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cases = {}
    for index, (layout, color) in enumerate(PALETTE.items()):
        cases = {row["request_id"]: row for row in rows if row["layout"] == layout}
        offset = (index - 2) * .16
        axes[0].bar(x + offset, [float(cases[key]["wall_median_ms_per_full28_step"]) for key in labels],
                    width=.155, color=color, label=LABELS[layout])
        axes[1].bar(x + offset, [int(cases[key]["reserved_D_bytes"]) / (1 << 20) for key in labels],
                    width=.155, color=color)
    for axis in axes:
        axis.set_xticks(x, [f'{int(cases[key]["prompt_rows"]):,}' for key in labels], rotation=25)
        axis.set_xlabel("Prompt tokens")
    axes[0].set_ylabel("Full28 isolated time, ms/step")
    axes[1].set_ylabel("D allocated KV MiB")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, "figure11_modality_vs_generic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    kv = table(args.data / "native_kv.csv")
    costs = table(args.data / "decode_cost_and_break_even.csv")
    cases = [json.loads(path.read_text()) for path in sorted(args.attention.glob("case_*.json"))]
    batch = args.attention / "ragged_batch4_attention.json"
    figure7(kv, args.out)
    full28 = args.attention / "ragged_batch4_full28.json"
    figure8(costs, json.loads(full28.read_text()) if full28.is_file() else None, args.out)
    figure9(cases, json.loads(batch.read_text()) if batch.is_file() else None, args.out)
    figure10(costs, args.out)
    figure11(costs, args.out)


if __name__ == "__main__":
    main()
