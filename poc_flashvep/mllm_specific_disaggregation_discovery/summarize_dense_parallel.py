"""Create the Track B3 scalar summaries, CSV tables, and diagnostic figures."""

import argparse
import csv
import json
import statistics
from pathlib import Path

SEGMENTS = ("rms_qkv_mrope_ms", "attention_ms", "o_residual_ms", "mlp_ms")
GENERIC = ("generic_46", "generic_50", "generic_54")
LABELS = {
    "rms_qkv_mrope_ms": "RMS + QKV + MRoPE",
    "attention_ms": "Attention",
    "o_residual_ms": "O projection + residual",
    "mlp_ms": "RMS + MLP",
    "single_full": "Single GPU",
    "local_split_50": "Local cut 50%",
    "generic_46": "Generic cut 46%",
    "generic_50": "Generic cut 50%",
    "generic_54": "Generic cut 54%",
    "boundary_balanced": "Nearest image end",
    "last_image_then_remaining_tokens": "Last image end",
    "tp2_p2p_bf16_partials": "TP-shaped P2P",
    "cp2_partial_lse": "CP-shaped LSE",
    "pp2_serial_layers": "PP2 serial",
}


def csv_write(path, lines):
    if not lines:
        raise ValueError(f"No measurements for {path.name}")
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(lines[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)


def paired_delta(candidate, baseline):
    return statistics.median(100 * (x / y - 1) for x, y in zip(candidate, baseline))


def summaries(measurement):
    placements, segments, concise = [], [], []
    for case in measurement["cases"]:
        common = {k: case[k] for k in ("request_id", "source", "length", "image_count", "visual_tokens")}
        arms = case["arms"]
        baseline = min(GENERIC, key=lambda a: arms[a]["median_ms"])
        prefix = case["causal_negative_control"]["prefix"]
        controls = case["control_pairs"]
        wall = {name: row["wall_ms"] for name, row in arms.items()}
        for name, result in arms.items():
            cut = case["cuts"].get(name)
            if name in ("local_split_50", "cp2_partial_lse"):
                cut = case["cuts"]["generic_50"]
            placements.append(dict(
                **common, arm=name, prefix_tokens=cut,
                suffix_tokens=case["length"] - cut if cut is not None else None,
                median_ms=result["median_ms"], p95_ms=result["p95_ms"],
                logical_peer_bytes=result["bytes_p2p_logical"],
                peak_extra_gpu0_bytes=result["extra_peak_allocated_bytes_by_device"][0],
                peak_extra_gpu1_bytes=result["extra_peak_allocated_bytes_by_device"][1],
                max_rank_extra_peak_bytes=max(result["extra_peak_allocated_bytes_by_device"]),
                causal_prefix_invariant=prefix["changed_fraction"] == 0,
                l2=result["state_difference_vs_single_full"]["relative_l2"],
                changed_fraction=result["state_difference_vs_single_full"]["changed_fraction"],
                median_pair_delta_vs_fastest_generic_pct=paired_delta(wall[name], wall[baseline]),
            ))
        for name, samples in case["segments"].items():
            medians = {segment: statistics.median(row[segment] for row in samples)
                       for segment in SEGMENTS}
            if name == "full":
                size = case["length"]
            elif name.endswith("_label_only"):
                size = case["cuts"]["generic_50"]
            else:
                size = case["content_control_rows"]
            segments.append(dict(**common, label=name, query_rows=size, **medians))

        def segment_delta(label_a, label_b):
            if label_a not in case["segments"]:
                return None
            a, b = case["segments"][label_a], case["segments"][label_b]
            return {
                key: paired_delta([s[key] for s in a], [s[key] for s in b]) for key in SEGMENTS
            }

        concise.append({
            **common, "fastest_timed_generic": baseline, "causal_prefix": prefix,
            "same_cut_control": controls,
            "label_only_vision_minus_text_pct": segment_delta("vision_label_only", "text_label_only"),
            "real_vision_minus_nonvision_content_pct": segment_delta(
                "actual_vision_rows_compacted", "actual_nonvision_rows_compacted"),
            "boundary_minus_fastest_generic_pct": paired_delta(wall["boundary_balanced"], wall[baseline])
            if "boundary_balanced" in wall else None,
            "terminal_boundary_minus_fastest_generic_pct": paired_delta(
                wall["last_image_then_remaining_tokens"], wall[baseline])
            if "last_image_then_remaining_tokens" in wall else None,
            "fastest_generic_minus_single_gpu_pct": paired_delta(wall[baseline], wall["single_full"]),
            "max_rank_peak_boundary_minus_fastest_generic_pct": 100 * (
                max(arms["boundary_balanced"]["extra_peak_allocated_bytes_by_device"])
                / max(arms[baseline]["extra_peak_allocated_bytes_by_device"]) - 1)
            if "boundary_balanced" in arms else None,
        })
    return placements, segments, concise


def plots(placements, segments, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    requests = list(dict.fromkeys(row["request_id"] for row in placements))
    names = ("single_full", "generic_50", "boundary_balanced", "last_image_then_remaining_tokens",
             "tp2_p2p_bf16_partials", "cp2_partial_lse", "pp2_serial_layers")
    by_name = {row["request_id"]: row for row in placements if row["arm"] == "single_full"}
    groups = {request: {row["arm"]: row["median_ms"] for row in placements if row["request_id"] == request}
              for request in requests}
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = .79 / len(names)
    for offset, name in enumerate(names):
        xx, yy = [], []
        for index, request in enumerate(requests):
            if name in groups[request]:
                xx.append(index + width * (offset - (len(names) - 1) / 2))
                yy.append(groups[request][name] / groups[request]["single_full"])
        ax.bar(xx, yy, width=width, label=LABELS[name])
    ax.axhline(1, color="#555555", linewidth=.8)
    ax.set_xticks(range(len(requests)), [f"{r.split('__')[-1]}\n{by_name[r]['length']:,} tokens" for r in requests])
    ax.set_ylabel("Median wall time / single GPU")
    ax.legend(ncols=4, fontsize=8)
    ax.grid(axis="y", linewidth=.5, alpha=.35)
    fig.tight_layout()
    fig.savefig(out / "b3_communication_inclusive_placement.png", dpi=185)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    full = [row for row in segments if row["label"] == "full"]
    for idx, axis in enumerate(axes):
        labels = ("vision_label_only", "text_label_only") if idx == 0 else (
            "actual_vision_rows_compacted", "actual_nonvision_rows_compacted")
        for label in labels:
            matching = [row for row in segments if row["label"] == label]
            if not matching:
                continue
            matching.sort(key=lambda x: x["query_rows"])
            x = [row["query_rows"] for row in matching]
            y = [sum(row[segment] for segment in SEGMENTS) for row in matching]
            axis.scatter(x, y, marker="x" if label == labels[-1] else "o", label={
                "vision_label_only": "Vision label",
                "text_label_only": "Text label",
                "actual_vision_rows_compacted": "Vision rows",
                "actual_nonvision_rows_compacted": "Non-vision rows",
            }[label])
        if idx == 0:
            axis.set_xscale("log")
        axis.set_ylabel("Sum of CUDA event envelopes (ms)")
        axis.set_xlabel("Query rows with equal shapes within each comparison")
        axis.legend(fontsize=8)
        axis.grid(linewidth=.5, alpha=.35)
    fig.tight_layout()
    fig.savefig(out / "b3_matched_shape_kernel_controls.png", dpi=185)
    plt.close(fig)

    full.sort(key=lambda x: x["length"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for segment in SEGMENTS:
        ax.plot([row["length"] for row in full], [row[segment] for row in full],
                marker="o", label=LABELS[segment])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Original frozen prompt length")
    ax.set_ylabel("Median CUDA event envelope (ms)")
    ax.legend(fontsize=8)
    ax.grid(linewidth=.5, alpha=.35)
    fig.tight_layout()
    fig.savefig(out / "b3_dense_operator_shapes.png", dpi=185)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    measured = json.loads(args.raw.read_text())
    placement, segments, concise = summaries(measured)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_write(args.out / "placement.csv", placement)
    csv_write(args.out / "segments.csv", segments)
    (args.out / "controls.json").write_text(json.dumps(concise, indent=2) + "\n")
    if args.plots:
        plots(placement, segments, args.out)


if __name__ == "__main__":
    main()
