"""Reduce physical representation and transfer measurements and render their figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


MIB = 2**20
GIB = 2**30


def combined_transfer(rows):
    records = []
    for row in rows:
        count = row["samples"][0]["copies"]
        gpu_ms = median(point["gpu_ms"] / point["copies"] for point in row["samples"])
        wall_ms = median(point["wall_ms"] / point["copies"] for point in row["samples"])
        records.append({"source": row["source"], "destination": row["destination"], "bytes": row["bytes"],
                        "size_labels": row["size_labels"], "gpu_ms_per_copy": gpu_ms,
                        "wall_ms_per_copy": wall_ms, "copies_per_sample": count,
                        "gpu_GiB_per_s": 1000 * row["bytes"] / (GIB * gpu_ms)})
    return records


def combined_interference(rows):
    records = []
    for row in rows:
        overlapping = [entry for entry in row["samples"] if entry["with_copy"]["enclosure_overlap_ms"] > 0]
        metrics = {"active_stage": row["active_stage"], "source": row["source"], "destination": row["destination"],
                   "bytes_per_copy": row["bytes_per_copy"], "samples": len(row["samples"]),
                   "positive_overlap_samples": len(overlapping)}
        if overlapping:
            metrics.update({
                "median_overlap_ms": median(s["with_copy"]["enclosure_overlap_ms"] for s in overlapping),
                "median_copy_ms_ratio": median(s["with_copy"]["copy"]["gpu_ms"] / s["isolated_copy"]["gpu_ms"]
                                              for s in overlapping),
                "median_stage_ms_ratio": median(s["with_copy"]["compute"]["gpu_ms"] /
                                               s["isolated_compute"]["compute"]["gpu_ms"] for s in overlapping),
            })
        records.append(metrics)
    return records


def economics(cases, transfers):
    costs = {(row["source"], row["destination"], row["bytes"]): row["gpu_ms_per_copy"] for row in transfers}
    result = []
    for row in cases:
        embedding = row["R3_lm_ready_embedding_bytes"]
        prefix = row["R4_image_complete_prefix_kv_bytes"]
        e_saved = row["median_recompute"]["encode"]["gpu_ms"]
        p_saved = row["median_recompute"]["image_complete_prefix_prefill"]["gpu_ms"]
        staged = costs["E", "P", prefix]
        pe, ed = costs["P", "E", prefix], costs["E", "D", prefix]
        result.append({
            "request_id": row["request_id"], "image_count": len(row["images"]), "prompt_rows": row["prompt_rows"],
            "embedding_bytes": embedding, "prefix_kv_bytes": prefix, "prefix_to_emb_ratio": prefix / embedding,
            "encode_ms": e_saved, "image_prefix_prefill_ms": p_saved,
            "E_to_P_embedding_ms": costs["E", "P", embedding], "CPU_to_P_embedding_ms": costs["CPU", "P", embedding],
            "E_to_P_prefix_kv_ms": staged, "CPU_to_P_prefix_kv_ms": costs["CPU", "P", prefix],
            "embedding_saved_stage_ms_per_GiB": e_saved * GIB / embedding,
            "prefix_saved_stage_ms_per_GiB": (e_saved + p_saved - staged) * GIB / prefix,
            "pending_direct_P_D_ms": costs["P", "D", prefix], "pending_sum_P_E_D_ms": pe + ed,
            "preemption_sum_D_E_D_ms": costs["D", "E", prefix] + ed,
            "preemption_sum_D_CPU_D_ms": costs["D", "CPU", prefix] + costs["CPU", "D", prefix],
        })
    return result


def write_csv(path, rows, keys=None):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, keys or rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(summary, directory):
    from matplotlib import pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    cases = summary["representations"]
    labels = [f"{len(row['images'])} img\n{row['prompt_rows']:,} tokens" for row in cases]
    x = list(range(len(cases)))
    colors = ("#7d8795", "#e9a632", "#ad5992", "#167b80", "#183c75")

    def save(fig, name):
        fig.savefig(directory / name, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.3, 4.7), layout="constrained")
    specs = [
        ("R0_raw_file_bytes", "Compressed image files"),
        ("R1_processed_pixel_bf16_bytes", "BF16 encoder input"),
        ("R2_encoder_pre_merger_bytes", "Before visual merger"),
        ("R3_lm_ready_embedding_bytes", "LM-ready embeddings"),
        ("R4_image_complete_prefix_kv_bytes", "Image-complete prefix KV"),
    ]
    for j, ((key, name), color) in enumerate(zip(specs, colors, strict=True)):
        ax.bar([i + (j - 2) * .15 for i in x], [row[key] / MIB for row in cases], .149, color=color, label=name)
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("MiB per request")
    ax.set_title("Actual Qwen representations by request")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -.13))
    ax.grid(axis="y", alpha=.2)
    save(fig, "representation_sizes.png")

    data = summary["economics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained", sharey=True)
    for ax, pairs, headline in zip(axes, (("encode_ms", "E_to_P_embedding_ms"),
                                       ("image_prefix_prefill_ms", "E_to_P_prefix_kv_ms")),
                                        ("Vision embeddings", "Image-complete prefix KV"), strict=True):
        for j, (key, name, color) in enumerate(zip(pairs, ("Local recompute", "E→P copy"),
                                                   ("#183c75", "#e49a29"), strict=True)):
            ax.bar([i + (j - .5) * .32 for i in x], [row[key] for row in data], .31, label=name, color=color)
        ax.set_xticks(x, labels)
        ax.set_yscale("log")
        ax.set_title(headline)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("GPU event time (ms, log scale)")
    fig.suptitle("Isolated recompute and contiguous peer retrieval")
    save(fig, "recompute_vs_transfer.png")

    fig, ax = plt.subplots(figsize=(8.7, 4.4), layout="constrained")
    for j, (key, name, color) in enumerate((
            ("embedding_saved_stage_ms_per_GiB", "Embeddings", "#167b80"),
            ("prefix_saved_stage_ms_per_GiB", "Prefix KV", "#183c75"))):
        ax.bar([i + (j - .5) * .32 for i in x], [row[key] for row in data], .31, color=color, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Potential stage time saved (ms/GiB)")
    ax.set_title("At a valid hit, before queue and reuse probabilities")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    save(fig, "cache_utility_per_gib.png")

    points = [row for row in summary["interference"] if row["positive_overlap_samples"]]
    if points:
        fig, ax = plt.subplots(figsize=(11.2, 8.1), layout="constrained")
        y = list(range(len(points)))
        labels = [f"{row['active_stage']} active · {row['source']}→{row['destination']} · "
                  f"{row['bytes_per_copy'] // MIB} MiB" for row in points]
        for key, name, color, marker in (("median_stage_ms_ratio", "Stage forward", "#183c75", "o"),
                                          ("median_copy_ms_ratio", "Transfer burst", "#e49a29", "s")):
            ax.scatter([100 * (row[key] - 1) for row in points], y, label=name, color=color, marker=marker)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Change from matched isolated CUDA-event time (%)")
        ax.axvline(0, color="#666666", linewidth=.8)
        ax.set_title("Copy bursts with measured stage-enclosure overlap")
        ax.grid(axis="x", alpha=.2)
        ax.legend(frameon=False, loc="lower right")
        save(fig, "transfer_interference.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    profiles = [json.loads(line) for line in raw.splitlines()]
    if profiles[-1]["kind"] != "complete":
        parser.error("The measurement has no completion record")
    transfers = combined_transfer([row for row in profiles if row["kind"] == "transfer"])
    reps = [row for row in profiles if row["kind"] == "representation"]
    interference = combined_interference([row for row in profiles if row["kind"] == "interference"])
    result = {"raw_sha256": hashlib.sha256(raw).hexdigest(), "config": profiles[0],
              "model_allocation": next(row for row in profiles if row["kind"] == "model_allocation"),
              "representations": reps, "transfers": transfers, "interference": interference,
              "economics": economics(reps, transfers)}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.csv_dir:
        args.csv_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_dir / "physical_transfer.csv", transfers)
        write_csv(args.csv_dir / "representation_utility.csv", result["economics"])
        if interference:
            columns = list(dict.fromkeys(key for row in interference for key in row))
            write_csv(args.csv_dir / "transfer_interference.csv", interference, columns)
    if args.figure_dir:
        render(result, args.figure_dir)


if __name__ == "__main__":
    main()
