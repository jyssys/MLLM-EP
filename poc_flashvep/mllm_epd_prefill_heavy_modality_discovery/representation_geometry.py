"""Calculate logical route payloads and image-boundary geometry from frozen CPU data."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median


BLOCK_ROWS = 128
MIB = 1024 ** 2
TARGETS = (0, 25, 50, 75, 95)


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def payloads(input_rows, visual_rows, geometry):
    embedding = visual_rows * geometry["lm_embedding_bytes_per_row"]
    kv = input_rows * geometry["all_layer_kv_bytes_per_row"]
    return {
        "embedding": embedding,
        "full_prompt_kv": kv,
        "full_prompt_kv_page_backing": round_up_rows(input_rows) * geometry["all_layer_kv_bytes_per_row"],
        "native_inter_gpu": embedding + kv,
        "d_local_inter_gpu": embedding,
        "e_helper_inter_gpu": kv,
        "e_helper_same_gpu_embedding_clone": embedding,
    }


def round_up_rows(rows):
    return (rows + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS


def shape_geometry(config):
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    dimension, remainder = divmod(hidden, heads)
    if remainder or config.get("use_sliding_window") or config["vision_config"]["out_hidden_size"] != hidden:
        raise ValueError("This accounting requires the audited dense Qwen2.5-VL geometry")
    kv_layer = 2 * config["num_key_value_heads"] * dimension * 2
    return {
        "dtype": "BF16", "hidden_size": hidden, "heads": heads,
        "kv_heads": config["num_key_value_heads"], "head_dim": dimension,
        "decoder_layers": config["num_hidden_layers"], "page_rows": BLOCK_ROWS,
        "lm_embedding_bytes_per_row": hidden * 2, "single_layer_kv_bytes_per_row": kv_layer,
        "all_layer_kv_bytes_per_row": config["num_hidden_layers"] * kv_layer,
    }


def summarize(rows, geometry):
    per_request = [payloads(row["input_tokens"], row["visual_tokens"], geometry) for row in rows]
    return {
        "requests": len(rows),
        "median_input_rows": median(row["input_tokens"] for row in rows),
        "median_visual_rows": median(row["visual_tokens"] for row in rows),
        "median_images": median(row["image_count"] for row in rows),
        "median_bytes_per_request": {field: median(item[field] for item in per_request) for field in per_request[0]},
        "median_full_kv_to_embedding_ratio": median(
            item["full_prompt_kv"] / item["embedding"] for item in per_request if item["embedding"]),
    }


def image_boundaries(rows, block):
    counts = Counter()
    backtracks = []
    requests_with_interior = set()
    for row in rows:
        spans = [image["visual_span"] for image in row["images"]]
        for cut in range(block, row["input_tokens"], block):
            counts["fixed_generic_cuts"] += 1
            image_start = next((left for left, right in spans if left < cut < right), None)
            if image_start is None:
                continue
            counts["cuts_inside_image"] += 1
            counts["first_cut_inside_image"] += cut == block
            backtracks.append(cut - image_start)
            requests_with_interior.add(row["request_id"])
    return {**counts, "requests_with_any_interior_cut": len(requests_with_interior),
            "median_rows_if_snapping_interiors_backward": median(backtracks) if backtracks else None,
            "max_rows_if_snapping_interiors_backward": max(backtracks, default=0)}


def figure(points, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = [point["visual_rows"] * 100 / point["input_rows"] for point in points]
    fig, ax = plt.subplots(figsize=(7.5, 3.95))
    for key, label, color in (
        ("native_inter_gpu", "Native E→P→D", "#355f84"),
        ("e_helper_inter_gpu", "Co-resident E Prefill → D", "#8c547a"),
        ("d_local_inter_gpu", "D-local E→D", "#077a63"),
    ):
        ax.plot(coords, [point["bytes"][key] / MIB for point in points],
                color=color, label=label, marker="o", markersize=4, linewidth=2)
    ax.set(title="Qwen2.5-VL-7B route payloads at 16K input",
           xlabel="Actual median visual share (%)", ylabel="Logical inter-GPU payload (MiB)")
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center right", frameon=False, fontsize=9)
    fig.text(.12, .025, "BF16 shape accounting. The E-side same-GPU embedding clone and wire costs are excluded.",
             fontsize=8, color="#404850")
    fig.subplots_adjust(left=.12, right=.98, top=.88, bottom=.22)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    audit, audit_hash = load(args.audit)
    receipts, receipts_hash = load(args.receipts)
    config, config_hash = load(args.config)
    geometry = shape_geometry(config)
    index = {row["request_id"]: row for row in audit}
    cohorts = {
        "main400": audit,
        "long_natural_n24": [index[key] for key in receipts["natural_very_long_n24"]["request_ids"]],
    }
    points = []
    for target in TARGETS:
        item = receipts[f"controlled_16384_v{target:02d}_n24"]
        input_rows, visual_rows = (item[key]["p50"] for key in ("input_tokens", "visual_tokens"))
        points.append({"visual_target_percent": target, "input_rows": int(input_rows),
                       "visual_rows": int(visual_rows), "requests": item["requests"],
                       "bytes": payloads(int(input_rows), int(visual_rows), geometry)})
    output = {
        "evidence": "DIAGNOSTIC",
        "scope": "Logical payload and exact prompt-geometry accounting. The fixed-shape sweep uses "
                 "separate rounded median input and visual sizes to display one shape per cohort.",
        "source_sha256": {"audit": audit_hash, "receipts": receipts_hash, "actual_model_config": config_hash},
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "geometry": geometry,
        "natural_cohorts": {name: summarize(rows, geometry) for name, rows in cohorts.items()},
        "fixed_16k_geometry": points,
        "repeated_fixed_chunk_geometry": {
            name: {chunk: image_boundaries(rows, chunk) for chunk in (4096, 8192)}
            for name, rows in cohorts.items()
        },
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.figure:
        figure(points, args.figure)


if __name__ == "__main__":
    main()
