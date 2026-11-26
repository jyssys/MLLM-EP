"""Plot measured H200 suffix calibration and frozen multimodal metadata."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from .historical_figures import csv_rows, persist, plt

COLORS = ("#2e6277", "#b15349", "#549b74", "#77639c", "#b4823f", "#308b92")


def suffix_cost(path, output):
    cells = defaultdict(lambda: defaultdict(list))
    rows = csv_rows(path)
    transports = {row.get("transport", "on_demand") for row in rows}
    if len(transports) != 1:
        raise ValueError("Plot one descriptor transport per calibration epoch")
    transport = transports.pop()
    for row in rows:
        if row["series"] == "cost":
            cells[(row["request_id"], int(row["input_tokens"]))][int(row["suffix_tokens"])].append(
                float(row["D_first_token_http_ms"]))
    if not cells:
        raise ValueError("Suffix calibration has no cost arms")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for (request, length), points in sorted(cells.items(), key=lambda item: item[0][1]):
        tokens = sorted(points)
        costs = [median(points[size]) for size in tokens]
        target = axes[0] if length < 16_000 else axes[1]
        target.plot(tokens, costs, marker="o", markersize=3, label=f"{length:,} · {request.split('__')[-1]}")
    for ax, title in zip(axes, ("Inputs below 16k", "Inputs at or above 16k"), strict=True):
        ax.grid(alpha=.22)
        ax.set(title=title, xlabel="Actual suffix tokens", ylabel="D time to one output token (ms)")
        ax.legend(title="Input size · request", fontsize=7, frameon=False)
    fig.suptitle(f"H200 isolated suffix cost · {transport.replace('_', ' ')} · imported KV")
    persist(fig, output, "fig03_h200_suffix_cost")
    return {"figure": 3, "request_geometries": len(cells), "evidence": "ISOLATED_GPU_MEASUREMENT", "transport": transport,
            "scope": "Each point is the median of isolated trials with the specified actual geometry."}


def paired_work(path, output):
    report = json.loads(path.read_text())
    transport = report.get("transport", "on_demand")
    arms = ("intra_image", "image_boundary")
    fields = (
        (lambda arm: arm.get("EC_duplicate_image_clones"), "Duplicate image clone count", 1),
        (lambda arm: arm.get("EC_duplicate_cloned_tensor_payload_bytes"), "Duplicate tensor payload (MiB)", 2**20),
        (lambda arm: (arm.get("NVLink_D_stage_device_delta") or {}).get("E_TX_bytes"),
         "E NVLink TX during D stage (MiB)", 2**20),
        (lambda arm: arm.get("D_first_token_http_ms"), "D time to one output token (ms)", 1),
        (lambda arm: arm.get("producer_host_marker_lifetime_max_ms"), "Max producer host-marker lifetime (ms)", 1),
        (lambda arm: arm.get("producer_buffer_storage_snapshot_bytes"), "Producer storage snapshot (MiB)", 2**20),
    )
    grouped = defaultdict(list)
    for pair in report["pairs"]:
        grouped[pair["request_id"]].append(pair["arms"])
    if not grouped:
        raise ValueError("No intra-image and image-boundary measurement pairs")
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.2), layout="constrained")
    for ax, (load, label, divisor) in zip(axes.flat, fields, strict=True):
        plotted = 0
        for index, request in enumerate(sorted(grouped)):
            observations = grouped[request]
            observed = [[value for arm in observations if (value := load(arm[name])) is not None] for name in arms]
            if not all(observed):
                continue
            plotted += 1
            values = [median(points) / divisor for points in observed]
            ax.plot([0, 1], values, "-o", color=COLORS[index % len(COLORS)], alpha=.7)
        ax.set(xticks=[0, 1], xticklabels=["Intra-image", "Image boundary"], ylabel=label)
        ax.grid(axis="y", alpha=.2)
        if not plotted:
            ax.text(.5, .5, "NOT_MEASURED", transform=ax.transAxes, ha="center", va="center")
    fig.suptitle(f"Paired cuts · {transport.replace('_', ' ')} · same request/KV blocks · one-token difference")
    persist(fig, output, "fig04_image_cut_work_matched")
    return {"figure": 4, "request_pairs": len(grouped), "isolated_repeat_pairs": len(report["pairs"]), "transport": transport,
            "evidence": report["evidence"], "scope": "Lines are request-level medians across paired trials. "
            "Clone payload and device-wide NVLink TX are separate observations. Storage is a snapshot."}


def predictor_fit(path, output):
    model = json.loads(path.read_text())
    transport = model.get("transport", "on_demand")
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 4), layout="constrained")
    high = 1.1 * max(max(row["actual_ms"], row["prediction_ms"]) for family in ("generic", "modality")
                    for row in model[family]["validation_points"])
    for ax, family in zip(axes[:2], ("generic", "modality"), strict=True):
        fit = model[family]
        rows = fit["validation_points"]
        ax.scatter([row["actual_ms"] for row in rows], [row["prediction_ms"] for row in rows],
                   s=13, alpha=.56, color=COLORS[family == "modality"])
        ax.plot([0, high], [0, high], color="#5f6870", linewidth=1, linestyle=":")
        ax.set(xlim=(0, high), ylim=(0, high), aspect="equal",
               xlabel="Measured D one-token HTTP (ms)", ylabel="Predicted D one-token HTTP (ms)",
               title=f"{family.capitalize()} · MAE {fit['validation_MAE_ms']:.1f} ms")
        ax.text(.04, .94, f"p95 underprediction {fit['validation_p95_underprediction_ms']:.1f} ms",
                transform=ax.transAxes, va="top", fontsize=8)
    for index, family in enumerate(("generic", "modality")):
        safety = model[family]["validation_safety_at_fixed_budgets"]
        grid = sorted(((int(budget), row) for budget, row in safety.items()))
        for metric, marker, linestyle, label in (
            ("false_safe_rate_among_predictions", "o", "-", "false-safe"),
            ("false_reject_rate_among_actual_safe", "s", ":", "excess rejection"),
        ):
            rates = [100 * row[metric] if row[metric] is not None else float("nan") for _, row in grid]
            axes[2].plot([budget for budget, _ in grid], rates, marker=marker, ls=linestyle, markersize=3,
                         color=COLORS[index], label=f"{family} · {label}")
    axes[2].set(title="Offline fixed-budget checks", xlabel="Budget (ms)", ylabel="Rate (%)", ylim=(0, 105))
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Same predictor family · {transport.replace('_', ' ')} · isolated heldout")
    persist(fig, output, "fig05_generic_modality_cost_prediction")
    return {"figure": 5, "evidence": model["evidence"], "scope": model["predictor_scope"], "transport": transport,
            "heldout_requests": model["heldout_calibration_requests"]}


def structure(row):
    if row["stratum"] == "long_document":
        return "Long document"
    if row["image_count"] == 1:
        return "Single image"
    if row["image_count"] <= 4:
        return "Few images (2–4)"
    return "Many images (5+)"


def structure_transfer(transfer_path, inventory_path, output):
    data = json.loads(transfer_path.read_text())
    inventory = {row["request_id"]: row for row in json.loads(inventory_path.read_text())["Calibration32"]}
    grouped = defaultdict(list)
    for pair in data["pairs"]:
        intra, edge = (pair["arms"][name] for name in ("intra_image", "image_boundary"))
        a, b = [arm["NVLink_D_stage_device_delta"] for arm in (intra, edge)]
        if a is not None and b is not None:
            grouped[pair["request_id"]].append((
                (a["E_TX_bytes"] - b["E_TX_bytes"]) / 2**20,
                intra["D_first_token_http_ms"] - edge["D_first_token_http_ms"],
            ))
    figures = defaultdict(list)
    for request_id, pairs in grouped.items():
        metadata = inventory[request_id]
        tx, http = map(median, zip(*pairs, strict=True))
        figures[structure(metadata)].append({
            "request_id": request_id, "input_tokens": metadata["input_tokens"],
            "images": metadata["image_count"], "TX_difference_MiB": tx, "D_HTTP_difference_ms": http,
        })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.75), layout="constrained")
    for color, (name, rows) in zip(COLORS, sorted(figures.items())):
        for ax, field in zip(axes, ("input_tokens", "TX_difference_MiB"), strict=True):
            ax.scatter([row[field] for row in rows], [row["D_HTTP_difference_ms"] for row in rows],
                       color=color, label=f"{name} ({len(rows)})", s=33)
    for ax, label in zip(axes, ("Original request input tokens", "Additional E NVLink TX (MiB)"), strict=True):
        ax.axhline(0, color="#5f6870", linewidth=.8, linestyle=":")
        ax.set(xlabel=label, ylabel="Intra − edge D first-token HTTP (ms)")
        ax.grid(alpha=.18)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Inline redirect · within-request image-cut contrasts by structure")
    persist(fig, output, "fig06_image_structure_transfer")
    return {"figure": 6, "evidence": data["evidence"], "request_medians": figures,
            "scope": "Median of paired, one-token-difference repeats for each request. "
                     "Input lengths differ across structures. No cross-stratum causal or serving claim.",
            "transfer_sha256": hashlib.sha256(transfer_path.read_bytes()).hexdigest(),
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest()}


def workload_structure(path, output, replay=None):
    data = json.loads(path.read_text())
    lengths = {row["request_id"]: int(row["natural_output_tokens"]) for row in csv_rows(replay)} if replay else {}
    if replay and (set(lengths) != {row["request_id"] for row in data["Main100"]}
                   or sum(lengths.values()) != 5269):
        raise ValueError("Expected frozen Main100 replay roster and 5,269 output tokens")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4), layout="constrained")
    groups = ("Single image", "Few images (2–4)", "Many images (5+)", "Long document")
    counts = {}
    for ax, set_name in zip(axes, ("Main100", "Calibration32"), strict=True):
        counts[set_name] = {}
        for color, label in zip(COLORS, groups):
            rows = [item for item in data[set_name] if structure(item) == label]
            counts[set_name][label] = len(rows)
            y = ([lengths[row["request_id"]] for row in rows] if lengths and set_name == "Main100"
                 else [row["image_count"] for row in rows])
            ax.scatter([row["input_tokens"] for row in rows], y,
                       c=color, alpha=.74, s=26, label=f"{label} ({len(rows)})")
        ylabel = "Natural output tokens · replay length reference" if lengths and set_name == "Main100" else "Image count"
        ax.set(title=set_name, xlabel="Frozen input tokens", ylabel=ylabel, yscale="log")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.22)
    fig.suptitle("Input geometry across image structures · frozen workload metadata")
    persist(fig, output, "fig06_frozen_workload_structure")
    return {"figure": 6, "scope": "Counts use frozen metadata, with long-document stratum taking precedence. "
            "If provided, Main100 output lengths use the historical replay reference. "
            "The panel describes workload geometry.", "counts": counts,
            "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest() if replay else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="Analysis path holding finalized receipts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="Frozen Main100 replay length table")
    parser.add_argument("--inventory", type=Path, help="Render the inline transfer-by-structure panel with this metadata")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plans = (
        (3, "suffix_calibration.csv", suffix_cost),
        (4, "modality_matched_transfer.json", paired_work),
        (5, "suffix_cost_model.json", predictor_fit),
        (6, "modality_inventory.json", lambda path, output: workload_structure(path, output, args.replay)),
    )
    receipt = {}
    for figure, name, render in plans:
        path = args.analysis / name
        receipt[str(figure)] = (render(path, args.output) | {"source": name,
                                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                               if path.is_file() else {"figure": figure, "evidence": "NOT_MEASURED"})
    if args.inventory:
        receipt["6_transfer"] = structure_transfer(args.analysis / "modality_matched_transfer.json", args.inventory,
                                                    args.output)
    (args.output / "calibration_figure_data.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
