"""Reduce serial Direction B physical receipts and render the measured comparisons."""

import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, quantiles


BUDGETS_MS = (50, 100, 150, 250, 500)
PREFIX_ARMS = ("vision_only", "prefill_only", "joint_prefix")
SUFFIX_ARMS = ("native_epd", "generic_token_suffix", "joint_suffix", "joint_suffix_compact")
COLORS = ("#377eb8", "#d95f02", "#008673", "#734a9e")
NAMES = {
    "vision_only": "D vision only",
    "prefill_only": "D prefill incl. E IPC",
    "joint_prefix": "D joint encode + prefill",
    "native_epd": "Native EPD",
    "generic_token_suffix": "Generic suffix, E encode",
    "joint_suffix": "Joint suffix, D encode",
    "joint_suffix_compact": "Joint suffix, compact inputs",
}


def json_lines(path):
    if not path.is_file():
        compressed = path.with_suffix(path.suffix + ".gz")
        if not compressed.is_file():
            return []
        path = compressed
    opener = gzip.open(path, "rt") if path.suffix == ".gz" else path.open()
    with opener as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quantile(values, percentile):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[percentile - 1]


def statistics(values):
    return {"n": len(values), "min": min(values), "median": median(values), "p95": quantile(values, 95),
            "max": max(values)} if values else {"n": 0}


def receipts(trial, imports):
    result = {}
    for phase, role in (("prefill", "prefill"), ("measurement", "decode")):
        stage = trial.get(phase)
        if not stage:
            continue
        received = [receipt for receipt in imports[role] if
                    stage["begin_ns"] <= receipt["begin"] * 1e9 <= receipt["end"] * 1e9 <= stage["end_ns"]]
        result[role] = {"logical_bytes": sum(row["logical_bytes"] for row in received),
                        "images": sum(row["images"] for row in received), "receipts": len(received),
                        "all_acked": all(row["clone_complete_before_ACK"] for row in received)}
    return result


def reduce_run(run):
    raw = run / "raw"
    trial_groups = defaultdict(list)
    failures = []
    imports = {role: [entry for path in raw.glob(f"ec_cuda_ipc_{role}_*.jsonl*") for entry in json_lines(path)]
               for role in ("prefill", "decode")}
    compact_renderers = {row["metadata_key"]: row for row in json_lines(raw / "joint_compact_renderer.jsonl")}
    for trial in json_lines(raw / "joint_trials.jsonl"):
        if "error" in trial:
            failures.append(trial)
        elif not trial["warmup"]:
            trial_groups[trial["fixture"], trial["arm"]].append(trial | {"ipc": receipts(trial, imports)})
    cases = defaultdict(dict)
    for (fixture, arm), rows in trial_groups.items():
        main = [row["measurement"]["host_ms"] for row in rows]
        first = rows[0]
        entry = {"test": first["test"], "prefix_images": first["prefix_images"], "cut_tokens": first["cut_tokens"],
                 "total_images": first["total_images"], "total_tokens": first["total_tokens"],
                 "host_ms": statistics(main)}
        if first["test"] == "prefix":
            entry["fits_ms"] = {budget: sum(elapsed <= budget for elapsed in main) for budget in BUDGETS_MS}
        if arm == "vision_only":
            entry["vision_cuda_inclusive_h2d_ms"] = statistics(
                [row["vision"]["h2d_inclusive_vision_cuda_ms"] for row in rows])
        if "TTFT_ms" in first:
            for key in ("TTFT_ms", "TPOT_ms", "E2E_ms"):
                entry[key] = statistics([row[key] for row in rows])
            entry["ITL_p99_ms"] = quantile([gap for row in rows for gap in row["ITL_ms"]], 99)
            entry["all_native_fixed16_agreement"] = None
            native = {row["repeat"]: row["output_token_ids"]
                      for row in trial_groups.get((fixture, "native_epd"), ())}
            if native:
                agreements = sum(row["output_token_ids"] == native.get(row["repeat"]) for row in rows)
                entry["native_fixed16_agreement_count"] = agreements
                entry["all_native_fixed16_agreement"] = agreements == len(rows)
            if arm == "joint_suffix_compact":
                render_receipts = [compact_renderers.get(sha256(row["tag"].encode()).hexdigest()) for row in rows]
                entry["compact_renderer"] = {key: statistics([receipt[key] for receipt in render_receipts if receipt])
                                             for key in ("raw_suffix_render_ms", "prefix_grid_splice_ms",
                                                         "raw_suffix_images", "grid_only_prefix_images")}
            entry["prefill_host_ms"] = statistics([row["prefill"]["host_ms"] for row in rows])
            entry["encoder_host_ms"] = statistics([row["native_encoder"]["host_ms"] for row in rows])
            entry["physical_NIXL"] = {key: statistics([row["nixl"][key] for row in rows
                                                      if row.get("nixl") and key in row["nixl"]])
                                      for key in ("bytes", "transfers", "descriptors", "transfer_ms", "post_ms")}
            for key in ("E_encoded_images", "D_local_encoded_images", "D_local_visual_tokens",
                        "padded_KV_bytes", "E_actual_prompt_tokens", "prefix_KV_tokens"):
                entry[key] = first[key]
        entry["physical_IPC"] = {
            role: {key: statistics([row["ipc"][role][key] for row in rows
                                    if role in row["ipc"]]) for key in ("logical_bytes", "images", "receipts")}
            for role in ("prefill", "decode")
        }
        cases[fixture][arm] = entry
    protection = {}
    interventions = json_lines(raw / "joint_interference.jsonl")
    for mode in ("control", "idle", "upcoming", "active"):
        rows = [row for row in interventions if row["mode"] == mode]
        if not rows:
            continue
        protection[mode] = {
            "decode_submit_first_ms": statistics([row["native_D_TTFT_ms_from_submit"] for row in rows]),
            "decode_TPOT_ms": statistics([row["native_D_TPOT_ms"] for row in rows]),
            "decode_ITL_p99_ms": quantile([gap for row in rows for gap in row["native_D_ITL_ms"]], 99),
            "decode_ITL_max_ms": max(gap for row in rows for gap in row["native_D_ITL_ms"]),
            "active_rejections": sum(row["active_rejection_ns"] is not None for row in rows),
            "planned_upcoming_while_prefix_in_flight": sum(bool(
                row["joint_prefix"] and row["joint_prefix"]["measurement"]["begin_ns"]
                <= row["native_D_dispatch"] * 1e9 <= row["joint_prefix"]["measurement"]["end_ns"]
            ) for row in rows),
        }
    return {"evidence": "MEASURED_PHYSICAL. Serial fixed requests executed on GPU.",
            "run": str(run), "cases": dict(cases), "failures": failures, "decode_interference": protection}


def draw(summary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(arms, metric, title) for arms, metric, title in (
        (PREFIX_ARMS, "host_ms", "D-local physical work budget"),
        (SUFFIX_ARMS, "TTFT_ms", "Fixed16 same-prompt causal suffix"),
    ) if any(arm in found for found in summary["cases"].values() for arm in arms)]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), layout="constrained", squeeze=False)
    for ax, (arms, metric, heading) in zip(axes[0], panels, strict=True):
        groups = [(case, found) for case, found in summary["cases"].items() if any(arm in found for arm in arms)]
        if metric == "host_ms":
            groups.sort(key=lambda group: next(iter(group[1].values()))["cut_tokens"])
        labels = []
        arms = tuple(arm for arm in arms if any(arm in found for _, found in groups))
        width = .78 / len(arms)
        high = 0
        for index, (case, found) in enumerate(groups):
            entry = next(iter(found.values()))
            suffix_case = entry["test"] != "prefix"
            images = entry["total_images"] - entry["prefix_images"] if suffix_case else entry["prefix_images"]
            tokens = entry["total_tokens"] - entry["cut_tokens"] if suffix_case else entry["cut_tokens"]
            label = case.removeprefix("suffix_") if suffix_case else ""
            labels.append(f"{label}\n{images} img · {tokens:,} rows".lstrip())
            for i, (arm, color) in enumerate(zip(arms, COLORS)):
                if arm not in found:
                    continue
                stats = found[arm][metric]
                high = max(high, stats["p95"])
                height = stats["median"]
                position = index + (i - (len(arms) - 1) / 2) * width
                ax.bar(position, height, width=width * .93, color=color,
                       label=NAMES[arm] if index == 0 else None, zorder=3)
                ax.plot((position, position), (height, stats["p95"]), color="#20242b", lw=1, zorder=4)
        for bound in (100, 250, 500) if metric == "host_ms" else ():
            ax.axhline(bound, color="#51555b", linestyle=":", alpha=.55)
            high = max(high, bound)
        ax.set_title(heading, loc="left", fontweight="bold")
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylabel("Serial physical median (ms)")
        ax.set_ylim(0, high * 1.24)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.16)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig(destination, dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    summary = reduce_run(args.run)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot and summary["cases"]:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        draw(summary, args.plot)


if __name__ == "__main__":
    main()
