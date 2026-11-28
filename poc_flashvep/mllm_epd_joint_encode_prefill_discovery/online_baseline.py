"""Recompute the Native pilot and render separately scoped online comparators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..mllm_epd_modality_aware_idle_execution_discovery.online_evidence import (
    compact_client,
    json_lines,
    measure,
    original_schedule_matches,
    payload_receipt,
    proxy_routes,
)

TAG = "joint_native_n24"
ARCHIVE_COMMIT = "d76e9785e0928811051c9fd8dd32bf5c7497df5b"
ARCHIVE = (
    Path(__file__).resolve().parent.parent
    / "mllm_epd_modality_aware_idle_execution_discovery/results/online_evidence.json"
)
PILOT_LABEL = "Native, current N24"
NATIVE_ARCHIVE_LABEL = "Native, earlier"
SERIES = (
    ("native_fixed16_persistence", "deflection-inline", "inline_native_n24", NATIVE_ARCHIVE_LABEL),
    ("fresh", "native-recovery", "fresh_native_n64", NATIVE_ARCHIVE_LABEL),
    ("fresh", "native-recovery", "fresh_protected_generic_n64", "E→D Isolation"),
    ("fresh", "matched-controls", "fresh_cost_generic_n64", "E→D Cost"),
    ("fresh", "native-recovery", "fresh_native_n128", NATIVE_ARCHIVE_LABEL),
    ("fresh", "matched-controls", "fresh_native_restart_n128", "Native, earlier restart"),
    ("fresh", "native-persistence", "fresh_native_n256", NATIVE_ARCHIVE_LABEL),
    ("fresh", "dp3-b2048", "fresh_dp3_n128", "DP3 B2048"),
    ("historical_DP3", "dp3-retry", "colocated_long_n24", "DP3 B8192"),
    ("historical_DP3", "dp3-b2048", "colocated_long_n24", "DP3 B2048"),
    ("historical_DP3", "dp3-retry", "colocated_long_n64", "DP3 B8192"),
    ("historical_DP3", "dp3-b2048", "colocated_long_n64", "DP3 B2048"),
    ("fresh", "native-persistence", "heldout_native_n24", NATIVE_ARCHIVE_LABEL),
    ("fresh", "native-persistence", "heldout_protected_generic_n24", "E→D Isolation"),
    ("fresh", "dp3-b2048", "heldout_dp3_n24", "DP3 B2048"),
)


def json_file(path):
    return json.loads(path.read_bytes())


def reduce_pilot(epoch, frozen_arrival):
    raw = epoch / "raw" / f"{TAG}_requests.jsonl"
    completion_file = epoch / "state" / f"{TAG}_complete.json"
    arrival_file = epoch / "configs" / f"{TAG}_arrival.json"
    setup, = [row for row in json_lines(epoch / "configs/cohorts.jsonl") if row["tag"] == TAG]
    complete = json_file(completion_file)
    arrival = json_file(arrival_file)
    environment = json_file(epoch / "state/environment.json")
    model = json_file(epoch / "state/model_download_complete.json")
    raw_receipt = payload_receipt(raw)
    rows = [compact_client(row) for row in json_lines(raw)]
    if raw_receipt["sha256"] != complete["raw_SHA256"]:
        raise ValueError("Native pilot raw SHA-256 differs from its completion receipt")
    if not original_schedule_matches({row["index"]: row for row in rows}, arrival):
        raise ValueError("Native pilot planned arrivals differ from the retained client schedule")
    if frozen_arrival:
        frozen = json_file(frozen_arrival)
        if any(arrival[key] != frozen[key] for key in ("seed", "QPS", "trace")):
            raise ValueError("Native pilot did not use the frozen N24 arrival schedule")

    routes = proxy_routes(epoch)
    summary = measure(rows, setup, complete, arrival, routes)
    if not (summary["complete"] and setup["route"] == "native" and setup["observer"] == "clean"
            and summary["proxy_routes_joined"] == summary["completed"]
            and summary["D_local_completed_requests"] == 0):
        raise ValueError("Native pilot is incomplete or has unsupported routing")
    return {
        "label": PILOT_LABEL,
        "scope": "CURRENT",
        "evidence": "MEASURED_ONLINE",
        "epoch": epoch.name,
        "tag": TAG,
        "setup": setup,
        "model_revision": model["revision"],
        "GPU_UUID_configuration": environment["authorized_uuid_by_gpu"],
        "source_base_commit": environment["commit"],
        "runner_sha256": environment["runner_sha256"],
        "runtime_packages": environment["runtime"]["packages"],
        "max_seqs": environment["configuration"]["max_seqs"],
        "raw_receipt": raw_receipt,
        "completion_receipt": payload_receipt(completion_file),
        "arrival_receipt": payload_receipt(arrival_file),
        "frozen_arrival_verified": frozen_arrival is not None,
        "arrival": arrival,
        "clients": sorted(rows, key=lambda row: row["index"]),
        **summary,
    }


def load_archive(path):
    audit = json_file(path)
    rows = []
    for key, epoch, tag, label in SERIES:
        row, = [record for record in audit[key] if record["epoch"] == epoch and record["tag"] == tag]
        if row["evidence"] != "MEASURED_ONLINE" or not row["complete"]:
            raise ValueError(f"Archive row is incomplete or unavailable: {epoch}/{tag}")
        rows.append({**row, "archive_group": key, "scope": "ARCHIVED", "label": label})
    return {
        "source_commit": ARCHIVE_COMMIT,
        "source_receipt": payload_receipt(path),
        "rows": rows,
        "matched_generic_pairs": audit["pairs"],
        "source_scope_note": "Earlier campaign records. A record called FRESH in the source is archived here.",
    }


def plot_tradeoff(rows, metric, percentile, absolute_ms, ylabel, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    style = {
        PILOT_LABEL: ("*", "#17675b", 110),
        NATIVE_ARCHIVE_LABEL: ("o", "#436984", 60),
        "Native, earlier restart": ("D", "#436984", 52),
        "E→D Isolation": ("s", "#a14f00", 61),
        "E→D Cost": ("v", "#a14f00", 65),
        "DP3 B2048": ("^", "#a13967", 70),
        "DP3 B8192": ("P", "#78478a", 74),
    }
    plt.rcParams["svg.hashsalt"] = "joint-native-pilot-compare"
    plt.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6), sharey=True)
    for ax, size in zip(axes, (24, 64, 128), strict=True):
        panel = [row for row in rows if row["planned"] == size and row["setup"]["output"] == 16]
        for row in panel:
            x = row["metrics_ms"]["TTFT_ms"]["p95"] / 1000
            y = row["metrics_ms"][metric][percentile]
            marker, color, marker_size = style[row["label"]]
            ax.scatter(x, y, marker=marker, color=color, s=marker_size, clip_on=False, zorder=3)
        ax.axhline(absolute_ms, color="#64748b", linewidth=1, linestyle=":")
        ax.set_title(f"N{size}", fontweight="bold", loc="left")
        ax.set_ylim(3, 750)
        ax.set_yscale("log")
        ax.set_yticks([4, 8, 16, 32, 64, 128, 256, 512])
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        endpoint = max(row["metrics_ms"]["TTFT_ms"]["p95"] for row in panel) / 1000
        ax.set_xlim(0, endpoint * 1.14)
        ax.set_xlabel("TTFT p95 (s)")
        ax.grid(color="#cbd5e1", alpha=0.45, which="major")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel + " (ms, log scale)")
    axes[-1].annotate(f"{absolute_ms} ms absolute limit", (0.96, absolute_ms), xycoords=("axes fraction", "data"),
                      xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8, color="#475569")
    handles = [
        plt.Line2D([], [], marker=marker, color=color, markersize=7, linestyle="none", label=label)
        for label, (marker, color, _) in style.items()
    ]
    fig.suptitle("Very-long Fixed16 · TTFT and " + ylabel, fontsize=15, fontweight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.888, "1.465 offered req/s · current Native pilot and earlier campaigns",
             fontsize=10, color="#475569")
    fig.legend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.016))
    fig.subplots_adjust(bottom=0.26, top=0.80, left=0.08, right=0.97, wspace=0.17)
    fig.savefig(filename, metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=Path, help="Group-sweep epoch with the Native pilot raw client trace")
    parser.add_argument("--frozen-arrival", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    evidence = args.out / "online_comparison.json"
    if args.epoch:
        args.out.mkdir(parents=True, exist_ok=True)
        data = {
            "native_pilot": reduce_pilot(args.epoch, args.frozen_arrival),
            "archive": load_archive(args.archive),
        }
        evidence.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    else:
        data = json_file(evidence)

    rows = [data["native_pilot"], *data["archive"]["rows"]]
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    plot_tradeoff(rows, "TPOT_ms", "p95", 33, "TPOT p95", figures / "online_ttft_tpot.svg")
    plot_tradeoff(rows, "ITL_ms", "p99", 34, "ITL p99", figures / "online_ttft_itl.svg")


if __name__ == "__main__":
    main()
