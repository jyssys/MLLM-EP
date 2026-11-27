"""Reduce physical A/B trials and render traceable operator figures."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def pctile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    index = p * (len(ordered) - 1)
    old, new = ordered[int(index)], ordered[min(int(index) + 1, len(ordered) - 1)]
    return old + (new - old) * (index - int(index))


def report(data):
    shape_cost, pairs, trips, placements, unavailable, resources = [], [], [], [], [], []
    setup = next(row for row in data if row["kind"] == "setup")
    for row in data:
        kind = row["kind"]
        if kind == "pair":
            iso_d, iso_p = (median(t["gpu_ms"] for t in row[f"isolated_{x}"]) for x in ("D", "P"))
            live = row["concurrent"]
            d_p95 = pctile([t["d_ms"] for t in live], 0.95)
            d_iso_p95 = pctile([t["gpu_ms"] for t in row["isolated_D"]], 0.95)
            pairs.append({
                "shape": row["shape"], "pair": row["pair"], "mode": row["mode"],
                "mean_context": sum(row["actual_kv_rows"]) / len(row["actual_kv_rows"]),
                "isolated_D_ms": iso_d, "isolated_P_ms": iso_p, "D_p95_ms": d_p95,
                "decoded_p95_change_pct": 100 * (d_p95 / d_iso_p95 - 1),
                "D_median_change_pct": 100 * (median(t["d_ms"] for t in live) / iso_d - 1),
                "P_median_change_pct": 100 * (median(t["p_ms"] for t in live) / iso_p - 1),
                "paired_ms": median(t["makespan_ms"] for t in live),
                "overlap_ms": median(t["overlap_ms"] for t in live),
                "representative_interval": live[len(live) // 2]["intervals_ms"],
            })
        elif kind == "round_trip":
            arms = {}
            for buffer in row["data"]:
                trials = buffer["trials"]
                values = [x["d_ms"] for x in trials if x["d_ms"] is not None]
                arms[buffer["mode"]] = {
                    "P_critical_wall_p50_ms": median(t["p_critical_wall_ms"] for t in trials),
                    "P_critical_wall_p95_ms": pctile([t["p_critical_wall_ms"] for t in trials], 0.95),
                    "D_device_p95_ms": pctile(values, 0.95), "correctness": buffer["correctness"],
                    "remote_compute_overlap_ms": pctile([x["remote_compute_overlap_ms"] for x in trials
                                                         if x.get("remote_compute_overlap_ms") is not None], 0.5)}
            ref, alt = arms["local_with_D"], arms["split_with_D"]
            bound = 100 * (1 - alt["P_critical_wall_p50_ms"] / ref["P_critical_wall_p50_ms"])
            harm = 100 * (alt["D_device_p95_ms"] / ref["D_device_p95_ms"] - 1)
            valid = all(entry["correctness"]["relative_l2"] <= 0.02 for entry in arms.values())
            trips.append({
                "shape": row["shape"], "ratio": row["fraction"], "remote_M": row["remote_M"],
                "allocation": row.get("allocation", "priority_streams"),
                "coincident_release": row.get("coincident_release", False),
                "includes_front": row["included_prefill_front"], "bytes": row["bytes_round_trip"],
                "P_gain_pct": bound, "D_harm_pct": harm, "correctness_pass": valid,
                "screen_pass": valid and bound >= 10.0 and harm <= 5.0, "arms": arms,
            })
        elif kind == "destination":
            keep = ("project_P", "direct_P_to_D", "hidden_to_D_plus_project", "pinned_host_to_D",
                    "pinned_host_KV_to_D", "hidden_D_to_P")
            result = {key: median(t["gpu_ms"] for t in row[key]) for key in keep}
            placements.append({"rows": row["rows"], "hidden_B": row["hidden_B"], "kv_B": row["kv_B"],
                               "relative_ratio": result["hidden_to_D_plus_project"] / result["direct_P_to_D"],
                               "milliseconds": result, "correctness": row["correctness"]})
        elif kind == "layer_cost":
            shape_cost.append({"rows": row["rows"],
                               "milliseconds": {key: median(t["gpu_ms"] for t in row[key])
                                                for key in ("front", "mlp", "full")}})
        elif kind == "unavailable":
            unavailable.append(row)
        elif kind == "isolated_resources":
            resources.append(row)
    return {"evidence": "MEASURED_PHYSICAL", "setup": setup, "pairs": pairs, "layer_cost": shape_cost,
            "round_trips": trips, "destination": placements, "unavailable": unavailable,
            "isolated_resources": resources,
            "full_front_screens_passed": sum(r["screen_pass"] and r["includes_front"] for r in trips)}


def figures(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    cases = [p for p in summary["pairs"] if p["mode"] == "streams"]
    decode = sorted(set(p["pair"][0] for p in cases))
    prefill = sorted(set(p["pair"][1] for p in cases))
    if cases:
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        points = np.full((len(decode), len(prefill)), np.nan)
        for i, name in enumerate(decode):
            for j, other in enumerate(prefill):
                rows = [p["D_median_change_pct"] for p in cases if p["pair"] == [name, other]]
                if rows:
                    points[i, j] = median(rows)
                    ax.text(j, i, f"{points[i, j]:.1f}%", ha="center", va="center", color="black")
        view = ax.imshow(points, interpolation="nearest", cmap="coolwarm")
        ax.set(xticks=range(len(prefill)), xticklabels=prefill, yticks=range(len(decode)), yticklabels=decode,
               xlabel="Prefill operator", ylabel="Decode or transfer operator", title="Concurrent decode slowdown")
        fig.colorbar(view, ax=ax, label="Median of shape-wise change (%)")
        save(fig, "operator_pair_slowdown")

    duo = [p for p in summary["pairs"] if p["pair"] == ["attention", "MLP"]]
    if duo:
        target = min({tuple(p["shape"]) for p in duo}, key=lambda v: abs(v[0]-4096) + abs(v[1]-4) * 100 + abs(v[2]-16384))
        chart = [p for p in duo if p["shape"] == list(target)]
        fig, ax = plt.subplots(figsize=(8, 3.3), layout="constrained")
        labels = []
        for i, row in enumerate(chart):
            labels.append(row["mode"])
            for j, (begin, end) in enumerate(row["representative_interval"]):
                ax.barh(i + 0.22*j, end-begin, left=begin, height=.18,
                        label=("Decode attention", "Prefill MLP")[j] if i == 0 else None)
        ax.set(yticks=range(len(labels)), yticklabels=labels,
               xlabel="CUDA event milliseconds from release", title=f"Operator span at {target}")
        ax.legend()
        save(fig, "temporal_spatial_operators")

    rows = summary["round_trips"]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for flag, mark, key in [(True, "o", "Front + MLP"), (False, "^", "MLP only")]:
            group = [r for r in rows if r["includes_front"] == flag]
            ax.scatter([r["D_harm_pct"] for r in group], [r["P_gain_pct"] for r in group], label=key, marker=mark)
        ax.axhline(10, color="#444", linestyle="--")
        ax.axvline(5, color="#444", linestyle="--")
        ax.set(xlabel="Decode p95 slowdown (%)", ylabel="Prefill critical-path saving (%)",
               title="Physical activation round-trip trade-off")
        ax.legend()
        save(fig, "operator_offload_break_even")

    destination = summary["destination"]
    if destination:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
        x = [r["rows"] for r in destination]
        for key, label in (("kv_B", "KV"), ("hidden_B", "Normalized hidden")):
            axes[0].plot(x, [r[key] / 1024**2 for r in destination], "o-", label=label)
        for key, label in (("direct_P_to_D", "KV peer copy"), ("hidden_to_D_plus_project", "Hidden peer + D projection"),
                           ("pinned_host_KV_to_D", "KV host load")):
            axes[1].plot(x, [r["milliseconds"][key] for r in destination], "o-", label=label)
        for axis in axes:
            axis.set(xlabel="Rows")
            axis.legend(fontsize=8)
        axes[0].set(ylabel="Payload MiB")
        axes[1].set(ylabel="CUDA event ms")
        save(fig, "destination_payload_cost")


def release_figure(targeted, tiled, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for label, rows, marker, color in (
        ("D released during P front", [row for row in targeted["round_trips"] if not row["coincident_release"]],
         "o", "#15705a"),
        ("Coincident release", [row for row in targeted["round_trips"] if row["coincident_release"]],
         "x", "#9e392f"),
        ("Coincident, smaller remote tiles", tiled["round_trips"], "^", "#bf7c15"),
    ):
        ax.scatter([row["P_gain_pct"] for row in rows], [row["D_harm_pct"] for row in rows],
                   label=label, marker=marker, color=color, alpha=.82)
    ax.axhline(5, linestyle="--", color=".45", linewidth=1)
    ax.axvline(10, linestyle="--", color=".45", linewidth=1)
    ax.set_yscale("symlog", linthresh=5)
    ax.set(xlabel="P block critical-path improvement (%)", ylabel="D actor p95 slowdown (%)",
           title="Decode protection depends on when remote work is released")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--tiled", type=Path, help="Optional tiled raw data to compare with --in")
    args = parser.parse_args()
    summary = report(load(args.raw))
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    if args.tiled:
        if args.figures is None:
            parser.error("--tiled requires --figures")
        args.figures.mkdir(parents=True, exist_ok=True)
        release_figure(summary, report(load(args.tiled)), args.figures / "operator_release_tradeoff.png")
    elif args.figures:
        figures(summary, args.figures)


if __name__ == "__main__":
    main()
