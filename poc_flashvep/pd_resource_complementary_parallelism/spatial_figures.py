"""Reduce physical Green Context measurements and render their resource trade-off."""
import argparse
import csv
from pathlib import Path

from .hybrid_figures import plt, records


def table(rows):
    for row in rows:
        if row["kind"] != "spatial_pair":
            continue
        p_rows, d_batch, context = row["shape"]
        yield {
            "sm": "/".join(map(str, row["sm"])) if row["sm"] else "all SMs",
            "prefill_rows": p_rows,
            "decode_batch": d_batch,
            "decode_context": context,
            "decode_op": row["pair"][0],
            "submission": row["submission"],
            "d_iso_ms": row["isolated_d_gpu_p50_ms"],
            "p_iso_ms": row["isolated_p_gpu_p50_ms"],
            "d_pair_ms": row["pair_d_gpu_p50_ms"],
            "p_pair_ms": row["pair_p_gpu_p50_ms"],
            "d_pair_to_all_sm_pct": row["d_slowdown_vs_unpartitioned_isolated_pct"],
            "p_pair_to_all_sm_pct": row["p_slowdown_vs_unpartitioned_isolated_pct"],
            "d_pair_to_same_sm_pct": row["d_slowdown_pct"],
            "p_pair_to_same_sm_pct": row["p_slowdown_pct"],
            "bitwise_checks_passed": all(
                check["exact"] for checks in row["correctness"].values() for check in checks),
        }


def render(rows, out):
    space = ("112/16", "96/32", "64/64", "32/96")
    sizes = [row for row in rows if row["sm"] in space and row["submission"] == "cuda_gate"]
    geometries = sorted({(row["prefill_rows"], row["decode_batch"], row["decode_op"]) for row in sizes})
    if not geometries:
        return
    lookup = {(row["sm"], row["prefill_rows"], row["decode_batch"], row["decode_op"]): row for row in sizes}
    if any((sms, *case) not in lookup for sms in space for case in geometries):
        raise SystemExit("The spatial sweep lacks a measured cell")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10.0, 5.8), layout="constrained")
    for ax, field, title in zip(axes, ("d_pair_to_all_sm_pct", "p_pair_to_all_sm_pct"),
                                ("Decode operator", "Prefill MLP"), strict=True):
        cells = [[lookup[sms, *case][field] for case in geometries] for sms in space]
        image = ax.imshow(cells, cmap="YlOrRd", aspect="auto")
        for y, line in enumerate(cells):
            for x, value in enumerate(line):
                ax.text(x, y, f"{value:+.0f}%", ha="center", va="center",
                        color="white" if image.norm(value) > .64 else "#111827", fontsize=9)
        ax.set_yticks(range(len(space)), space)
        ax.set(ylabel="D/P SM count", title=title)
    axes[-1].set_xticks(range(len(geometries)),
                        [f"P{p // 1024}K · B{b}\n{op}" for p, b, op in geometries])
    fig.suptitle("Qwen3-8B Green Context pair time vs. isolated full-H200 time")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    rows = list(table(records(args.raw)))
    if not rows:
        raise SystemExit("No measured spatial cells")
    with args.table.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    render(rows, args.figure)


if __name__ == "__main__":
    main()
