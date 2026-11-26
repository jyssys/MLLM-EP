"""Regenerate the measured CUDA IPC cut and same-cut ablation figures."""

import argparse
import csv
from pathlib import Path
from statistics import median


MIB = 1024 ** 2


def values(rows, key, scale=1):
    measured = [float(row[key]) / scale for row in rows if row[key] != ""]
    return median(measured) if measured else float("nan")


def plot(source, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with source.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError("The ablation CSV contains no measured pairs")
    edge = [row for row in rows if int(row["input_tokens"]) == 7823
            and 7802 <= int(row["prefix_tokens"]) <= 7807]
    cuts = sorted({int(row["prefix_tokens"]) for row in edge})
    if not cuts:
        raise ValueError("The ablation CSV contains no measured 7823-token cut")
    target.mkdir(parents=True, exist_ok=True)
    colors = {"whole": "#355C7D", "slice": "#C7553D"}
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(9, 6), layout="constrained")
    edge_fields = (
        ("decode_logical_bytes", "D clone payload (MiB)", MIB),
        ("D_stage_E_TX_bytes", "E NVLink TX during D stage (MiB)", MIB),
        ("decode_clone_operations", "D clone operations", 1),
        ("D_first_token_http_ms", "D submit to first token (ms)", 1),
    )
    for axis, (key, label, scale) in zip(axs.flat, edge_fields, strict=True):
        for arm, color in colors.items():
            axis.plot(cuts, [values([row for row in edge if int(row["prefix_tokens"]) == cut],
                                    f"{arm}_{key}", scale) for cut in cuts],
                      "o", color=color, label=f"{arm.capitalize()} clone")
        axis.axvline(7805, color="#666666", linestyle=":", label="Visual span end")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.2)
        axis.set_xticks(cuts)
    axs[0, 0].legend(frameon=False, fontsize=8)
    for axis in axs[0]:
        suffix = axis.secondary_xaxis("top", functions=(lambda cut: 7823 - cut, lambda count: 7823 - count))
        suffix.set_xticks([7823 - cut for cut in cuts])
        suffix.set_xlabel("D suffix length (tokens)")
    for axis in axs[1]:
        axis.set_xlabel("P prefix length (tokens)")
    fig.suptitle("Same 7,823-token prompt at six measured cuts")
    fig.savefig(target / "same_request_cut_sweep.png", dpi=180)
    plt.close(fig)

    cross = [row for row in rows if float(row["saved_decode_logical_bytes"]) > 0]
    fields = (
        ("decode_logical_bytes", "D clone payload (MiB)", MIB),
        ("D_stage_E_TX_bytes", "E NVLink TX during D stage (MiB)", MIB),
        ("decode_clone_sync_host_ms", "D clone host enclosure (ms)", 1),
        ("D_first_token_http_ms", "D submit to first token (ms)", 1),
    )
    fig, axs = plt.subplots(2, 2, figsize=(8.5, 7.4), layout="constrained")
    other = [row for row in rows if float(row["saved_decode_logical_bytes"]) <= 0]
    for axis, (key, label, scale) in zip(axs.flat, fields, strict=True):
        for cohort, name, color in ((cross, "Crossing", "#C7553D"),
                                    (other, "Other cuts", "#355C7D")):
            whole = [float(row[f"whole_{key}"]) / scale for row in cohort
                     if row[f"slice_{key}"] != "" and row[f"whole_{key}"] != ""]
            sliced = [float(row[f"slice_{key}"]) / scale for row in cohort
                      if row[f"slice_{key}"] != "" and row[f"whole_{key}"] != ""]
            axis.scatter(whole, sliced, label=name, alpha=.65, s=17, color=color)
        upper = max(axis.get_xlim()[1], axis.get_ylim()[1])
        axis.plot([0, upper], [0, upper], color="#666666", linestyle=":", linewidth=1)
        axis.set(xlim=(0, upper), ylim=(0, upper))
        axis.set_xlabel(f"Whole clone · {label}")
        axis.set_ylabel(f"Slice clone · {label}")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Whole vs slice at the same imported-KV cut")
    fig.savefig(target / "same_cut_transfer_ablation.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    plot(args.pairs, args.output_dir)


if __name__ == "__main__":
    main()
