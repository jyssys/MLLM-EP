"""Recreate architecture and H200 campaign figures from published JSON receipts."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

GIB = 2**30
COLORS = ("#345275", "#16847b", "#798697", "#ba7145")
FAMILIES = {
    "cog": "Hard modality routing", "ernie": "Hard modality routing",
    "idefics": "Cross-attention", "composer": "Conditional adapter",
    "ocr2": "Staged visual encoder", "dsvl1": "Parallel visual towers",
    "vl2": "Hidden-state MoE", "q3moe": "Hidden-state MoE", "ar": "Hidden-state MoE",
    "uni": "Adaptive MoE", "bagel": "Task-dependent weights", "janus": "Task-dependent weights",
    "omni": "Task-dependent weights", "molmo": "Shared decoder",
}
LABELS = {"dp2": "Co-located DP2", "resident_text_fast": "PD resident", "visual_pruned": "PD visual-pruned",
          "generic_pruned": "PD reachability-pruned", "generic_reachability_pruned": "PD reachability-pruned"}


def read(path):
    return json.loads(path.read_text())


def save(fig, out, name, subtitle):
    fig.text(.08, .045, subtitle, fontsize=8, color="#465567")
    fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def landscape(results, out):
    candidates = read(results / "candidate_source_metadata.json")["candidates"]
    groups = {}
    for key in candidates:
        if key in FAMILIES:
            groups.setdefault(FAMILIES[key], []).append(key)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for i, (label, keys) in enumerate(groups.items()):
        ax.barh(i, len(keys), color=COLORS[i % len(COLORS)])
        ax.text(len(keys) + .1, i, ", ".join(keys), va="center", fontsize=8)
    ax.set_yticks(range(len(groups)), groups)
    ax.invert_yaxis()
    ax.set_xlim(0, max(map(len, groups.values())) + 2.6)
    ax.set_xlabel("Pinned checkpoints")
    ax.set_title("Architecture landscape")
    fig.set_constrained_layout_pads(h_pad=.3)
    save(fig, out, "fig01_architecture_landscape", "Source-verified boundaries. A checkpoint can contain multiple structural features.")


def cog(results, out):
    data = read(results / "cogvlm2" / "smoke.json")
    modes = ("resident_text_fast", "visual_pruned", "generic_reachability_pruned")
    rows = [data["layouts"][mode] for mode in modes]
    title = [LABELS[mode] for mode in modes]
    weights = [row["parameter_plus_buffer_bytes"]/GIB for row in rows]
    kv = [row["KV_capacity_trial"]["allocation_bytes"]/GIB for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.subplots_adjust(bottom=.24)
    ax.bar(title, weights, color=COLORS[0], label="D parameter payload")
    ax.bar(title, kv, bottom=weights, color=COLORS[1], label="Real KV allocation")
    for i, row in enumerate(rows):
        ax.text(i, weights[i] + kv[i]/2, f'{row["KV_capacity_trial"]["allocated_slots"]} slots',
                ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylabel("GiB")
    ax.set_ylim(0, 78)
    ax.set_title("CogVLM2 parameter residency and physical KV storage")
    ax.legend(loc="upper left", ncols=2, frameon=False)
    save(fig, out, "fig03_cog_parameter_residency", "Isolated H200 allocation. The common 64-GiB weights+KV trial is not an active batch limit.")


def phase_matrix(results, out):
    path = results / "module_phase_matrix.json"
    if not path.is_file():
        return
    data = read(path)
    states = {None: -1, False: 0, True: 1}
    matrix = [[states[value] for value in row["called"]] for row in data["rows"]]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    fig.subplots_adjust(left=.34, bottom=.26)
    ax.imshow(matrix, cmap=ListedColormap(["#eee9e2", "#e9eff4", "#83c1b2"]), aspect="auto", vmin=-1, vmax=1)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, {-1: "Unmeasured", 0: "Inactive", 1: "Called"}[value],
                    va="center", ha="center", fontsize=9)
    ax.set_yticks(range(len(matrix)), [row["module"] for row in data["rows"]])
    ax.set_xticks(range(len(data["phases"])), data["phases"], rotation=24, ha="right")
    ax.set_title("Observed nonempty module use by phase")
    save(fig, out, "fig02_module_phase", "Scope and conditional in-source branch receipts are recorded alongside the matrix.")


def correctness_summary(results, out):
    data = read(results / "cogvlm2" / "smoke.json")
    modes = ("resident_text_fast", "visual_pruned", "generic_reachability_pruned")
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.subplots_adjust(bottom=.27, wspace=.38)
    titles = ("Resident", "Visual-pruned", "Reachability")
    slots = [data["layouts"][mode]["KV_capacity_trial"]["allocated_slots"] for mode in modes]
    axs[0].bar(titles, slots, color=COLORS[:3])
    axs[0].set_ylabel("Physically allocated KV slots")
    axs[0].set_title("Capacity headroom")
    resident = data["layouts"][modes[0]]["parameter_plus_buffer_bytes"]
    minus = [(resident - data["layouts"][mode]["parameter_plus_buffer_bytes"])/GIB for mode in modes]
    axs[1].bar(titles, minus, color=COLORS[:3])
    axs[1].set_title("Same optimization from phase reachability")
    axs[1].set_ylabel("D payload removed vs resident (GiB)")
    for ax in axs:
        ax.tick_params(axis="x", labelrotation=14)
    save(fig, out, "fig06_08_cog_headroom_ablation",
         "Both pruned loaders load the same parameter manifest and retain all source KV. The phase rule supplies the guarantee.")


def online(results, out):
    summary_path = results / "cog_online" / "comparison.json"
    if not summary_path.is_file():
        return
    result = read(summary_path)
    table = result["summaries"]
    names = [LABELS[row["mode"]] for row in table]
    agreement = {pair["candidate"]: 100 * pair["natural_exact"] / pair["count"]
                 for pair in result["matched_outputs"] if pair["baseline"] == "dp2"}
    metrics = (
        ("ttft_p95_ms", "TTFT p95 (s)", .001), ("tpot_p95_ms", "TPOT p95 (ms)", 1),
        ("itl_p99_ms", "ITL p99 (ms)", 1), ("e2e_p95_ms", "E2E p95 (s)", .001),
        ("slo_goodput_requests_per_s", "Goodput (req/s)", 1), ("joint_slo", "Joint SLO (%)", 100),
        ("step_weighted_decode_batch", "Mean prepared batch", 1), ("exact", "Exact token IDs vs DP2 (%)", 1),
    )
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    fig.subplots_adjust(bottom=.18, hspace=.85, wspace=.41)
    for ax, (key, label, scale) in zip(axes.flat, metrics, strict=True):
        values = ([100. if row["mode"] == "dp2" else agreement.get(row["mode"], float("nan")) for row in table]
                  if key == "exact" else [row[key] * scale for row in table])
        bars = ax.bar(names, values, color=COLORS[:len(table)])
        ax.set_title(label)
        ax.tick_params(axis="x", labelrotation=32, labelsize=7)
        ax.bar_label(bars, fmt="%.4f" if key == "slo_goodput_requests_per_s" else "%.2f", fontsize=8)
        ax.margins(y=.17)
    fig.suptitle(f"CogVLM2 finite-cohort online pilot · {table[0]['offered_qps']:g} offered req/s")
    save(fig, out, "fig09_cog_online", "In-process token emissions, one start per arm, intact images and Natural EOS. Historical per-request SLO.")


def cross_handoff(results, out):
    path = results / "idefics" / "trials.csv"
    if not path.is_file():
        return
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    cases = {}
    for row in rows:
        cases.setdefault(row["run"], set()).add(row["request_id"])
    main_run = max(cases, key=lambda name: len(cases[name]))
    rows = [row for row in rows if row["run"] == main_run]
    longest = max(int(row["teacher_steps"]) for row in rows)
    modes = ("projected_handoff_p", "projected_visibility_padded_p")
    generic = ("generic_packed_d", "generic_visibility_padded_d")
    data = {}
    for row in rows:
        if int(row["teacher_steps"]) == longest:
            data.setdefault((row["request_id"], int(row["images"])), {}).setdefault(row["mode"], []).append(
                float(row["prefill_plus_d_wall_ms"])
            )
    fig, ax = plt.subplots(figsize=(8, 4.1))
    fig.subplots_adjust(bottom=.24)
    for index, (candidate, base) in enumerate(zip(modes, generic, strict=True)):
        observed = [(images, median(arms[candidate]) - median(arms[base]))
                    for (_, images), arms in data.items() if {candidate, base} <= arms.keys()]
        if observed:
            images, elapsed = zip(*observed, strict=True)
            ax.scatter([count + index * .15 for count in images], elapsed, alpha=.74, color=COLORS[index],
                       label=("All slots" if not index else "Visible slots with original D shape"))
    if not ax.has_data():
        plt.close(fig)
        return
    ax.axhline(0, linewidth=1, color="#445061")
    ax.set_xlabel("Images in the original prompt")
    ax.set_ylabel("P-projected minus D-projected (ms)")
    ax.set_title(f"Cross-state handoff break-even at {longest} teacher steps")
    ax.legend(frameon=False)
    save(fig, out, "fig07_idefics_handoff_break_even",
         "Observed per-request median of timed repeats, with actual handoff and prefill. Below zero favors P projection.")


def ernie_cost(results, out):
    path = results / "ernie45vl" / "native.json"
    if not path.is_file():
        return
    data = read(path)
    image_rows = [row for row in data["requests"] if row["condition"] == "image_text"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True)
    fig.subplots_adjust(bottom=.26, wspace=.38)
    for index, group in enumerate(("vision_experts", "text_experts")):
        prefill, decode = [], []
        for row in image_rows:
            events = [item for rank in row["trace"] for item in rank if item["group"] == group]
            prefill.append(sum(item["cuda_stream_ms"] for item in events if item["invocation"] == 0))
            decode.append(sum(item["cuda_stream_ms"] for item in events if item["invocation"] > 0)
                          / (row["output_tokens"] - 1))
        label = "Visual FFN bank" if not index else "Language FFN bank"
        offsets = [i + (index-.5)*.36 for i in range(len(image_rows))]
        axes[0].bar(offsets, prefill, width=.36, label=label, color=COLORS[index])
        axes[1].bar(offsets, decode, width=.36, label=label, color=COLORS[index])
    for ax, title in zip(axes, ("Mixed prefill", "Generated text decode"), strict=True):
        ax.set_title(title)
        ax.set_xlabel("Original prompt tokens")
        ax.set_xticks(range(len(image_rows)), [str(row["prompt_tokens"]) for row in image_rows])
        ax.set_ylabel("Bank scope time (ms / forward)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].text(.98, .95, "Visual bank: 0 calls", transform=axes[1].transAxes, ha="right", va="top", fontsize=8)
    fig.suptitle("ERNIE-4.5-VL · native vLLM modality-specific expert work")
    save(fig, out, "fig05_ernie_native_expert_cost",
         "Hooked CUDA-stream enclosures include the modules’ inner work and first-shape effects. Single H200 diagnostic.")


def decisions(results, out):
    source = results / "decision_matrix.json"
    if not source.is_file():
        return
    data = read(source)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    table = ax.table(cellText=[[row[key] for key in ("candidate", "structure", "evidence", "decision")]
                               for row in data], colLabels=("Candidate", "Structural boundary", "Highest evidence", "Decision"),
                     cellLoc="left", colLoc="left", colWidths=(.21, .29, .22, .28), loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.85)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#e4eef5" if not row else "#f0f3f5" if row % 2 else "white")
    ax.set_title("Architecture-specific research decisions")
    save(fig, out, "fig10_decisions", "A system mechanism, production correctness, and an online gain have separate evidence gates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for plot in (landscape, phase_matrix, cog, ernie_cost, correctness_summary, cross_handoff, online, decisions):
        plot(args.results, args.out)


if __name__ == "__main__":
    main()
