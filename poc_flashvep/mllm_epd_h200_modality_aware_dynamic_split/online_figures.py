"""Render available dynamic serving panels without treating missing arms as zeros."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .historical_figures import csv_rows, persist, plt

DYNAMIC = ("late10", "generic_pure", "generic", "hard", "soft", "fake_boundary")
PRIMARY_TAG = "Natural EOS"
COLORS = ("#2e6277", "#b15349", "#549b74", "#77639c", "#b4823f")
DISPLAY_NAMES = {
    "native_strong": "NativeStrong",
    "native_common": "NativeCommon",
    "native_matched_on_demand": "Native matched (on-demand)",
    "native_matched_inline_redirect": "Native matched (inline)",
    "idle10": "Idle10",
}


def method_family(row):
    return row["method_family"].replace("-", "_")


def method_kind(row):
    if row.get("execution_mode") == "shadow":
        return None
    family = method_family(row)
    return next((name for name in DYNAMIC if family == name or family.startswith(name + "_")), None)


def group(methods):
    collect = defaultdict(list)
    for row in methods:
        collect[(method_family(row), row["decode_cap"])].append(row)
    return collect


def axes_label(arm):
    family, cap = arm
    method = family.removesuffix("_natural_hold_inline_redirect_shadow")
    if method != family:
        label = f"{method.replace('_', ' ')} (shadow)"
    else:
        label = DISPLAY_NAMES.get(family, family.replace("_", " "))
    return f"{label} {'Free' if cap == 'free' else 'B' + cap}"


def offload_ratio(rows, output):
    arms = group([row for row in rows if method_kind(row) is not None
                  and row.get("actual_offload_ratio") not in ("", None)])
    if not arms:
        return None
    # Nearest nominal labels within 1.5 percentage points; all other actual ratios stay grouped separately.
    targets = (5, 10, 20, 30)
    labels = ["0%", "≈5%", "≈10%", "≈20%", "≈30%", "Other cut"]
    fig, ax = plt.subplots(figsize=(8, 3.7), layout="constrained")
    data = {}
    for arm, records in arms.items():
        counts = Counter()
        for row in records:
            percent = 100 * float(row["actual_offload_ratio"])
            nearest = min(targets, key=lambda value: abs(value - percent))
            if percent == 0:
                label = "0%"
            elif abs(nearest - percent) <= 1.5:
                label = f"≈{nearest}%"
            else:
                label = "Other cut"
            counts[label] += 1
        data[axes_label(arm)] = dict(counts)
    width = .86 / len(data)
    for offset, (arm, counts) in enumerate(sorted(data.items())):
        positions = [i + width * (offset - (len(data) - 1) / 2) for i in range(len(labels))]
        ax.bar(positions, [100 * counts.get(label, 0) / sum(counts.values()) for label in labels],
               width=width, label=arm)
    ax.set(title="Online actual suffix selection · QPS5 Natural EOS", ylabel="% of completed requests",
           xticks=range(len(labels)), xticklabels=labels)
    ax.legend(frameon=False, fontsize=8)
    persist(fig, output, "fig07_actual_suffix_ratio")
    return {"figure": 7, "evidence": "MEASURED_ONLINE", "request_counts": data}


def direct_neighbors(cohorts, names):
    groups = defaultdict(dict)
    for row in cohorts:
        name = method_kind(row)
        if name not in names:
            continue
        signature = (row["QPS"], row["decode_cap"], row["output_mode"], row["subset"],
                     row["manifest_sha256"], row["arrival_sha256"], row["burner_condition"],
                     row["independent_start_id"], row["transport"], row["P_boundary_policy"],
                     row["serving_source_sha256"])
        groups[signature].setdefault(name, []).append(row)
    return [arms for _, arms in sorted(groups.items(), key=lambda item: repr(item[0])) if all(
        name in arms and len(arms[name]) == 1 for name in names)]


def boundary_comparison(cohorts, output):
    neighbors = direct_neighbors(cohorts, ("hard", "soft"))
    if not neighbors:
        return None
    arms = neighbors[0]
    data = [arms[name][0] for name in ("hard", "soft")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), layout="constrained")
    for ax, field, label, multiplier in zip(
        axes, ("workload_offload_ratio", "TTFT_p95_ms", "absolute_decode_slo"),
        ("Actual offloaded input (%)", "TTFT p95 (s)", "Absolute Decode SLO (%)"),
        (100, .001, 100), strict=True
    ):
        values = [row[field] for row in data]
        if all(value is not None for value in values):
            ax.bar(["Hard", "Soft"], [float(value) * multiplier for value in values], color=COLORS[:2])
        else:
            ax.text(.5, .5, "NOT_MEASURED", transform=ax.transAxes, ha="center", va="center")
        ax.set(ylabel=label)
    fig.suptitle("Hard vs Soft · same input/arrival contract · clean screening")
    persist(fig, output, "fig08_hard_vs_soft")
    return {"figure": 8, "evidence": "MEASURED_ONLINE", "cohorts": [row["cohort"] for row in data],
            "scope": "The full decision/transfer evidence remains in the request-level and calibration receipts."}


def comparison_cdf(rows, output):
    arms = group(rows)
    if not arms or not any(row["method_family"].startswith(("native", "Native")) for row in rows):
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.0), layout="constrained")
    for arm, records in sorted(arms.items()):
        values = sorted(float(row["TTFT_ms"]) / 1_000 for row in records)
        ax.step(values, [(index + 1) / len(values) for index in range(len(values))],
                where="post", label=axes_label(arm))
    ax.axvline(5, color="#42474d", linewidth=.8, linestyle=":", label="5 s TTFT SLO")
    ax.set(title="H200 QPS5 Natural EOS · observed clean cohorts", xlabel="TTFT (s)", ylabel="Cumulative fraction")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.grid(alpha=.2)
    persist(fig, output, "fig09_clean_ttft_cdf")
    return {"figure": 9, "evidence": "MEASURED_ONLINE", "arms": [axes_label(arm) for arm in arms],
            "complete_native_dynamic_comparison": any(method_kind(row) is not None for row in rows),
            "scope": "Descriptive pooled clean CDF. Exact matched, independent-start contrasts are separate."}


def diagnostic_ablation(cohorts, output):
    neighbors = direct_neighbors(cohorts, ("generic", "hard", "soft"))
    if not neighbors:
        return None
    arms = neighbors[0]
    selected = [arms[name][0] for name in ("generic", "hard", "soft")]
    fig, ax = plt.subplots(figsize=(7.6, 3.9), layout="constrained")
    ax.bar(["Generic", "Hard", "Soft"], [float(row["TTFT_p95_ms"]) / 1_000 for row in selected], color=COLORS[:3])
    ax.set(title="Common-method boundary selection ablation · observed clean screen", ylabel="TTFT p95 (s)")
    persist(fig, output, "fig10_boundary_selection_ablation")
    return {"figure": 10, "evidence": "MEASURED_ONLINE", "cohorts": [row["cohort"] for row in selected],
            "scope": "Boundary arms only; independently switched encoding and transfer-cost arms require their own runs."}


def request_timeline(rows, cohorts, output):
    matches = direct_neighbors(cohorts, ("generic", "hard"))
    if not matches:
        return None
    for matched in matches:
        arms = [matched[name][0] for name in ("generic", "hard")]
        lookup = [{row["request_id"]: row for row in rows if (row["start"], row["cohort"]) ==
                   (arm["start"], arm["cohort"]) and row["cut_ready_after_arrival_ms"] and
                   row["D_first_token_after_arrival_ms"] and row["image_spans"]} for arm in arms]
        common = sorted(lookup[0].keys() & lookup[1].keys())
        if common:
            selected = [records[common[0]] for records in lookup]
            break
    else:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.4), layout="constrained")
    stages = (("P_start_after_arrival_ms", "P submit"), ("cut_ready_after_arrival_ms", "P cut ready"),
              ("D_submit_after_arrival_ms", "D submit"), ("D_first_token_after_arrival_ms", "D first token"))
    for index, row in enumerate(selected):
        offsets = [float(row[key]) for key, _ in stages]
        axes[0].scatter(offsets, [index] * len(offsets), label=row["method"])
        for point, (_, label) in zip(offsets, stages, strict=True):
            axes[0].annotate(label, (point, index), xytext=(1, 5), textcoords="offset points", fontsize=7,
                             rotation=16)
        for left, right in json.loads(row["image_spans"]):
            axes[1].barh(index, right - left, left=left, height=.38, color=COLORS[index])
        axes[1].vlines(float(row["prefix_tokens"]), index - .3, index + .3, color="#222222",
                       linestyle="--", linewidth=1)
    axes[0].set(xlabel="Host monotonic time since scheduled arrival (ms)", ylim=(-.3, 1.4))
    axes[1].set(xlabel="Original token position · dashed line is the actual cut")
    for ax in axes:
        ax.set(yticks=[0, 1], yticklabels=[row["method"] for row in selected])
    fig.suptitle(f"Paired executed suffix · {selected[0]['request_id']}")
    persist(fig, output, "fig11_paired_request_host_timeline")
    return {"figure": 11, "evidence": "MEASURED_ONLINE", "request_id": selected[0]["request_id"],
            "rule": "Lexically first mutually executed request within the first matched Generic/Hard group.",
            "scope": "Observed host landmarks. Per-request exclusive GPU and in-batch intervals are unavailable."}


def decision_breakdown(rows, output):
    arms = group([row for row in rows if method_kind(row) is not None])
    if not arms:
        return None
    counts = {}
    for arm, records in arms.items():
        counts[axes_label(arm)] = Counter("Suffix executed" if float(row["actual_offload_ratio"] or 0) > 0
            else row["last_decision_reason"] or ("P continuation" if row["P_continued"] == "True" else "No decision")
            for row in records)
    categories = sorted({k for values in counts.values() for k in values})
    names = sorted(counts)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    base = [0] * len(names)
    for category in categories:
        values = [counts[name][category] for name in names]
        axes[0].barh(names, values, left=base, label=category.replace("_", " "))
        base = [a + b for a, b in zip(base, values)]
    axes[0].set(xlabel="Completed requests", title="Last decision or completed suffix")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    failure = {axes_label(arm): sum(row["decode_pass"] == "False" for row in records)
               for arm, records in arms.items()}
    axes[1].barh(names, [failure[name] for name in names], color=COLORS[1])
    axes[1].set(xlabel="Completed requests", title="Absolute Decode SLO failures")
    fig.suptitle("Online decisions and failure counts · QPS5 Natural EOS")
    persist(fig, output, "fig12_decision_and_slo_breakdown")
    return {"figure": 12, "evidence": "MEASURED_ONLINE", "decisions": counts, "decode_failures": failure}


def shadow_decisions(cohorts, output):
    arms = [row for row in cohorts if row["execution_mode"] == "shadow" and row["proxy_coverage"] == row["completed"]]
    if not arms:
        return {"figure": 12, "evidence": "NOT_RUN", "mode": "shadow"}
    labels = [axes_label((method_family(row), row["decode_cap"])) +
              (" · Natural EOS" if row["output_mode"] == PRIMARY_TAG else " · Replay") for row in arms]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, max(3.65, len(arms) * .48 + 1)), layout="constrained")
    no_callbacks = [row["completed"] - row["shadow_requests_with_callbacks"] for row in arms]
    rejected = [row["shadow_requests_with_callbacks"] - row["shadow_eligible_requests"] for row in arms]
    eligible = [row["shadow_eligible_requests"] for row in arms]
    axes[0].barh(labels, no_callbacks, color="#dce7eb", label="No callback")
    axes[0].barh(labels, rejected, left=no_callbacks, color=COLORS[0], label="Callback, no eligible cut")
    axes[0].barh(labels, eligible, left=[a + b for a, b in zip(no_callbacks, rejected, strict=True)],
                 color=COLORS[2], label="At least one eligible cut")
    axes[0].set(title="Request decisions", xlabel="Completed requests")
    axes[0].legend(frameon=False, fontsize=7)
    decision_counts = [json.loads(row["all_decision_reasons"]) for row in arms]
    reasons = sorted({reason for counts in decision_counts for reason in counts})
    left = [0] * len(arms)
    for index, reason in enumerate(reasons):
        counts = [row.get(reason, 0) for row in decision_counts]
        axes[1].barh(labels, counts, left=left, color=COLORS[index % len(COLORS)],
                     label=reason.replace("_", " ").capitalize())
        left = [a + b for a, b in zip(left, counts, strict=True)]
    axes[1].set(title="Logged decision outcomes", xlabel="Callbacks", yticklabels=[])
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_ylim(axes[0].get_ylim())
    fig.suptitle("Online shadow screening · zero executed D suffixes")
    persist(fig, output, "fig12_shadow_decision_eligibility")
    return {
        "figure": 12, "mode": "shadow", "evidence": "DIAGNOSTIC",
        "cohorts": [row["cohort"] for row in arms],
        "request_counts": {label: {"no_callbacks": a, "callbacks_no_eligible": b, "eligible": c}
                           for label, a, b, c in zip(labels, no_callbacks, rejected, eligible, strict=True)},
        "decision_counts": dict(zip(labels, decision_counts, strict=True)),
        "scope": "Real coordinator snapshots. Zero projected reservations were committed; "
                 "remaining Native arrivals and Decode SLO may differ under active split.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((args.online_analysis / "online_evidence.json").read_text())
    primary = [row for row in evidence["clean_cohorts"]
               if row["QPS"] == 5 and row["output_mode"] == PRIMARY_TAG and row["subset"] == "all"]
    cohorts = [row for row in primary if row["observation"] == "clean"]
    tags = {(row["start"], row["cohort"]) for row in cohorts}
    client = csv_rows(args.online_analysis / "dynamic_requests.csv")
    rows = [row for row in client if (row["start"], row["cohort"]) in tags]
    measured_methods = {row["method"] for row in cohorts}
    before_tags = {(row["start"], row["cohort"]) for row in primary if row["observation"] == "clean_before"
                   and row["method"].startswith("native") and row["method"] not in measured_methods}
    cdf_rows = rows + [row for row in client if (row["start"], row["cohort"]) in before_tags]
    plots = [offload_ratio(rows, args.output), boundary_comparison(cohorts, args.output),
             comparison_cdf(cdf_rows, args.output), diagnostic_ablation(cohorts, args.output),
             request_timeline(rows, cohorts, args.output), decision_breakdown(rows, args.output)]
    data = {str(number): plot or {"figure": number, "evidence": "NOT_MEASURED"}
            for number, plot in enumerate(plots, start=7)}
    data["12_shadow"] = shadow_decisions([row for row in evidence["clean_cohorts"]
                                          if row["QPS"] == 5 and row["observation"] == "clean"], args.output)
    (args.output / "online_figure_data.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
