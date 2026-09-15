"""Build evidence-labeled tables and measured-only Pareto plots."""

import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def write_csv(name, rows, fields):
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory():
    complete = []
    attempts = []
    for result in sorted(RESULTS.iterdir()):
        if not result.is_dir() or not (result / "command.txt").exists():
            continue
        log = (result / "benchmark.log").read_text() if (result / "benchmark.log").exists() else ""
        quality_file = result / "quality.json"
        points = re.findall(r"\[iter\s+\d+\]nfe=\s*(\d+), token number=\s*(\d+), sample_time=([\d.]+)", log)
        command = (result / "command.txt").read_text()
        mini = re.search(r"--mini_batch_size (\d+)", command)
        generation = re.search(r"--gen_len (\d+)", command)
        threshold = re.search(r"--threshold ([\d.]+)", command)
        observer = (result / "decisions.jsonl").exists() or (result / "ep_trace").exists()
        policy = (result / "policy.txt").read_text().strip() if (result / "policy.txt").exists() else "static"
        error = ""
        if "OutOfMemoryError" in log:
            error = "OOM"
        elif "EADDRINUSE" in log:
            error = "TCP_port_collision"
        elif not quality_file.exists():
            error = "incomplete_or_unscored"
        attempts.append({
            "label": result.name, "status": "complete" if quality_file.exists() and points else "failed",
            "error": error, "observer_heavy": observer,
            "started_at": (result / "physical_gpu_inventory.csv").stat().st_mtime if (result / "physical_gpu_inventory.csv").exists() else "",
            "result": str(result),
        })
        if not quality_file.exists() or not points:
            continue
        quality = json.loads(quality_file.read_text())
        complete.append({
            "label": result.name, "task": quality["task"], "n": quality["total"],
            "mini": int(mini.group(1)) if mini else "", "generation": int(generation.group(1)) if generation else "",
            "threshold": float(threshold.group(1)) if threshold else "",
            "policy": policy, "policy_class": "dynamic" if policy != "static" else "static",
            "observer_heavy": observer, "correct": quality["correct"],
            "accuracy_pct": quality["accuracy"] * 100,
            "nfe": sum(int(point[0]) for point in points),
            "generated_tokens": sum(int(point[1]) for point in points),
            "bct_s": sum(float(point[2]) for point in points),
            "waves": len(points), "evidence": "actual_full_trajectory",
        })
    return complete, attempts


def main():
    complete, attempts = inventory()
    fields = [
        "label", "task", "n", "mini", "generation", "threshold", "policy", "policy_class",
        "observer_heavy", "correct", "accuracy_pct", "nfe", "generated_tokens", "bct_s", "waves", "evidence",
    ]
    clean = [row for row in complete if not row["observer_heavy"]]
    baseline = [row for row in clean if row["policy"] == "static" and row["threshold"] == 0.9]
    phase = [row for row in clean if row["policy"] != "static"]
    write_csv("BASELINE_RESULTS.csv", baseline, fields)
    write_csv("PHASE_AGGRESSION_SWEEP.csv", phase, fields)
    write_csv("E2E_RESULTS.csv", clean, fields)
    write_csv("POLICY_SWEEP.csv", phase, fields)
    write_csv("ATTEMPT_LOG.csv", attempts, ["label", "status", "error", "observer_heavy", "started_at", "result"])

    grouped = {}
    for row in baseline:
        grouped.setdefault((row["task"], row["n"], row["mini"], row["generation"]), []).append(row)
    anchors = {}
    for key, rows in grouped.items():
        if len({row["correct"] for row in rows}) != 1:
            raise RuntimeError(f"baseline quality varies across restarts for {key}")
        anchors[key] = (rows[0]["accuracy_pct"], statistics.median(row["bct_s"] for row in rows),
                        statistics.median(row["nfe"] for row in rows))
    pareto = []
    for row in clean:
        key = (row["task"], row["n"], row["mini"], row["generation"])
        if key not in anchors:
            continue
        base_quality, base_bct, base_nfe = anchors[key]
        pareto.append({
            **row, "baseline_quality_pct": base_quality, "quality_delta_pp": row["accuracy_pct"] - base_quality,
            "baseline_median_bct_s": base_bct, "latency_gain_pct": 100 * (base_bct - row["bct_s"]) / base_bct,
            "nfe_reduction_pct": 100 * (base_nfe - row["nfe"]) / base_nfe,
            "statistical_status": "descriptive_only_use_paired_stats_json",
        })
    pareto_fields = fields + [
        "baseline_quality_pct", "quality_delta_pp", "baseline_median_bct_s",
        "latency_gain_pct", "nfe_reduction_pct", "statistical_status",
    ]
    write_csv("QUALITY_LATENCY_PARETO.csv", pareto, pareto_fields)

    # Empty fields are intentional: EP-conditioned policy / risk calibration did
    # not pass the causal-oracle gate, and inventing values would be misleading.
    write_csv("EP_SPECIFICITY_ABLATIONS.csv", [], [
        "task", "n", "semantic_only_gain_pct", "confidence_live_m_gain_pct",
        "moe_cost_gain_pct", "full_ep_cost_gain_pct", "ep_incremental_gain_pct", "evidence",
    ])
    write_csv("RISK_CALIBRATION.csv", [], [
        "task", "n", "phase", "action", "predicted_risk", "observed_final_quality_risk", "evidence",
    ])

    figure_root = ROOT / "figures"
    figure_root.mkdir(exist_ok=True)
    for task, n, mini in sorted({(row["task"], row["n"], row["mini"]) for row in pareto}):
        rows = [row for row in pareto if (row["task"], row["n"], row["mini"]) == (task, n, mini)]
        if len(rows) < 2:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for policy_class, marker, color in (("static", "o", "#2463a6"), ("dynamic", "s", "#df6b2d")):
            group = [row for row in rows if row["policy_class"] == policy_class]
            for row in group:
                axes[0].scatter(row["quality_delta_pp"], row["bct_s"], color=color, marker=marker)
                axes[1].scatter(row["quality_delta_pp"], row["latency_gain_pct"], color=color, marker=marker)
                axes[2].scatter(row["quality_delta_pp"], row["nfe"], color=color, marker=marker)
        axes[0].set(xlabel="Quality change (pp)", ylabel="Pool BCT (s)")
        axes[1].set(xlabel="Quality change (pp)", ylabel="Gain vs baseline median (%)")
        axes[2].set(xlabel="Quality change (pp)", ylabel="NFE (sum of waves)")
        fig.suptitle(f"{task} n={n}, mini={mini}: actual trajectories only")
        fig.tight_layout()
        fig.savefig(figure_root / f"quality_latency_nfe_{task}{n}_mini{mini}.png", dpi=160)
        plt.close(fig)
    print(f"complete={len(clean)} attempts={len(attempts)} pareto={len(pareto)}")


if __name__ == "__main__":
    main()
