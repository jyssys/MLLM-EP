"""Paired quality and wave-latency analysis for a matched evaluation cohort."""

import argparse
import json
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def wave_times(result):
    log = (result / "benchmark.log").read_text()
    return [float(value) for value in re.findall(r"\[iter\s+\d+\]nfe=\s*\d+.*?sample_time=([\d.]+)", log)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    args = parser.parse_args()
    baseline = ROOT / "results" / args.baseline
    candidate = ROOT / "results" / args.candidate
    base_details = json.loads((baseline / "quality.json").read_text())["details"]
    cand_details = json.loads((candidate / "quality.json").read_text())["details"]
    if len(base_details) != len(cand_details):
        raise RuntimeError("evaluation cohorts differ")
    if any(
        (base["id"], base.get("gold"), base.get("entry_point"))
        != (cand["id"], cand.get("gold"), cand.get("entry_point"))
        for base, cand in zip(base_details, cand_details)
    ):
        raise RuntimeError("paired sample IDs or task references differ")
    n = len(base_details)
    deltas = [int(bool(cand_details[i]["pass"])) - int(bool(base_details[i]["pass"])) for i in range(n)]
    wins = deltas.count(1)
    losses = deltas.count(-1)
    discordant = wins + losses
    mcnemar_p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
    rng = random.Random(20260915)
    boot_quality = [100 * sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(args.repetitions)]
    base_waves = wave_times(baseline)
    cand_waves = wave_times(candidate)
    latency = None
    if len(base_waves) == len(cand_waves) and base_waves:
        wave_n = len(base_waves)
        boot_latency = []
        for _ in range(args.repetitions):
            indices = [rng.randrange(wave_n) for _ in range(wave_n)]
            total_base = sum(base_waves[index] for index in indices)
            total_candidate = sum(cand_waves[index] for index in indices)
            boot_latency.append(100 * (total_base - total_candidate) / total_base)
        latency = {
            "paired_waves": wave_n,
            "baseline_total_s": sum(base_waves),
            "candidate_total_s": sum(cand_waves),
            "gain_pct": 100 * (sum(base_waves) - sum(cand_waves)) / sum(base_waves),
            "conditional_wave_bootstrap_95_pct": [percentile(boot_latency, 0.025), percentile(boot_latency, 0.975)],
            "caveat": "Waves within one process are not independent model restarts; use restart medians for promotion.",
        }
    report = {
        "baseline": args.baseline, "candidate": args.candidate, "n": n,
        "one_sample_pp": 100 / n,
        "quality_delta_pp": 100 * sum(deltas) / n,
        "candidate_wins": wins, "candidate_losses": losses,
        "paired_mcnemar_exact_p": mcnemar_p,
        "paired_quality_bootstrap_95_pp": [percentile(boot_quality, 0.025), percentile(boot_quality, 0.975)],
        "noninferiority_certified_at_pp": {
            str(budget): percentile(boot_quality, 0.025) >= -budget for budget in (0.3, 0.5, 1.0, 2.0)
        },
        "latency": latency,
    }
    output = candidate / f"paired_vs_{args.baseline}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
