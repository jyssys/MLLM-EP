#!/usr/bin/env python3
"""Analyze paired full-model runs and strict per-layer backend oracles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poc_rankfanout.rankfanout.oracle import compute_oracles  # noqa: E402


def normalize_backend(name: str) -> str:
    aliases = {
        "allgather_reducescatter": "agrs",
        "deepep_high_throughput": "deepep_ht",
    }
    return aliases.get(str(name), str(name))


def load_driver_rows(run: Path) -> tuple[dict, list[dict]]:
    proof = None
    rows = []
    for path in sorted(run.glob("driver_dp*.json")):
        payload = json.loads(path.read_text())
        proof = payload["proof"] if proof is None else proof
        rows.extend(payload["rows"])
    return proof or {}, rows


def load_route_rows(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("route_rows_dp*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def load_spans(run: Path) -> list[dict]:
    rows = []
    for path in sorted((run / "profile").glob("moe_spans_pid*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def request_table(run: Path, restart: int) -> list[dict]:
    proof, driver_rows = load_driver_rows(run)
    output = []
    for row in driver_rows:
        if row["warmup"]:
            continue
        for request in row["requests"]:
            output.append({
                "restart": restart,
                "backend": normalize_backend(proof["backend_requested"]),
                "workload_id": row["workload_id"],
                "iteration": row["iteration"],
                "request_id": request["request_id"],
                "ttft_ms": request["ttft_ms"],
                "e2e_ms": request["e2e_ms"],
                "output_token_ids": request["output_token_ids"],
                "prompt_tokens": request["prompt_tokens"],
            })
    return output


def route_index(run: Path) -> dict[tuple[str, int, int], dict]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in load_route_rows(run):
        if int(row["step_id"]) < 0:
            continue
        grouped[(row["workload_id"], int(row["step_id"]), int(row["layer_id"]))].append(row)
    output = {}
    for key, rows in grouped.items():
        hashes = sorted(str(row["route_hash"]) for row in rows)
        combined_hash = hashlib.sha256("|".join(hashes).encode()).hexdigest()
        total_tokens = sum(int(row["num_tokens"]) for row in rows)
        mean_fanout = sum(float(row["mean_fanout"]) * int(row["num_tokens"]) for row in rows) / total_tokens
        output[key] = {
            "route_hash": combined_hash,
            "num_tokens": total_tokens,
            "mean_fanout": mean_fanout,
            "frac_full_fanout": sum(float(row["frac_full_fanout"]) * int(row["num_tokens"]) for row in rows) / total_tokens,
            "rank_max_mean": max(float(row["rank_load_max_over_mean"]) for row in rows),
            "expert_max_mean": max(float(row["expert_load_max_over_mean"]) for row in rows),
            "expert_placement_hash": rows[0]["placement_hash"],
        }
    return output


def profile_invocations(run: Path, restart: int) -> tuple[list[dict], dict]:
    proof, driver_rows = load_driver_rows(run)
    backend = normalize_backend(proof["backend_requested"])
    routes = route_index(run)
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in load_spans(run):
        if row.get("warmup") or row.get("moe_total_ms") is None or "workload_id" not in row or "iteration" not in row:
            continue
        key = (str(row["workload_id"]), int(row["iteration"]), int(row["layer_id"]))
        grouped[key].append(row)
    output = []
    path_calls: set[str] = set()
    for key, spans in grouped.items():
        if key not in routes:
            continue
        route = routes[key]
        # DP workers issue scheduler-padding/dummy MoE calls (often M=1)
        # under the same driver context.  They are real runtime work but are
        # not the route-aligned prefill invocation.  Select the largest-M
        # call, which is the captured prompt route, before taking rank max.
        routed_m = max(int(span["num_tokens"]) for span in spans)
        spans = [span for span in spans if int(span["num_tokens"]) == routed_m]
        for span in spans:
            path_calls.update(span.get("backend_calls", []))
        output.append({
            "restart": restart,
            "backend": backend,
            "workload_id": key[0],
            "request_id": "wave",
            "phase": "prefill",
            "step_id": key[1],
            "layer_id": key[2],
            "num_tokens": route["num_tokens"],
            "rank_local_num_tokens": routed_m,
            "expert_placement_hash": route["expert_placement_hash"],
            "route_hash": route["route_hash"],
            "mean_fanout": route["mean_fanout"],
            "frac_full_fanout": route["frac_full_fanout"],
            "rank_max_mean": route["rank_max_mean"],
            "expert_max_mean": route["expert_max_mean"],
            "moe_total_ms": max(float(span["moe_total_ms"]) for span in spans),
            "rank_rows": len(spans),
        })
    profile_ttft = request_table(run, restart)
    proof_output = {
        "backend": backend,
        "runtime_backend_calls": sorted(path_calls),
        "moe_rows": len(output),
        "profile_request_rows": len(profile_ttft),
        "placement_files": [path.name for path in sorted((run / "profile").glob("placement_*.json"))],
    }
    return output, proof_output


def clean_pair_rows(agrs: Path, deep: Path, restart: int) -> tuple[list[dict], dict]:
    a = request_table(agrs, restart)
    d = request_table(deep, restart)
    key = lambda row: (row["workload_id"], row["iteration"], row["request_id"])
    ai, di = {key(row): row for row in a}, {key(row): row for row in d}
    if set(ai) != set(di):
        raise ValueError(f"clean request alignment mismatch restart {restart}")
    rows = []
    agree = exact = compared = 0
    for item in sorted(ai):
        left, right = ai[item], di[item]
        if left["prompt_tokens"] != right["prompt_tokens"]:
            raise ValueError(f"prompt-token mismatch {item}")
        lt, rt = left["output_token_ids"], right["output_token_ids"]
        exact += int(lt == rt)
        compared += min(len(lt), len(rt))
        agree += sum(x == y for x, y in zip(lt, rt))
        rows.append({
            "restart": restart,
            "workload_id": item[0],
            "iteration": item[1],
            "request_id": item[2],
            "prompt_tokens": left["prompt_tokens"],
            "agrs_ttft_ms": left["ttft_ms"],
            "deepep_ttft_ms": right["ttft_ms"],
            "agrs_e2e_ms": left["e2e_ms"],
            "deepep_e2e_ms": right["e2e_ms"],
            "token_exact": lt == rt,
        })
    return rows, {
        "restart": restart,
        "requests": len(rows),
        "exact_request_fraction": exact / len(rows),
        "token_agreement": agree / compared if compared else 1.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def per_workload_oracle(rows: list[dict], clean: list[dict], profile_request_rows: list[dict]) -> list[dict]:
    output = []
    for restart in sorted({int(row["restart"]) for row in rows}):
        for workload in sorted({row["workload_id"] for row in rows if int(row["restart"]) == restart}):
            selected = [row for row in rows if int(row["restart"]) == restart and row["workload_id"] == workload]
            oracle = compute_oracles(selected)
            clean_selected = [row for row in clean if row["restart"] == restart and row["workload_id"] == workload]
            agrs_clean = statistics.median(row["agrs_ttft_ms"] for row in clean_selected)
            deep_clean = statistics.median(row["deepep_ttft_ms"] for row in clean_selected)
            best_clean_backend = "agrs" if agrs_clean <= deep_clean else "deepep_ht"
            best_clean_ttft = min(agrs_clean, deep_clean)
            prof = [row for row in profile_request_rows if row["restart"] == restart and row["workload_id"] == workload and row["backend"] == best_clean_backend]
            profile_ttft = statistics.median(row["ttft_ms"] for row in prof)
            replaceable_static = float(oracle["best_static_ms"])
            replaceable_oracle = float(oracle["per_invocation_oracle_ms"])
            # Profiling-only route return and step-end synchronization inflate
            # absolute time.  Use the observed replaceable share on the same
            # profile timeline, then apply that fraction to clean TTFT.
            removable_profile_ms = replaceable_static - replaceable_oracle
            projected_pct = 100.0 * removable_profile_ms / profile_ttft
            output.append({
                "restart": restart,
                "workload_id": workload,
                "best_clean_backend": best_clean_backend,
                "agrs_clean_ttft_ms": agrs_clean,
                "deepep_clean_ttft_ms": deep_clean,
                "best_clean_ttft_ms": best_clean_ttft,
                "profile_ttft_ms": profile_ttft,
                "profile_observer_tax_pct": 100.0 * (profile_ttft - best_clean_ttft) / best_clean_ttft,
                "best_static_moe_ms": replaceable_static,
                "layer_oracle_moe_ms": replaceable_oracle,
                "moe_oracle_improvement_pct": oracle["per_invocation_improvement_pct"],
                "step_oracle_improvement_pct": oracle["per_step_improvement_pct"],
                "request_oracle_improvement_pct": oracle["per_request_improvement_pct"],
                "projected_clean_ttft_saved_ms": best_clean_ttft * projected_pct / 100.0,
                "projected_ttft_oracle_improvement_pct": projected_pct,
            })
    return output


def threshold_analysis(rows: list[dict]) -> dict:
    restarts = sorted({int(row["restart"]) for row in rows})
    if len(restarts) < 2:
        return {"status": "insufficient_restarts"}
    train = [row for row in rows if int(row["restart"]) == restarts[0]]
    test = [row for row in rows if int(row["restart"]) != restarts[0]]

    def pairs(values):
        grouped = defaultdict(dict)
        for row in values:
            key = (row["restart"], row["workload_id"], row["step_id"], row["layer_id"])
            grouped[key][row["backend"]] = row
        return [group for group in grouped.values() if set(group) == {"agrs", "deepep_ht"}]

    train_pairs, test_pairs = pairs(train), pairs(test)
    thresholds = sorted({round(float(pair["agrs"]["mean_fanout"]), 4) for pair in train_pairs})
    candidates = []
    for threshold in thresholds:
        cost = sum(
            float(pair["agrs" if pair["agrs"]["mean_fanout"] >= threshold else "deepep_ht"]["moe_total_ms"])
            for pair in train_pairs
        )
        candidates.append((cost, threshold))
    if not candidates or not test_pairs:
        return {"status": "insufficient_pairs"}
    _, threshold = min(candidates)
    totals = {
        backend: sum(float(pair[backend]["moe_total_ms"]) for pair in test_pairs)
        for backend in ("agrs", "deepep_ht")
    }
    best_static = min(totals.values())
    oracle = sum(min(float(pair["agrs"]["moe_total_ms"]), float(pair["deepep_ht"]["moe_total_ms"])) for pair in test_pairs)
    policy = sum(float(pair["agrs" if pair["agrs"]["mean_fanout"] >= threshold else "deepep_ht"]["moe_total_ms"]) for pair in test_pairs)
    oracle_gain = best_static - oracle
    return {
        "status": "held_out_restart",
        "train_restart": restarts[0],
        "test_restarts": restarts[1:],
        "threshold": threshold,
        "best_static_ms": best_static,
        "perfect_oracle_ms": oracle,
        "fanout_threshold_ms": policy,
        "perfect_oracle_improvement_pct": 100.0 * oracle_gain / best_static,
        "threshold_improvement_pct": 100.0 * (best_static - policy) / best_static,
        "oracle_recovery_pct": 100.0 * (best_static - policy) / oracle_gain if oracle_gain > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agrs-clean", type=Path, action="append", required=True)
    parser.add_argument("--deepep-clean", type=Path, action="append", required=True)
    parser.add_argument("--agrs-profile", type=Path, action="append", required=True)
    parser.add_argument("--deepep-profile", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lengths = {len(args.agrs_clean), len(args.deepep_clean), len(args.agrs_profile), len(args.deepep_profile)}
    if len(lengths) != 1:
        raise SystemExit("all run lists must have the same number of restarts")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_rows, correctness = [], []
    invocation_rows, path_proofs, profile_request_rows = [], [], []
    for restart, (ac, dc, ap, dp) in enumerate(zip(args.agrs_clean, args.deepep_clean, args.agrs_profile, args.deepep_profile), 1):
        paired, proof = clean_pair_rows(ac, dc, restart)
        clean_rows.extend(paired)
        correctness.append(proof)
        for run in (ap, dp):
            rows, path = profile_invocations(run, restart)
            invocation_rows.extend(rows)
            path_proofs.append({"restart": restart, "run": str(run), **path})
            profile_request_rows.extend(request_table(run, restart))

    write_csv(args.output_dir / "clean_request_pairs.csv", clean_rows)
    write_csv(args.output_dir / "aligned_profile_invocations.csv", invocation_rows)
    (args.output_dir / "correctness.json").write_text(json.dumps(correctness, indent=2) + "\n")
    (args.output_dir / "runtime_path_execution.json").write_text(json.dumps(path_proofs, indent=2) + "\n")

    # Route identity is a hard precondition for any backend oracle.  Validate
    # it before writing oracle artifacts; full-model executions can generate
    # identical tokens while tiny numerical differences change later routes.
    paired = defaultdict(dict)
    for row in invocation_rows:
        key = (row["restart"], row["workload_id"], row["step_id"], row["layer_id"])
        paired[key][row["backend"]] = row
    delta_rows = []
    for key, values in sorted(paired.items()):
        if set(values) != {"agrs", "deepep_ht"}:
            continue
        a, d = values["agrs"], values["deepep_ht"]
        if a["route_hash"] != d["route_hash"]:
            raise ValueError(f"route mismatch {key}")
        delta_rows.append({
            "restart": key[0],
            "workload_id": key[1],
            "step_id": key[2],
            "layer_id": key[3],
            "num_tokens": a["num_tokens"],
            "mean_fanout": a["mean_fanout"],
            "frac_full_fanout": a["frac_full_fanout"],
            "rank_max_mean": a["rank_max_mean"],
            "expert_max_mean": a["expert_max_mean"],
            "agrs_moe_ms": a["moe_total_ms"],
            "deepep_moe_ms": d["moe_total_ms"],
            "deepep_minus_agrs_ms": d["moe_total_ms"] - a["moe_total_ms"],
            "winner": "agrs" if a["moe_total_ms"] < d["moe_total_ms"] else "deepep_ht",
        })
    write_csv(args.output_dir / "backend_delta_vs_fanout.csv", delta_rows)

    oracle_rows = per_workload_oracle(invocation_rows, clean_rows, profile_request_rows)
    write_csv(args.output_dir / "oracle_by_workload_restart.csv", oracle_rows)
    threshold = threshold_analysis(invocation_rows)
    (args.output_dir / "fanout_threshold_oracle.json").write_text(json.dumps(threshold, indent=2) + "\n")
    if delta_rows:
        y = np.asarray([row["deepep_minus_agrs_ms"] for row in delta_rows], dtype=float)
        x = np.asarray([[1.0, row["mean_fanout"], np.log1p(row["num_tokens"]), row["rank_max_mean"], row["expert_max_mean"]] for row in delta_rows])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        prediction = x @ beta
        r2 = 1.0 - float(np.square(y - prediction).sum() / np.square(y - y.mean()).sum()) if np.square(y - y.mean()).sum() else 0.0
        regression = {"features": ["intercept", "mean_fanout", "log1p_M", "rank_max_mean", "expert_max_mean"], "coefficients": beta.tolist(), "r2_in_sample_diagnostic": r2, "n": len(y)}
        (args.output_dir / "fanout_explainability.json").write_text(json.dumps(regression, indent=2) + "\n")

        figures = args.output_dir / "figures"
        figures.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        for workload in sorted({row["workload_id"] for row in delta_rows}):
            selected = [row for row in delta_rows if row["workload_id"] == workload]
            ax.scatter([row["mean_fanout"] for row in selected], [row["deepep_minus_agrs_ms"] for row in selected], label=workload, alpha=.6)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("mean destination-rank fanout")
        ax.set_ylabel("DeepEP HT - AGRS whole-MoE (ms)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / "B6_backend_delta_vs_real_fanout.png", dpi=180)
        plt.close(fig)

        if oracle_rows:
            labels = [f"r{row['restart']}:{row['workload_id']}" for row in oracle_rows]
            xloc = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(max(8, len(labels) * .7), 5))
            ax.bar(xloc, [row["moe_oracle_improvement_pct"] for row in oracle_rows], label="MoE oracle")
            ax.bar(xloc, [row["projected_ttft_oracle_improvement_pct"] for row in oracle_rows], label="projected TTFT oracle")
            ax.set_xticks(xloc, labels=labels, rotation=45, ha="right")
            ax.set_ylabel("improvement (%)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / "B8_communication_vs_ttft_oracle.png", dpi=180)
            plt.close(fig)

    result = {
        "restarts": next(iter(lengths)),
        "clean_pairs": len(clean_rows),
        "profile_invocations": len(invocation_rows),
        "route_aligned_pairs": len(delta_rows),
        "correctness": correctness,
        "runtime_paths": path_proofs,
        "oracle_rows": oracle_rows,
        "threshold": threshold,
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
