#!/usr/bin/env python3
"""Aggregate GPU results and run an EP-stage online route-replay harness."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rank_critical(pattern: str, keys: list[str], values: list[str]):
    frames = [pd.read_json(f, lines=True) for f in glob.glob(pattern)]
    if not frames: raise RuntimeError(f"no inputs for {pattern}")
    d = pd.concat(frames, ignore_index=True)
    return d.groupby(keys + ["repeat"], as_index=False)[values].max()


def interp_model(table: pd.DataFrame, policy: str):
    x = table[table.policy == policy].sort_values("global_m")
    xp = np.log2(x.global_m.to_numpy(float)); yp = x.transaction_ms.to_numpy(float)
    def predict(m): return float(np.interp(math.log2(max(1, m)), xp, yp))
    return predict


@dataclass
class Request:
    rid: int
    task: str
    arrival: float
    waves: list[int]
    cursor: int = 0
    queue_wait: float = 0.0


def percentile(values, q): return float(np.percentile(values, q)) if values else 0.0


def simulate(arrivals, tasks, templates, service, batch_slots, pair_factor, output_tokens):
    pending = [Request(i, task, float(at), templates[task])
               for i, (at, task) in enumerate(zip(arrivals, tasks))]
    ready: list[Request] = []; completed = []; t = 0.0; next_i = 0
    total_rows = 0
    while len(completed) < len(pending):
        while next_i < len(pending) and pending[next_i].arrival <= t:
            pending[next_i].ready_since = pending[next_i].arrival
            ready.append(pending[next_i]); next_i += 1
        if not ready:
            t = pending[next_i].arrival; continue
        ready.sort(key=lambda r: (r.ready_since, r.rid))
        selected = ready[: min(batch_slots, len(ready))]
        del ready[:len(selected)]
        costs = []; rows = []
        for req in selected:
            req.queue_wait += t - req.ready_since
            m = req.waves[req.cursor]; rows.append(m); costs.append(service(m))
        duration = sum(costs) * (pair_factor if len(selected) > 1 else 1.0)
        t += duration; total_rows += sum(rows)
        for req in selected:
            req.cursor += 1
            if req.cursor == len(req.waves):
                req.finish = t; completed.append(req)
            else:
                req.ready_since = t; ready.append(req)
    lat = [r.finish - r.arrival for r in completed]
    waits = [r.queue_wait for r in completed]
    makespan = max(r.finish for r in completed) - min(r.arrival for r in completed)
    return {"requests_per_s": len(completed) / makespan,
            "generated_tokens_per_s": sum(output_tokens[r.task] for r in completed) / makespan,
            "refinement_waves_per_s": sum(len(r.waves) for r in completed) / makespan,
            "rows_per_s": total_rows / makespan,
            "request_latency_p50_ms": percentile(lat, 50) * 1000,
            "request_latency_p95_ms": percentile(lat, 95) * 1000,
            "request_latency_p99_ms": percentile(lat, 99) * 1000,
            "queue_delay_p50_ms": percentile(waits, 50) * 1000,
            "queue_delay_p95_ms": percentile(waits, 95) * 1000,
            "queue_delay_p99_ms": percentile(waits, 99) * 1000}


def simulate_closed_loop(concurrency, n, task_sequence, templates, service,
                         batch_slots, pair_factor, output_tokens):
    created = 0; completed = []; ready = []; t = 0.0; total_rows = 0
    def add_request(now):
        nonlocal created
        task = task_sequence[created % len(task_sequence)]
        req = Request(created, task, now, templates[task]); req.ready_since = now
        ready.append(req); created += 1
    for _ in range(min(concurrency, n)): add_request(0.0)
    while len(completed) < n:
        ready.sort(key=lambda r: (r.ready_since, r.rid))
        selected = ready[: min(batch_slots, len(ready))]; del ready[:len(selected)]
        costs = []; rows = []
        for req in selected:
            req.queue_wait += t - req.ready_since
            m = req.waves[req.cursor]; rows.append(m); costs.append(service(m))
        t += sum(costs) * (pair_factor if len(selected) > 1 else 1.0)
        total_rows += sum(rows)
        for req in selected:
            req.cursor += 1
            if req.cursor == len(req.waves):
                req.finish = t; completed.append(req)
                if created < n: add_request(t)
            else:
                req.ready_since = t; ready.append(req)
    lat = [r.finish - r.arrival for r in completed]
    waits = [r.queue_wait for r in completed]
    return {"requests_per_s": n / t,
            "generated_tokens_per_s": sum(output_tokens[r.task] for r in completed) / t,
            "refinement_waves_per_s": sum(len(r.waves) for r in completed) / t,
            "rows_per_s": total_rows / t,
            "request_latency_p50_ms": percentile(lat, 50) * 1000,
            "request_latency_p95_ms": percentile(lat, 95) * 1000,
            "request_latency_p99_ms": percentile(lat, 99) * 1000,
            "queue_delay_p50_ms": percentile(waits, 50) * 1000,
            "queue_delay_p95_ms": percentile(waits, 95) * 1000,
            "queue_delay_p99_ms": percentile(waits, 99) * 1000}


def poisson_arrivals(n, rate, seed):
    rng = random.Random(seed); t = 0.0; out = []
    for _ in range(n): t += rng.expovariate(rate); out.append(t)
    return out


def burst_arrivals(n, base_interval, seed):
    rng = random.Random(seed); out = []; t = 0.0
    pattern = [(0.5, 1), (0.08, 4), (0.4, 1), (0.04, 8)]
    while len(out) < n:
        gap_scale, count = pattern[(len(out) // 20) % len(pattern)]
        for _ in range(min(count, n - len(out))):
            t += base_interval * gap_scale * (0.7 + 0.6 * rng.random()); out.append(t)
        t += base_interval * (3.0 if count == 1 else 0.2)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); a = ap.parse_args()
    root = a.root; fig = root / "figures"; fig.mkdir(parents=True, exist_ok=True)

    cross = rank_critical(str(root / "results/raw/crossover_*_r1/rank*.jsonl"),
                          ["case_id", "global_m", "routing", "policy"], ["transaction_ms"])
    cs = cross.groupby(["global_m", "routing", "policy"], as_index=False).transaction_ms.median()
    cs.to_csv(root / "M_CROSSOVER.csv", index=False)

    cap = rank_critical(str(root / "results/raw/capacity_matrix_r1/rank*.jsonl"),
                        ["global_m", "contract_capacity_per_rank", "policy"],
                        ["transaction_ms", "dispatch_ms", "combine_ms"])
    caps = cap.groupby(["global_m", "contract_capacity_per_rank", "policy"], as_index=False).median(numeric_only=True)
    caps["capacity_is_per_source_rank"] = True
    caps.to_csv(root / "CAPACITY_MATRIX.csv", index=False)

    inflight_frames = []
    for directory in glob.glob(str(root / "results/raw/*inflight_r*")):
        raw = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(directory + "/rank*.jsonl")])
        d = raw.groupby(["workload", "q", "policy", "repeat"], as_index=False).agg(
            wall_ms=("wall_ms", "max"), waves_per_s=("waves_per_s", "min"),
            p50_completion_ms=("p50_completion_ms", "max"),
            p95_completion_ms=("p95_completion_ms", "max"),
            p99_completion_ms=("p99_completion_ms", "max"),
            max_relative_l2=("max_relative_l2", "max"))
        s = d.groupby(["workload", "q", "policy"], as_index=False).median(numeric_only=True)
        s["restart"] = Path(directory).name; inflight_frames.append(s)
    ir = pd.concat(inflight_frames, ignore_index=True)
    ir.to_csv(root / "INFLIGHT_RESTARTS.csv", index=False)
    ins = ir.groupby(["workload", "q", "policy"], as_index=False).median(numeric_only=True)
    ins.to_csv(root / "INFLIGHT_RESULTS.csv", index=False)

    real = cs[cs.routing == "real_like"]
    ll = interp_model(real, "low_latency"); normal = interp_model(real, "normal")
    stream = pd.read_json(root / "REFINEMENT_STREAMS.jsonl", lines=True)
    templates = {task: list(g.sort_values("wave").global_m.astype(int))
                 for task, g in stream.groupby("dataset")}
    output_tokens = {"gsm8k": 256, "humaneval": 384}
    mean_ll_s = np.mean([sum(ll(m) for m in waves) for waves in templates.values()]) / 1000
    # Three-restart actual real-age-offset Q16 ratio. This is the strongest legal LL baseline.
    realq = ins[(ins.workload == "real_age_offset") & (ins.q == 16)].set_index("policy")
    pair_factor = float(realq.loc["ll_two_slot", "wall_ms"] / realq.loc["ll_serial", "wall_ms"])
    q8 = float(ins[(ins.workload == "real_age_offset") & (ins.q == 8) &
                   (ins.policy == "ll_two_slot")].waves_per_s.iloc[0])
    q16 = float(realq.loc["ll_two_slot", "waves_per_s"])
    residual_inflight = max(0.0, q16 / q8 - 1.0)
    # Capacity has no monotonic tax. A conservative 1% optimistic ceiling is retained.
    capacity_ceiling = 0.01

    policies = {
        "normal_only": (normal, 1, 1.0),
        "ll_only": (ll, 2, pair_factor),
        "existing_backend_oracle": (lambda m: min(ll(m), normal(m)), 2, pair_factor),
        "simple_normal_ll_switch": (lambda m: normal(m) if m >= 6144 else ll(m), 2, pair_factor),
        "o1_variable_capacity_ll": (lambda m: ll(m) * (1 - capacity_ceiling), 2, pair_factor),
        "o2_unlimited_inflight_ll": (lambda m: ll(m) / (1 + residual_inflight), 2, pair_factor),
        "o3_combined": (lambda m: ll(m) * (1 - capacity_ceiling) / (1 + residual_inflight), 2, pair_factor),
        "o4_credible_refinestreamep": (lambda m: ll(m) * (1 - capacity_ceiling) / (1 + residual_inflight), 2, pair_factor),
    }
    effective_request_service = mean_ll_s * pair_factor
    saturation_rps = 1 / effective_request_service
    results = []
    for seed in range(5):
        rng = random.Random(9000 + seed)
        tasks = [rng.choice(list(templates)) for _ in range(240)]
        for load in [0.30, 0.60, 0.825, 0.95]:
            arr = poisson_arrivals(len(tasks), saturation_rps * load, 12000 + seed)
            for name, (fn, slots, factor) in policies.items():
                results.append({"arrival": "poisson", "offered_load": load, "seed": seed,
                                "policy": name, **simulate(arr, tasks, templates,
                                lambda m, f=fn: f(m) / 1000, slots, factor, output_tokens)})
        base = effective_request_service / 0.825
        arr = burst_arrivals(len(tasks), base, 15000 + seed)
        for name, (fn, slots, factor) in policies.items():
            results.append({"arrival": "bursty", "offered_load": 0.825, "seed": seed,
                            "policy": name, **simulate(arr, tasks, templates,
                            lambda m, f=fn: f(m) / 1000, slots, factor, output_tokens)})
        for concurrency in [1, 2, 4, 8, 16, 32]:
            n = 192; tasks2 = [rng.choice(list(templates)) for _ in range(n)]
            for name, (fn, slots, factor) in policies.items():
                results.append({"arrival": "closed_loop", "offered_load": concurrency, "seed": seed,
                                "policy": name, **simulate_closed_loop(concurrency, n, tasks2,
                                templates, lambda m, f=fn: f(m) / 1000,
                                slots, factor, output_tokens)})
    sr = pd.DataFrame(results); sr.to_csv(root / "SERVING_RESULTS.csv", index=False)
    ss = sr.groupby(["arrival", "offered_load", "policy"], as_index=False).median(numeric_only=True)
    base = ss[ss.policy == "ll_only"].set_index(["arrival", "offered_load"])
    oracle_rows = []
    for _, row in ss.iterrows():
        b = base.loc[(row.arrival, row.offered_load)]
        oracle_rows.append({**row.to_dict(),
            "throughput_gain_vs_ll_pct": (row.requests_per_s / b.requests_per_s - 1) * 100,
            "p50_reduction_vs_ll_pct": (1 - row.request_latency_p50_ms / b.request_latency_p50_ms) * 100,
            "p95_reduction_vs_ll_pct": (1 - row.request_latency_p95_ms / b.request_latency_p95_ms) * 100,
            "p99_reduction_vs_ll_pct": (1 - row.request_latency_p99_ms / b.request_latency_p99_ms) * 100,
            "route_replay_only": True})
    oracle = pd.DataFrame(oracle_rows); oracle.to_csv(root / "SERVING_ORACLES.csv", index=False)

    # Required figures.
    plt.figure(figsize=(7, 4))
    for policy, g in cs[cs.routing == "real_like"].groupby("policy"):
        plt.plot(g.global_m, g.transaction_ms, marker="o", label=policy)
    plt.xscale("log", base=2); plt.yscale("log"); plt.xlabel("Global M"); plt.ylabel("transaction ms")
    plt.legend(); plt.tight_layout(); plt.savefig(fig / "01_normal_ll_crossover.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4))
    for policy, g in cs[cs.routing == "real_like"].groupby("policy"):
        plt.plot(g.global_m, g.global_m / (g.transaction_ms / 1000), marker="o", label=policy)
    plt.xscale("log", base=2); plt.yscale("log"); plt.xlabel("Global M"); plt.ylabel("rows/s")
    plt.legend(); plt.tight_layout(); plt.savefig(fig / "02_normal_ll_throughput.png", dpi=180); plt.close()
    llcap = caps[caps.policy == "low_latency"]
    pivot = llcap.pivot(index="global_m", columns="contract_capacity_per_rank", values="transaction_ms")
    plt.figure(figsize=(8, 4)); plt.imshow(pivot, aspect="auto", origin="lower")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45); plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("LL capacity/source rank"); plt.ylabel("Global M"); plt.colorbar(label="ms")
    plt.tight_layout(); plt.savefig(fig / "03_capacity_heatmap.png", dpi=180); plt.close()
    capacities = np.array([32, 64, 128, 256, 512, 1024, 2048])
    # Legacy V1 exact heap hint is 8.125 MiB per capacity unit for this geometry.
    heap_gib = capacities * 8.125 / 1024
    double_recv_gib = 2 * 64 * 4 * capacities * 4096 * 2 / 2**30
    plt.figure(figsize=(7, 4)); plt.plot(capacities, heap_gib, marker="o", label="NVSHMEM heap")
    plt.plot(capacities, double_recv_gib, marker="o", label="two BF16 recv buffers")
    plt.xscale("log", base=2); plt.xlabel("capacity/source rank"); plt.ylabel("GiB/rank")
    plt.legend(); plt.tight_layout(); plt.savefig(fig / "04_ll_memory_vs_capacity.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4))
    for p, g in ins[ins.workload == "real_age_offset"].groupby("policy"):
        plt.plot(g.q, g.waves_per_s, marker="o", label=p)
    plt.xlabel("Ready waves Q"); plt.ylabel("waves/s"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "05_inflight_throughput.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4))
    for p, g in ins[ins.workload == "real_age_offset"].groupby("policy"):
        plt.plot(g.q, g.p99_completion_ms, marker="o", label=p)
    plt.xlabel("Ready waves Q"); plt.ylabel("completion p99 ms"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "06_inflight_latency.png", dpi=180); plt.close()
    q16plot = ins[ins.q == 16]
    plt.figure(figsize=(8, 4))
    for p, g in q16plot.groupby("policy"):
        plt.plot(g.workload, g.waves_per_s, marker="o", label=p)
    plt.ylabel("waves/s at Q=16"); plt.xticks(rotation=20); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "07_homogeneous_vs_mixed.png", dpi=180); plt.close()
    plt.figure(figsize=(8, 4))
    for task, g in stream.groupby("dataset"):
        plt.plot(g.wave, g.global_m, marker=".", label=task)
    plt.xlabel("refinement wave"); plt.ylabel("compacted global M"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "08_real_refinement_stream.png", dpi=180); plt.close()
    base_stream = stream[stream.dataset == "gsm8k"].sort_values("wave")
    plt.figure(figsize=(8, 4))
    for i, off in enumerate([0, 8, 20, 36]):
        x = np.arange(len(base_stream)) + off
        plt.plot(x, base_stream.global_m, alpha=0.8, label=f"request {i} offset {off}")
    plt.xlabel("serving time index"); plt.ylabel("ready-wave M"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "09_interleaved_stream.png", dpi=180); plt.close()
    po = ss[(ss.arrival == "poisson") & ss.policy.isin(["ll_only", "normal_only", "o4_credible_refinestreamep"])]
    plt.figure(figsize=(7, 4))
    for p, g in po.groupby("policy"):
        plt.plot(g.offered_load, g.requests_per_s, marker="o", label=p)
    plt.xlabel("offered load"); plt.ylabel("requests/s (EP-stage replay)"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "10_throughput_load_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4))
    for p, g in po.groupby("policy"):
        plt.plot(g.offered_load, g.request_latency_p99_ms, marker="o", label=p)
    plt.xlabel("offered load"); plt.ylabel("request p99 ms (EP-stage replay)"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "11_tail_latency_curve.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 4))
    for p, g in po.groupby("policy"):
        plt.plot(g.offered_load, g.queue_delay_p99_ms, marker="o", label=p)
    plt.xlabel("offered load"); plt.ylabel("queue p99 ms"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "12_queue_delay.png", dpi=180); plt.close()
    util = po.copy(); util["ep_service_utilization_proxy"] = np.minimum(
        1.0, util.requests_per_s / saturation_rps)
    plt.figure(figsize=(7, 4))
    for p, g in util.groupby("policy"):
        plt.plot(g.offered_load, g.ep_service_utilization_proxy, marker="o", label=p)
    plt.xlabel("offered load"); plt.ylabel("EP service utilization proxy"); plt.legend(); plt.tight_layout()
    plt.savefig(fig / "13_backend_utilization_proxy.png", dpi=180); plt.close()
    key = oracle[(oracle.arrival == "poisson") & (oracle.offered_load == 0.825)]
    for number, policy, title in [
        (14, "existing_backend_oracle", "Best-existing backend oracle"),
        (15, "o1_variable_capacity_ll", "Variable-capacity oracle"),
        (16, "o2_unlimited_inflight_ll", "Unlimited-inflight oracle"),
        (17, "o4_credible_refinestreamep", "Credible RefineStreamEP oracle")]:
        row = key[key.policy == policy].iloc[0]
        vals = [row.throughput_gain_vs_ll_pct, row.p95_reduction_vs_ll_pct,
                row.p99_reduction_vs_ll_pct]
        plt.figure(figsize=(5, 3.5)); plt.bar(["throughput", "p95", "p99"], vals)
        plt.ylabel("improvement vs legal LL (%)"); plt.title(title); plt.tight_layout()
        plt.savefig(fig / f"{number:02d}_{policy}.png", dpi=180); plt.close()
    summary = {"pair_factor": pair_factor, "q8_waves_per_s": q8, "q16_waves_per_s": q16,
               "residual_unlimited_inflight_ceiling": residual_inflight,
               "capacity_latency_ceiling": capacity_ceiling, "ep_stage_saturation_rps": saturation_rps}
    (root / "results" / "serving_model_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
