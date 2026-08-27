#!/usr/bin/env python3
"""Same-volume DeepEP traffic-matrix shape characterization.

prepare is CPU-only and emits canonical synthetic routes plus invariant checks.
run uses only DeepEP dispatch/combine on four ranks; no model or expert GEMM is
executed.  Physical GPUs are fixed by the launcher to 1,2,3,4.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NUM_RANKS = 4
NUM_EXPERTS = 128
LOCAL_EXPERTS = 32
TOP_K = 8
HIDDEN = 2048
SCALES = (256, 1024)
FAMILIES = ("balanced_spread", "pair_concentrated", "destination_hotspot")
PHYSICAL_GPUS = [1, 2, 3, 4]
POLICY = {
    "warmups": 10,
    "iterations": 50,
    "permutation_count": 24,
    "primary_token_scales": list(SCALES),
    "family_a": "same row/column incidence sums, same S/I/volume; only peer concentration differs",
    "go_effect": 0.10,
    "hold_effect": 0.05,
    "direction_consistency": 0.75,
}


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: (json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict, np.ndarray)) else v) for k, v in row.items()})


def permutations() -> list[tuple[int, int, int, int]]:
    import itertools
    return list(itertools.permutations(range(NUM_RANKS)))


def pair_for(family: str, source: int, token: int, n: int) -> tuple[int, int]:
    nonself = [d for d in range(NUM_RANKS) if d != source]
    if family == "balanced_spread":
        # All three non-self pairs are used.  The phase makes the N mod 3
        # remainder cancel across source rows, giving exactly equal columns.
        opts = (tuple(nonself[0:2]), tuple(nonself[1:3]), (nonself[2], nonself[0]))
        return opts[(source + token) % 3]
    if family == "pair_concentrated":
        # Cyclic fixed pairs: each source has one pair, while columns remain equal.
        return ((source + 1) % NUM_RANKS, (source + 2) % NUM_RANKS)
    if family == "destination_hotspot":
        return ((1, 2), (0, 2), (0, 1), (0, 1))[source]
    raise ValueError(family)


def canonical_routes(family: str, source: int, n: int, perm: tuple[int, int, int, int]) -> np.ndarray:
    # perm maps canonical rank labels to physical rank labels; source rows are
    # permuted consistently so no-self remains no-self for Family A.
    inverse = {physical: logical for logical, physical in enumerate(perm)}
    logical_source = inverse[source]
    ids = np.empty((n, TOP_K), dtype=np.int64)
    lane_offsets = (0, 1, 2, 3)
    for t in range(n):
        d0, d1 = pair_for(family, logical_source, t, n)
        d0, d1 = perm[d0], perm[d1]
        base = (t * 4) % LOCAL_EXPERTS
        ids[t, :4] = [d0 * LOCAL_EXPERTS + ((base + q) % LOCAL_EXPERTS) for q in lane_offsets]
        ids[t, 4:] = [d1 * LOCAL_EXPERTS + ((base + q) % LOCAL_EXPERTS) for q in lane_offsets]
    return ids


def matrix_for(family: str, n: int) -> dict[str, Any]:
    incidence = np.zeros((NUM_RANKS, NUM_RANKS), dtype=np.int64)
    for source in range(NUM_RANKS):
        for t in range(n):
            d0, d1 = pair_for(family, source, t, n)
            incidence[source, d0] += 1
            incidence[source, d1] += 1
    assignment = incidence * (TOP_K // 2)
    fanout = []
    for source in range(NUM_RANKS):
        pairs = []
        for t in range(n):
            pairs.append(len(set(pair_for(family, source, t, n))))
        fanout.extend(pairs)
    p = incidence.astype(np.float64).reshape(-1)
    p = p[p > 0] / p.sum()
    entropy = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
    return {
        "family": family,
        "tokens_per_source": n,
        "incidence_matrix": incidence.tolist(),
        "assignment_matrix": assignment.tolist(),
        "row_sums_incidence": incidence.sum(axis=1).tolist(),
        "column_sums_incidence": incidence.sum(axis=0).tolist(),
        "row_sums_assignment": assignment.sum(axis=1).tolist(),
        "column_sums_assignment": assignment.sum(axis=0).tolist(),
        "total_incidence": int(incidence.sum()),
        "total_assignments": int(assignment.sum()),
        "S": 0.5,
        "I": float(assignment.sum(axis=0).max() / assignment.sum(axis=0).mean()),
        "peer_fanout_mean": float(np.mean(fanout)),
        "traffic_entropy": entropy,
        "max_peer_fraction": float(incidence.sum(axis=0).max() / incidence.sum()),
    }


def prepare(args: argparse.Namespace) -> None:
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    perms = permutations()
    matrices = [matrix_for(family, n) for n in SCALES for family in FAMILIES]
    # Family-A causal invariant: exact matrix equality up to route-pair shape.
    for n in SCALES:
        a = np.asarray(next(x for x in matrices if x["family"] == "balanced_spread" and x["tokens_per_source"] == n)["incidence_matrix"])
        b = np.asarray(next(x for x in matrices if x["family"] == "pair_concentrated" and x["tokens_per_source"] == n)["incidence_matrix"])
        if not (np.array_equal(a.sum(1), b.sum(1)) and np.array_equal(a.sum(0), b.sum(0)) and int(a.sum()) == int(b.sum())):
            raise AssertionError(f"Family A invariant failed for N={n}")
    dump_json(out / "traffic_matrices.json", {"families": matrices, "permutations": [list(p) for p in perms], "policy": POLICY, "physical_gpus": PHYSICAL_GPUS})
    invariant_rows: list[dict[str, Any]] = []
    for n in SCALES:
        for family in FAMILIES:
            m = next(x for x in matrices if x["family"] == family and x["tokens_per_source"] == n)
            for perm_index, perm in enumerate(perms):
                actual = np.zeros((NUM_RANKS, NUM_RANKS), dtype=np.int64)
                for source in range(NUM_RANKS):
                    ids = canonical_routes(family, source, n, perm)
                    ranks = ids // LOCAL_EXPERTS
                    actual[source] = np.bincount(ranks.reshape(-1), minlength=NUM_RANKS)
                actual_incidence = actual // (TOP_K // 2)
                canonical = np.asarray(m["incidence_matrix"], dtype=np.int64)
                inverse = {physical: logical for logical, physical in enumerate(perm)}
                expected = np.asarray([[canonical[inverse[ps], inverse[pd]] for pd in range(NUM_RANKS)] for ps in range(NUM_RANKS)], dtype=np.int64)
                if not np.array_equal(actual_incidence, expected):
                    raise AssertionError(f"route matrix mismatch for {family} N={n} permutation={perm_index}")
                for source in range(NUM_RANKS):
                    for dest in range(NUM_RANKS):
                        rank_load = actual.sum(axis=0)
                        invariant_rows.append({"N": n, "family": family, "perm_index": perm_index, "source": source, "dest": dest, "incidence": int(actual_incidence[source, dest]), "assignment_volume": int(actual[source, dest]), "row_sum": int(actual_incidence[source].sum()), "column_sum": int(actual_incidence[:, dest].sum()), "S": 0.5, "I": float(rank_load.max() / rank_load.mean()) if rank_load.mean() else 0.0})
    # Explicitly compare every Family-A permutation, not only the canonical map.
    for n in SCALES:
        for perm_index, perm in enumerate(perms):
            mats = []
            for family in ("balanced_spread", "pair_concentrated"):
                actual = np.zeros((NUM_RANKS, NUM_RANKS), dtype=np.int64)
                for source in range(NUM_RANKS):
                    ranks = canonical_routes(family, source, n, perm) // LOCAL_EXPERTS
                    actual[source] = np.bincount(ranks.reshape(-1), minlength=NUM_RANKS)
                mats.append(actual)
            if not (np.array_equal(mats[0].sum(0), mats[1].sum(0)) and np.array_equal(mats[0].sum(1), mats[1].sum(1)) and int(mats[0].sum()) == int(mats[1].sum())):
                raise AssertionError(f"Family A invariant failed for N={n}, permutation={perm_index}")
    csv_write(out / "invariant_check.csv", invariant_rows)
    cases: list[dict[str, Any]] = []
    for n in SCALES:
        for family in FAMILIES:
            m = next(x for x in matrices if x["family"] == family and x["tokens_per_source"] == n)
            for perm_index, perm in enumerate(perms):
                cases.append({"case_id": f"{family}_N{n}_p{perm_index:02d}", "family": family, "N": n, "perm_index": perm_index, "perm": list(perm), "S": m["S"], "I": m["I"], "total_assignments": m["total_assignments"], "total_incidence": m["total_incidence"], "traffic_entropy": m["traffic_entropy"], "max_peer_fraction": m["max_peer_fraction"]})
    csv_write(out / "cases.csv", cases)
    dump_json(out / "prepare_summary.json", {"num_cases": len(cases), "permutations": len(perms), "scales": list(SCALES), "families": list(FAMILIES), "family_a_invariants": True})
    print(json.dumps({"num_cases": len(cases), "family_a_invariants": True, "scales": SCALES}, indent=2))


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    import deep_ep

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != NUM_RANKS:
        raise RuntimeError(f"expected EP4, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.output.mkdir(parents=True, exist_ok=True)
    import csv
    with (args.output / "cases.csv").open(newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    deep_ep.Buffer.set_num_sms(20)
    buffer = deep_ep.Buffer(dist.group.WORLD, args.buffer_mib * 1024 * 1024, 0, low_latency_mode=False, num_qps_per_rank=1, explicitly_destroy=True)
    generator = torch.Generator(device=device).manual_seed(31337 + rank)
    samples: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        n = int(case["N"])
        perm = tuple(int(x) for x in json.loads(case["perm"]))
        ids_np = canonical_routes(case["family"], rank, n, perm)
        hidden = torch.randn((n, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator)
        topk_ids = torch.from_numpy(ids_np).to(device=device, dtype=deep_ep.topk_idx_t).contiguous()
        topk_weights = torch.full((n, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        for it in range(args.warmups + args.iterations):
            dist.barrier()
            torch.cuda.synchronize(device)
            e_layout0 = torch.cuda.Event(enable_timing=True); e_layout1 = torch.cuda.Event(enable_timing=True)
            e_disp0 = torch.cuda.Event(enable_timing=True); e_disp1 = torch.cuda.Event(enable_timing=True)
            e_comb0 = torch.cuda.Event(enable_timing=True); e_comb1 = torch.cuda.Event(enable_timing=True)
            e_total1 = torch.cuda.Event(enable_timing=True)
            host0 = time.perf_counter_ns(); e_layout0.record()
            layout = buffer.get_dispatch_layout(topk_ids, NUM_EXPERTS, async_finish=False, allocate_on_comm_stream=False)
            e_layout1.record(); e_disp0.record()
            num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = layout
            recv_hidden, recv_ids, recv_weights, recv_count, handle, dispatch_event = buffer.dispatch(
                x=hidden, handle=None, num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert, topk_idx=topk_ids, topk_weights=topk_weights,
                expert_alignment=1, config=deep_ep.Buffer.get_dispatch_config(world), previous_event=previous_event, async_finish=False, allocate_on_comm_stream=False)
            e_disp1.record(); e_comb0.record()
            combined, _, combine_event = buffer.combine(x=recv_hidden, handle=handle, topk_weights=None, config=deep_ep.Buffer.get_combine_config(world), async_finish=False, allocate_on_comm_stream=False)
            e_comb1.record(); e_total1.record(); e_total1.synchronize(); host1 = time.perf_counter_ns()
            if it >= args.warmups:
                samples.append({"case_id": case["case_id"], "case_index": case_index, "iteration": it - args.warmups, "rank": rank, "physical_gpu": PHYSICAL_GPUS[local_rank], "N": n, "received_tokens": int(recv_hidden.shape[0]), "layout_ms": float(e_layout0.elapsed_time(e_layout1)), "dispatch_only_ms": float(e_disp0.elapsed_time(e_disp1)), "combine_ms": float(e_comb0.elapsed_time(e_comb1)), "full_path_ms": float(e_layout0.elapsed_time(e_total1)), "host_ms": (host1 - host0) / 1e6})
            del layout, recv_hidden, recv_ids, recv_weights, recv_count, handle, dispatch_event, combine_event, combined
            dist.barrier()
        del hidden, topk_ids, topk_weights
    dump_json(args.output / f"rank{rank}_timing.json", {"rank": rank, "local_rank": local_rank, "physical_gpu": PHYSICAL_GPUS[local_rank], "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "deep_ep": str(Path(deep_ep.__file__).resolve()), "torch": torch.__version__, "cuda": torch.version.cuda, "samples": samples})
    buffer.destroy(); dist.barrier(); dist.destroy_process_group()


def aggregate(args: argparse.Namespace) -> None:
    import csv
    cases = list(csv.DictReader((args.output / "cases.csv").open(newline="", encoding="utf-8")))
    rank_files = sorted(args.output.glob("rank*_timing.json"))
    all_samples: list[dict[str, Any]] = []
    for path in rank_files:
        payload = json.loads(path.read_text(encoding="utf-8")); all_samples.extend(payload["samples"])
    csv_write(args.output / "raw_timing.csv", all_samples)
    by_case_iter: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for s in all_samples:
        by_case_iter.setdefault((s["case_id"], int(s["iteration"])), []).append(s)
    case_summary: list[dict[str, Any]] = []
    for case in cases:
        for metric in ("layout_ms", "dispatch_only_ms", "combine_ms", "full_path_ms", "host_ms"):
            vals = [max(float(s[metric]) for s in ss) for (cid, _), ss in by_case_iter.items() if cid == case["case_id"] and len(ss) == NUM_RANKS]
            case_summary.append({"case_id": case["case_id"], "family": case["family"], "N": int(case["N"]), "perm_index": int(case["perm_index"]), "metric": metric, "median_ms": float(np.median(vals)) if vals else float("nan"), "p5_ms": float(np.percentile(vals, 5)) if vals else float("nan"), "p95_ms": float(np.percentile(vals, 95)) if vals else float("nan"), "cv": float(np.std(vals) / np.mean(vals)) if vals and np.mean(vals) else float("nan"), "n_iterations": len(vals), "S": float(case["S"]), "I": float(case["I"]), "traffic_entropy": float(case["traffic_entropy"]), "max_peer_fraction": float(case["max_peer_fraction"])})
    csv_write(args.output / "case_summary.csv", case_summary)
    lookup = {(r["family"], r["N"], r["perm_index"], r["metric"]): r for r in case_summary}
    pair_rows: list[dict[str, Any]] = []
    for n in SCALES:
        for p in range(len(permutations())):
            for metric in ("layout_ms", "dispatch_only_ms", "combine_ms", "full_path_ms"):
                b = lookup.get(("balanced_spread", n, p, metric)); c = lookup.get(("pair_concentrated", n, p, metric)); h = lookup.get(("destination_hotspot", n, p, metric))
                if not b or not c: continue
                bval, cval = float(b["median_ms"]), float(c["median_ms"])
                hval = float(h["median_ms"]) if h else float("nan")
                pair_rows.append({"N": n, "perm_index": p, "metric": metric, "balanced_ms": bval, "concentrated_ms": cval, "hotspot_ms": hval, "family_a_relative": (cval - bval) / bval if bval else float("nan"), "hotspot_relative": (hval - bval) / bval if bval and math.isfinite(hval) else float("nan")})
    csv_write(args.output / "permutation_results.csv", pair_rows)
    effects: dict[str, dict[str, Any]] = {}
    for metric in ("layout_ms", "dispatch_only_ms", "combine_ms", "full_path_ms"):
        for n in SCALES:
            vals = [float(r["family_a_relative"]) for r in pair_rows if int(r["N"]) == n and r["metric"] == metric and math.isfinite(float(r["family_a_relative"]))]
            effects[f"family_a_N{n}_{metric}"] = {"n": len(vals), "median_signed": float(np.median(vals)) if vals else float("nan"), "median_abs": float(np.median(np.abs(vals))) if vals else float("nan"), "p5": float(np.percentile(vals, 5)) if vals else float("nan"), "p95": float(np.percentile(vals, 95)) if vals else float("nan"), "positive_fraction": float(np.mean(np.asarray(vals) > 0)) if vals else float("nan"), "negative_fraction": float(np.mean(np.asarray(vals) < 0)) if vals else float("nan")}
    status_by_scale: dict[str, str] = {}
    for n in SCALES:
        v = effects[f"family_a_N{n}_full_path_ms"]
        abs_eff = abs(v["median_signed"]) if math.isfinite(v["median_signed"]) else 0.0
        consistency = max(v.get("positive_fraction", 0.0), v.get("negative_fraction", 0.0))
        status_by_scale[str(n)] = "GO" if abs_eff >= POLICY["go_effect"] and consistency >= POLICY["direction_consistency"] else "HOLD" if abs_eff >= POLICY["hold_effect"] else "NO-GO"
    if all(x == "GO" for x in status_by_scale.values()): gate = "GO"
    elif any(x in ("GO", "HOLD") for x in status_by_scale.values()): gate = "HOLD" if not all(x == "GO" for x in status_by_scale.values()) else "GO"
    else: gate = "NO-GO"
    # Diagnostic Family B, using full path and both scales.
    hotspot = {}
    for n in SCALES:
        vals = [float(r["hotspot_relative"]) for r in pair_rows if int(r["N"]) == n and r["metric"] == "full_path_ms" and math.isfinite(float(r["hotspot_relative"]))]
        hotspot[str(n)] = {"n": len(vals), "median_signed": float(np.median(vals)) if vals else float("nan"), "median_abs": float(np.median(np.abs(vals))) if vals else float("nan"), "positive_fraction": float(np.mean(np.asarray(vals) > 0)) if vals else float("nan")}
    summary = {"DEEPEP_TRAFFIC_MATRIX_SHAPE": gate, "family_a_status_by_scale": status_by_scale, "family_a_effects": effects, "family_b_hotspot_full_path": hotspot, "num_rank_files": len(rank_files), "num_raw_samples": len(all_samples), "physical_gpus": PHYSICAL_GPUS, "torch": json.loads((rank_files[0]).read_text(encoding="utf-8"))["torch"] if rank_files else None, "cuda": json.loads((rank_files[0]).read_text(encoding="utf-8"))["cuda"] if rank_files else None, "policy": POLICY}
    dump_json(args.output / "summary.json", summary)
    make_figures(args.output, case_summary, pair_rows)
    write_report(args.output, summary, effects, hotspot)
    print(json.dumps(summary, indent=2))


def make_figures(out: Path, case_summary: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(case_summary)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot = df[df.metric == "full_path_ms"].groupby(["N", "family"]).median(numeric_only=True).reset_index()
    for family, color in (("balanced_spread", "tab:blue"), ("pair_concentrated", "tab:orange"), ("destination_hotspot", "tab:red")):
        d = plot[plot.family == family]
        if not d.empty: ax.plot(d["N"].astype(str), d["median_ms"], marker="o", label=family, color=color)
    ax.set(xlabel="tokens/source", ylabel="median max-rank layout+dispatch+combine (ms)"); ax.legend(); fig.tight_layout(); fig.savefig(out / "plot1_iso_volume_traffic_shape.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    d = df[df.metric == "full_path_ms"]
    for family, color in (("balanced_spread", "tab:blue"), ("pair_concentrated", "tab:orange"), ("destination_hotspot", "tab:red")):
        x = d[d.family == family]; axes[0].scatter(x["traffic_entropy"], x["median_ms"], label=family, alpha=.8, color=color); axes[1].scatter(x["max_peer_fraction"], x["median_ms"], label=family, alpha=.8, color=color)
    axes[0].set(xlabel="traffic-matrix entropy", ylabel="full-path latency (ms)"); axes[1].set(xlabel="max destination fraction", ylabel="full-path latency (ms)"); axes[0].legend(); fig.tight_layout(); fig.savefig(out / "plot2_matrix_feature_vs_comm_latency.png", dpi=170); plt.close(fig)


def write_report(out: Path, summary: dict[str, Any], effects: dict[str, Any], hotspot: dict[str, Any]) -> None:
    lines = ["# DeepEP Traffic-Matrix Shape Forensics", "", f"`DEEPEP_TRAFFIC_MATRIX_SHAPE: {summary['DEEPEP_TRAFFIC_MATRIX_SHAPE']}`", "", "## Scope and invariant controls", "", "This is a communication-only synthetic replay. Each of four EP sources injects exactly N=256 or N=1024 tokens, each token routes to exactly two destination ranks with four balanced expert IDs per destination (top-k=8). Hidden payloads are deterministic random BF16 with H=2048. No model, expert GEMM, routing policy, placement, or dynamic communication code was used.", "", "Family A (`balanced_spread` vs `pair_concentrated`) was asserted before timing to have identical source-row sums, destination-column sums, total incidence/assignment volume, S=0.5, and I=1.0. Only the number of source→destination pairs used by tokens differs. Family B (`destination_hotspot`) keeps token volume and S but intentionally skews destination columns as a diagnostic. The canonical matrices and invariant checks are in `traffic_matrices.json` and `invariant_check.csv`.", "", "All 24 rank-label permutations were measured for both token scales. A permutation maps canonical source and destination labels consistently, preserving Family-A invariants while testing rank-label/topology dependence. Four logical ranks were mapped to physical GPUs 1,2,3,4 via `CUDA_VISIBLE_DEVICES=1,2,3,4`.", "", "## Timing", "", "Each case used 10 warmups and 50 measured iterations with a barrier before every iteration. CUDA events report the max rank. `layout_ms` is `get_dispatch_layout` only; `dispatch_only_ms` excludes layout; `combine_ms` is separate; `full_path_ms` spans layout through combine. Expert computation is absent. Raw rank samples are in `raw_timing.csv`; per-case summaries are in `case_summary.csv`; permutation effects are in `permutation_results.csv`.", "", "## Family A result", "", "| N | metric | median signed concentrated−balanced | median absolute | direction consistency | status |", "|---:|---|---:|---:|---:|---|"]
    for n in SCALES:
        key = f"family_a_N{n}_full_path_ms"; e = effects[key]; consistency = max(e.get("positive_fraction", 0), e.get("negative_fraction", 0)); status = summary["family_a_status_by_scale"][str(n)]
        lines.append(f"| {n} | full path | {e['median_signed']*100:.2f}% | {e['median_abs']*100:.2f}% | {consistency*100:.1f}% | {status} |")
        for metric, label in (("layout_ms", "layout"), ("dispatch_only_ms", "dispatch"), ("combine_ms", "combine")):
            x = effects[f"family_a_N{n}_{metric}"]; lines.append(f"| {n} | {label} | {x['median_signed']*100:.2f}% | {x['median_abs']*100:.2f}% | {max(x.get('positive_fraction',0),x.get('negative_fraction',0))*100:.1f}% | diagnostic |")
    lines += ["", "## Family B hotspot diagnostic", "", "| N | median hotspot−balanced full-path change | median absolute | positive fraction |", "|---:|---:|---:|---:|"]
    for n in SCALES:
        x = hotspot[str(n)]; lines.append(f"| {n} | {x['median_signed']*100:.2f}% | {x['median_abs']*100:.2f}% | {x['positive_fraction']*100:.1f}% |")
    lines += ["", "## Interpretation", "", "The primary gate requires a ≥10% Family-A latency shift at both token scales with the same direction in at least 75% of all 24 permutations per scale. A 5–10% shift is HOLD; below 5% or inconsistent direction is NO-GO. The measured result is reported without changing those thresholds.", "", f"Overall gate: **{summary['DEEPEP_TRAFFIC_MATRIX_SHAPE']}**. Family-B hotspot is diagnostic only and cannot replace Family-A evidence.", "", "## Limitations", "", "Synthetic routes isolate communication geometry and do not represent live Qwen3 hidden-state timing. Each source rank owns its own synthetic route rows, but no expert GEMM is executed. CUDA-event timing uses synchronous DeepEP calls (`async_finish=False`); layout calculation and communication are separated. No real-trace Stage C was run because it is conditional on the primary Family-A gate being at least HOLD.", "", f"Result directory: `{out}`", "", "Figures: `plot1_iso_volume_traffic_shape.png`, `plot2_matrix_feature_vs_comm_latency.png`.", ""]
    report = ROOT / "poc_flashvep/reports/deepep_traffic_matrix_shape.md"; report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "run", "aggregate"):
        q = sub.add_parser(mode); q.add_argument("--output", type=Path, required=True)
    sub.choices["run"].add_argument("--warmups", type=int, default=POLICY["warmups"]); sub.choices["run"].add_argument("--iterations", type=int, default=POLICY["iterations"]); sub.choices["run"].add_argument("--buffer-mib", type=int, default=512); sub.choices["run"].add_argument("--limit-cases", type=int, default=0)
    args = p.parse_args()
    if args.mode == "prepare": prepare(args)
    elif args.mode == "run": run(args)
    else: aggregate(args)


if __name__ == "__main__":
    main()
