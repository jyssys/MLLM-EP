#!/usr/bin/env python3
"""Offline complementary request-rebatching oracle.

This experiment is intentionally artifact-only: it loads already captured
Qwen3-VL routed-expert IDs and never imports/initializes CUDA, vLLM, or a model.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EP_RANKS = 4
EXPERTS = 128
LAYERS = 48
IMAGE_TOKEN_ID = 151655
SEED = 20260827
CONFIGS = [(8, 4), (12, 4), (12, 6), (16, 8), (24, 8), (24, 12)]
RANDOM_TRIALS = 1000


@dataclass(frozen=True)
class RequestTrace:
    pair_id: int
    request_id: str
    category: str
    bucket: str
    route_path: Path
    routes: np.ndarray  # [tokens, layer, top_k]
    token_ids: np.ndarray


def _json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _resolve_route(root: Path, rel: str) -> Path:
    p = root / rel
    if p.exists():
        return p
    # The manifest is portable but the route artifact may have been moved.
    p = root / Path(rel).name
    if p.exists():
        return p
    raise FileNotFoundError(f"route artifact not found: {rel}")


def load_traces(manifest_path: Path, route_root: Path) -> tuple[list[RequestTrace], list[RequestTrace], dict]:
    manifest = json.loads(manifest_path.read_text())
    visions: list[RequestTrace] = []
    texts: list[RequestTrace] = []
    for pair in manifest["pairs"]:
        for modality, out in (("vision", visions), ("text", texts)):
            item = pair[modality]
            path = _resolve_route(route_root, item["route_file"])
            with np.load(path, allow_pickle=False) as z:
                routes = np.asarray(z["routed_experts"], dtype=np.int64)
                token_ids = np.asarray(z["prompt_token_ids"], dtype=np.int64)
            if routes.ndim != 3 or routes.shape[1:] != (LAYERS, 8):
                raise ValueError(f"unexpected route shape for {path}: {routes.shape}")
            if routes.shape[0] != token_ids.shape[0]:
                raise ValueError(f"token/route length mismatch for {path}")
            trace = RequestTrace(
                pair_id=int(pair["pair_id"]),
                request_id=str(item["request_id"]),
                category=str(item["category"]),
                bucket=str(pair["token_bucket"]),
                route_path=path,
                routes=routes,
                token_ids=token_ids,
            )
            out.append(trace)
    if len(visions) != 24 or len(texts) != 24:
        raise ValueError(f"expected 24 vision/text traces, got {len(visions)}/{len(texts)}")
    return visions, texts, manifest


def rank_mass(routes: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return [layer, rank] assignment counts for selected token positions."""
    if len(positions) == 0:
        return np.zeros((LAYERS, EP_RANKS), dtype=np.int64)
    ids = routes[positions].reshape(len(positions), LAYERS, 8)
    # Linear expert placement is the validated EP4 placement in prior artifacts.
    ranks = ids // (EXPERTS // EP_RANKS)
    out = np.zeros((LAYERS, EP_RANKS), dtype=np.int64)
    for layer in range(LAYERS):
        out[layer] = np.bincount(ranks[:, layer].reshape(-1), minlength=EP_RANKS)
    return out


def build_footprints(visions: list[RequestTrace], texts: list[RequestTrace]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    all_integrity: list[dict] = []
    # The schedule objective is defined on the 24 real-image requests. Text
    # controls are retained for the required modality characterization.
    for trace, modality in [(t, "vision") for t in visions] + [(t, "text") for t in texts]:
        if modality == "vision":
            vpos = np.flatnonzero(trace.token_ids == IMAGE_TOKEN_ID)
        else:
            vpos = np.empty(0, dtype=np.int64)
        tpos = np.setdiff1d(np.arange(len(trace.token_ids)), vpos, assume_unique=True)
        vm = rank_mass(trace.routes, vpos)
        tm = rank_mass(trace.routes, tpos)
        rm = vm + tm
        # Check the per-layer assignment invariant before writing any result.
        full = rank_mass(trace.routes, np.arange(len(trace.token_ids)))
        if not np.array_equal(rm, full):
            raise AssertionError(f"R != V + T for {trace.request_id}")
        all_integrity.append(
            {
                "request_id": trace.request_id,
                "modality": modality,
                "tokens": int(len(trace.token_ids)),
                "vision_tokens": int(len(vpos)),
                "text_tokens": int(len(tpos)),
                "assignment_invariant": True,
                "image_span_contiguous": bool(
                    len(vpos) == 0 or np.array_equal(vpos, np.arange(vpos[0], vpos[-1] + 1))
                ),
            }
        )
        for layer in range(LAYERS):
            r, v, t = rm[layer], vm[layer], tm[layer]
            rows.append(
                {
                    "record_type": "request_layer",
                    "pair_id": trace.pair_id,
                    "request_id": trace.request_id,
                    "category": trace.category,
                    "token_bucket": trace.bucket,
                    "modality": modality,
                    "layer": layer,
                    "tokens": len(trace.token_ids),
                    "vision_tokens": len(vpos),
                    "text_tokens": len(tpos),
                    "total_assignments": int(r.sum()),
                    "vision_assignments": int(v.sum()),
                    "text_assignments": int(t.sum()),
                    "vision_fraction": float(v.sum() / r.sum()) if r.sum() else 0.0,
                    "R_R0": int(r[0]), "R_R1": int(r[1]), "R_R2": int(r[2]), "R_R3": int(r[3]),
                    "V_R0": int(v[0]), "V_R1": int(v[1]), "V_R2": int(v[2]), "V_R3": int(v[3]),
                    "T_R0": int(t[0]), "T_R1": int(t[1]), "T_R2": int(t[2]), "T_R3": int(t[3]),
                    "active_ranks_R": int(np.count_nonzero(r)),
                    "critical_rank_R": int(np.argmax(r)),
                    "critical_rank_V": int(np.argmax(v)) if v.sum() else -1,
                    "critical_rank_T": int(np.argmax(t)) if t.sum() else -1,
                    "critical_V_eq_R": bool(v.sum() and np.argmax(v) == np.argmax(r)),
                    "critical_T_eq_R": bool(t.sum() and np.argmax(t) == np.argmax(r)),
                }
            )
    return pd.DataFrame(rows), {"integrity": all_integrity}


def _groups_from_perm(perm: Sequence[int], be: int) -> list[tuple[int, ...]]:
    return [tuple(int(x) for x in perm[i : i + be]) for i in range(0, len(perm), be)]


def _partition_count(n: int, be: int) -> int:
    g = n // be
    return math.factorial(n) // (math.factorial(be) ** g * math.factorial(g))


def exact_partitions(n: int, be: int) -> Iterable[list[tuple[int, ...]]]:
    """Unique equal-size partitions; practical for n<=16 in this PoC."""
    items = tuple(range(n))

    def rec(rem: tuple[int, ...], groups: list[tuple[int, ...]]):
        if not rem:
            yield list(groups)
            return
        first = rem[0]
        for rest in itertools.combinations(rem[1:], be - 1):
            chosen = (first,) + rest
            chosen_set = set(chosen)
            nxt = tuple(x for x in rem if x not in chosen_set)
            yield from rec(nxt, groups + [chosen])

    yield from rec(items, [])


def group_stats(groups: Sequence[Sequence[int]], layer_vecs: np.ndarray) -> dict:
    """Compute layer cost/straggler metrics for request vectors [N, 4]."""
    rank_loads = np.asarray([layer_vecs[list(g)].sum(axis=0) for g in groups], dtype=np.int64)
    maxes = rank_loads.max(axis=1)
    means = rank_loads.mean(axis=1)
    cvs = rank_loads.std(axis=1) / np.maximum(means, 1e-12)
    return {
        "cost": int(maxes.sum()),
        "max_rank_load": int(maxes.sum()),
        "mean_rank_load": float(means.sum()),
        "cv_mean": float(cvs.mean()),
        "critical_rank_mode": int(np.bincount(rank_loads.argmax(axis=1), minlength=EP_RANKS).argmax()),
        "critical_set_3pct_mean": float(np.mean([np.sum(x >= 0.97 * x.max()) for x in rank_loads])),
        "critical_set_5pct_mean": float(np.mean([np.sum(x >= 0.95 * x.max()) for x in rank_loads])),
        "critical_set_10pct_mean": float(np.mean([np.sum(x >= 0.90 * x.max()) for x in rank_loads])),
    }


def objective(groups: Sequence[Sequence[int]], vectors: np.ndarray) -> float:
    # vectors [requests, layers, ranks] or [requests, ranks] after reduction.
    if vectors.ndim == 3:
        return float(sum(group_stats(g, vectors[:, l, :])['cost'] for l in range(vectors.shape[1])))
    return float(sum(np.asarray(vectors[list(g)]).sum(axis=0).max() for g in groups))


def optimize_partition(vectors: np.ndarray, be: int) -> tuple[list[tuple[int, ...]], str, int]:
    n = vectors.shape[0]
    count = _partition_count(n, be)
    if count <= 10000:
        best_groups = None
        best = float("inf")
        for groups in exact_partitions(n, be):
            value = objective(groups, vectors)
            if value < best - 1e-9:
                best, best_groups = value, groups
        assert best_groups is not None
        return best_groups, "exact", count
    # Deterministic greedy construction plus exhaustive pair swaps.  The
    # request order is fixed; ties use lexicographic group order.
    if vectors.ndim == 3:
        score_vec = vectors.sum(axis=1)
    else:
        score_vec = vectors
    order = sorted(range(n), key=lambda i: (-float(score_vec[i].max()), -float(score_vec[i].sum()), i))
    groups: list[list[int]] = [[] for _ in range(n // be)]
    for i in order:
        candidates = [g for g in range(len(groups)) if len(groups[g]) < be]
        values = []
        for g in candidates:
            trial = [list(x) for x in groups]
            trial[g] = trial[g] + [i]
            values.append((objective(trial, vectors), g))
        groups[min(values)[1]].append(i)
    current = [tuple(sorted(g)) for g in groups]
    current_value = objective(current, vectors)
    improved = True
    while improved:
        improved = False
        for a in range(len(current)):
            for b in range(a + 1, len(current)):
                for ia in range(be):
                    for ib in range(be):
                        trial = [list(g) for g in current]
                        trial[a][ia], trial[b][ib] = trial[b][ib], trial[a][ia]
                        trial_t = [tuple(sorted(g)) for g in trial]
                        value = objective(trial_t, vectors)
                        if value < current_value - 1e-9:
                            current, current_value = trial_t, value
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return current, "greedy_swap", count


def request_vectors(foot: pd.DataFrame, requests: list[str]) -> np.ndarray:
    sub = foot[(foot.request_id.isin(requests)) & (foot.record_type == "request_layer")].sort_values(["request_id", "layer"])
    return sub[["R_R0", "R_R1", "R_R2", "R_R3"]].to_numpy().reshape(len(requests), LAYERS, EP_RANKS)


def windows_for(requests: list[str], bs: int) -> list[list[str]]:
    # Pre-registered policy: disjoint full windows in manifest order. Any
    # remainder is reported as uncovered rather than duplicated or dropped from
    # the source artifacts.
    return [requests[i : i + bs] for i in range(0, len(requests) - bs + 1, bs)]


def aggregate_grouping(
    vectors_by_req: dict[str, np.ndarray], requests: list[str], bs: int, be: int,
    method: str, seed: int | None = None,
) -> tuple[float, list[dict], list[list[tuple[int, ...]]], str]:
    all_layer = []
    grouping_records = []
    chosen: list[object] = []
    mode = ""
    rng = np.random.default_rng(seed) if seed is not None else None
    for wi, window in enumerate(windows_for(requests, bs)):
        arr = np.stack([vectors_by_req[x] for x in window])
        if method == "fifo":
            groups = [tuple(range(i, i + be)) for i in range(0, bs, be)]
            mode = "fixed_order"
        elif method == "random":
            assert rng is not None
            perm = rng.permutation(bs)
            groups = _groups_from_perm(perm, be)
            mode = "random_permutation"
        elif method == "oracle_l":
            # Layer-local oracle: the request partition is recomputed from the
            # exact rank footprint for each decoder layer.
            groups = None
            mode = "layer_local"
            chosen_layer = []
            for layer in range(LAYERS):
                g_l, m_l, _ = optimize_partition(arr[:, layer, :], be)
                chosen_layer.append(g_l)
            chosen.append(chosen_layer)
            for layer in range(LAYERS):
                s = group_stats(chosen_layer[layer], arr[:, layer, :])
                s.update({"window_id": wi, "layer": layer, "group_count": len(chosen_layer[layer]), "method": method})
                grouping_records.append(s)
                all_layer.append(s["cost"])
            continue
        elif method == "oracle_f":
            groups, mode, _ = optimize_partition(arr.sum(axis=1), be)
        else:
            raise ValueError(method)
        chosen.append(groups)
        for layer in range(LAYERS):
            s = group_stats(groups, arr[:, layer, :])
            s.update({"window_id": wi, "layer": layer, "group_count": len(groups), "method": method})
            grouping_records.append(s)
            all_layer.append(s["cost"])
    return float(sum(all_layer)), grouping_records, chosen, mode


def summarize_grouping(
    foot: pd.DataFrame, requests: list[str], out: Path, random_trials: int = RANDOM_TRIALS
) -> tuple[pd.DataFrame, dict]:
    vectors_by_req = {
        rid: request_vectors(foot, [rid])[0] for rid in requests
    }
    rows: list[dict] = []
    detail: list[dict] = []
    policy = []
    for ci, (bs, be) in enumerate(CONFIGS):
        windows = windows_for(requests, bs)
        covered = sum(len(w) for w in windows)
        # FIFO, layer-local, and fixed grouping are one deterministic trial.
        deterministic = {}
        for method in ("fifo", "oracle_l", "oracle_f"):
            total, records, chosen, optimizer = aggregate_grouping(vectors_by_req, requests, bs, be, method)
            deterministic[method] = (total, records, chosen, optimizer)
            vals = np.asarray([x["cost"] for x in records], dtype=float)
            rows.append({
                "B_s": bs, "B_e": be, "method": method, "seed": -1,
                "window_count": len(windows), "covered_requests": covered, "uncovered_requests": len(requests) - covered,
                "window_cost_sum": total, "layer_cost_mean": float(vals.mean()),
                "layer_cost_p95": float(np.percentile(vals, 95)),
                "fifo_speedup": float(deterministic["fifo"][0] / total),
                "optimizer": optimizer,
            })
            for r in records:
                detail.append({"B_s": bs, "B_e": be, "seed": -1, **r})
        # Random baseline uses >=1000 fixed-seed partitions. Store one row per
        # seed to preserve the distribution, while detail stores only aggregate
        # trial costs (not 48*windows redundant rows).
        random_totals = []
        for trial in range(random_trials):
            seed = SEED + ci * 100000 + trial
            total, records, _, _ = aggregate_grouping(vectors_by_req, requests, bs, be, "random", seed)
            random_totals.append(total)
            vals = np.asarray([x["cost"] for x in records], dtype=float)
            rows.append({
                "B_s": bs, "B_e": be, "method": "random", "seed": seed,
                "window_count": len(windows), "covered_requests": covered, "uncovered_requests": len(requests) - covered,
                "window_cost_sum": total, "layer_cost_mean": float(vals.mean()),
                "layer_cost_p95": float(np.percentile(vals, 95)),
                "fifo_speedup": float(deterministic["fifo"][0] / total), "optimizer": "random_permutation",
            })
        arr = np.asarray(random_totals)
        policy.append({
            "B_s": bs, "B_e": be, "window_count": len(windows), "covered_requests": covered,
            "random_trials": random_trials, "random_seed_base": SEED + ci * 100000,
            "random_median_cost": float(np.median(arr)), "random_p5_cost": float(np.percentile(arr, 5)),
            "random_best_cost": float(arr.min()),
            "fifo_cost": deterministic["fifo"][0], "oracle_l_cost": deterministic["oracle_l"][0],
            "oracle_f_cost": deterministic["oracle_f"][0],
            "oracle_l_speedup": float(deterministic["fifo"][0] / deterministic["oracle_l"][0]),
            "oracle_f_speedup": float(deterministic["fifo"][0] / deterministic["oracle_f"][0]),
            "random_median_speedup": float(deterministic["fifo"][0] / np.median(arr)),
            "random_p5_speedup": float(deterministic["fifo"][0] / np.percentile(arr, 5)),
            "random_best_speedup": float(deterministic["fifo"][0] / arr.min()),
        })
    grouping = pd.DataFrame(rows)
    grouping.to_csv(out / "grouping_results.csv", index=False)
    pd.DataFrame(detail).to_csv(out / "grouping_layer_detail.csv", index=False)
    _json(out / "grouping_policy.json", {
        "configs": [list(x) for x in CONFIGS], "random_trials": random_trials, "seed": SEED,
        "window_policy": "disjoint full windows in manifest order; remainder is reported uncovered",
        "oracle_l": "exact equal-size partition when partition count <=10000, otherwise deterministic greedy+pair-swap, independently per layer",
        "oracle_f": "same optimizer on 48-layer aggregate rank footprint, fixed per window across layers",
        "objective": "sum over layers/windows of sum over groups max_r(sum_i_in_group R_i,l[r])",
        "request_set": "24 vision requests; text controls are diagnostic only",
    })
    return grouping, {"configs": policy}


def modality_summary(foot: pd.DataFrame, grouping_detail: pd.DataFrame, visions: list[RequestTrace], texts: list[RequestTrace], out: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for _, x in foot.iterrows():
        if x.record_type != "request_layer":
            continue
        rows.append({
            "record_type": "request_layer", "pair_id": x.pair_id, "request_id": x.request_id,
            "category": x.category, "token_bucket": x.token_bucket, "modality": x.modality, "layer": x.layer,
            "vision_fraction": x.vision_fraction, "V_critical_rank": x.critical_rank_V,
            "T_critical_rank": x.critical_rank_T, "R_critical_rank": x.critical_rank_R,
            "V_equals_R": x.critical_V_eq_R, "T_equals_R": x.critical_T_eq_R,
            "R_total": x.total_assignments, "V_total": x.vision_assignments, "T_total": x.text_assignments,
        })
    # Complementarity modality decomposition for the total-R Oracle-L grouping.
    reqs = [t.request_id for t in visions]
    vec_v = {}
    vec_t = {}
    vec_r = {}
    for rid in reqs:
        q = foot[(foot.request_id == rid) & (foot.record_type == "request_layer")].sort_values("layer")
        vec_v[rid] = q[["V_R0", "V_R1", "V_R2", "V_R3"]].to_numpy()
        vec_t[rid] = q[["T_R0", "T_R1", "T_R2", "T_R3"]].to_numpy()
        vec_r[rid] = q[["R_R0", "R_R1", "R_R2", "R_R3"]].to_numpy()
    for bs, be in CONFIGS:
        windows = windows_for(reqs, bs)
        for wi, window in enumerate(windows):
            arr_r = np.stack([vec_r[r] for r in window])
            for layer in range(LAYERS):
                # Reuse the total-R layer-local oracle definition, rather than
                # optimizing V/T independently (the latter is only a diagnostic
                # comparison below).
                groups, opt, _ = optimize_partition(arr_r[:, layer, :], be)
                for source, store in (("total", vec_r), ("vision", vec_v), ("text", vec_t)):
                    rank_loads = np.asarray([np.stack([store[window[i]][layer] for i in g]).sum(axis=0) for g in groups])
                    source_total = np.stack([store[r][layer] for r in window]).sum()
                    total_assignments = arr_r[:, layer, :].sum()
                    rows.append({
                        "record_type": "oracle_l_group_layer", "pair_id": -1, "request_id": f"window_{wi}",
                        "category": "all", "token_bucket": "all", "modality": source, "layer": layer,
                        "B_s": bs, "B_e": be, "window_id": wi, "optimizer": opt,
                        "group_cost": float(rank_loads.max(axis=1).sum()),
                        "source_assignments": int(source_total),
                        "rank_load_mean": float(rank_loads.mean()), "rank_load_cv_mean": float(
                            np.mean(rank_loads.std(axis=1) / np.maximum(rank_loads.mean(axis=1), 1e-12))
                        ),
                        "critical_rank": int(rank_loads.sum(axis=0).argmax()),
                        "vision_fraction": float(np.stack([vec_v[r][layer] for r in window]).sum() / total_assignments) if total_assignments else 0.0,
                    })
                # Counterfactual diagnostic: optimize on V-only or T-only,
                # then evaluate the resulting groups on the unchanged total R.
                for grouping_source, grouping_store in (("total", vec_r), ("vision", vec_v), ("text", vec_t)):
                    source_arr = np.stack([grouping_store[r][layer] for r in window])
                    cf_groups, cf_opt, _ = optimize_partition(source_arr, be)
                    eval_loads = np.asarray([np.stack([vec_r[window[i]][layer] for i in g]).sum(axis=0) for g in cf_groups])
                    rows.append({
                        "record_type": "counterfactual_group_layer", "pair_id": -1, "request_id": f"window_{wi}",
                        "category": "all", "token_bucket": "all", "modality": "total", "layer": layer,
                        "B_s": bs, "B_e": be, "window_id": wi, "optimizer": cf_opt,
                        "grouping_source": grouping_source,
                        "group_cost": float(eval_loads.max(axis=1).sum()),
                        "source_assignments": int(arr_r[:, layer, :].sum()),
                        "rank_load_mean": float(eval_loads.mean()),
                        "rank_load_cv_mean": float(np.mean(eval_loads.std(axis=1) / np.maximum(eval_loads.mean(axis=1), 1e-12))),
                        "critical_rank": int(eval_loads.sum(axis=0).argmax()),
                        "vision_fraction": float(np.stack([vec_v[r][layer] for r in window]).sum() / total_assignments) if total_assignments else 0.0,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(out / "modality_characterization.csv", index=False)
    return df


def figures(foot: pd.DataFrame, grouping: pd.DataFrame, modality: pd.DataFrame, visions: list[RequestTrace], out: Path) -> list[str]:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    # Figure 1: deterministic FIFO, random distribution and both oracles.
    summary_rows = []
    for (bs, be), g in grouping.groupby(["B_s", "B_e"], sort=False):
        fifo = float(g.loc[g.method == "fifo", "fifo_speedup"].iloc[0])
        ol = float(g.loc[g.method == "oracle_l", "fifo_speedup"].iloc[0])
        of = float(g.loc[g.method == "oracle_f", "fifo_speedup"].iloc[0])
        rg = g[g.method == "random"]
        summary_rows.extend([
            (f"{bs}/{be}", "FIFO", fifo), (f"{bs}/{be}", "Random median", float(rg.fifo_speedup.median())),
            (f"{bs}/{be}", "Random p5", float(rg.fifo_speedup.quantile(.05))),
            (f"{bs}/{be}", "Oracle-L", ol), (f"{bs}/{be}", "Oracle-F", of),
        ])
    s = pd.DataFrame(summary_rows, columns=["config", "method", "speedup"])
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, sub in s.groupby("method"):
        ax.plot(sub.config, sub.speedup, marker="o", label=method)
    ax.axhline(1.15, color="tab:green", ls="--", lw=1, label="GO 1.15x")
    ax.axhline(1.07, color="tab:orange", ls=":", lw=1, label="HOLD 1.07x")
    ax.set_ylabel("FIFO window cost / method cost")
    ax.set_xlabel("(B_s, B_e)")
    ax.set_title("Complementary rebatching headroom (route-artifact oracle)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); p = fig_dir / "plot1_fifo_random_oracle_headroom.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))

    # Figure 2: pre-registered representative: config with largest Oracle-L
    # aggregate headroom, then layer with largest rank spread in that window.
    cfgs = []
    for (bs, be), g in grouping.groupby(["B_s", "B_e"], sort=False):
        ol = float(g.loc[g.method == "oracle_l", "fifo_speedup"].iloc[0]); cfgs.append((ol, bs, be))
    _, bs, be = max(cfgs)
    reqs = [t.request_id for t in visions]
    win = windows_for(reqs, bs)[0]
    sub = foot[foot.request_id.isin(win)].sort_values(["request_id", "layer"])
    pivot = sub.pivot(index="request_id", columns="layer", values="vision_fraction")
    # Use request x rank total footprint for an interpretable complementary example.
    q = foot[(foot.request_id.isin(win)) & (foot.layer == 0)].set_index("request_id").loc[win]
    mat = q[["R_R0", "R_R1", "R_R2", "R_R3"]].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(win))); ax.set_yticklabels(win, fontsize=7); ax.set_xticks(range(4)); ax.set_xticklabels(["R0", "R1", "R2", "R3"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]): ax.text(j, i, str(int(mat[i,j])), ha="center", va="center", color="white" if mat[i,j] < mat.max()*.55 else "black", fontsize=7)
    ax.set_title(f"Request rank footprints, layer 0; representative max Oracle-L config ({bs},{be})")
    fig.colorbar(im, ax=ax, label="routed assignments"); fig.tight_layout(); p = fig_dir / "plot2_request_rank_footprint_heatmap.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))

    # Figure 3: critical rank and source contribution.
    x = foot[foot.modality == "vision"]
    vals = [float((x.critical_rank_V == x.critical_rank_R).mean()), float((x.critical_rank_T == x.critical_rank_R).mean())]
    vfrac = float(x.vision_fraction.mean())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Vision vs Total", "Text vs Total"], vals, color=["#4c78a8", "#f58518"]); axes[0].set_ylim(0, 1); axes[0].set_ylabel("critical-rank agreement")
    axes[1].bar(["Vision", "Text"], [vfrac, 1-vfrac], color=["#4c78a8", "#f58518"]); axes[1].set_ylim(0,1); axes[1].set_ylabel("assignment fraction"); axes[1].set_title("Vision/Text contribution (vision requests)")
    fig.suptitle("Modality critical-rank characterization"); fig.tight_layout(); p = fig_dir / "plot3_modality_critical_rank.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(str(p))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/workload_manifest.json"))
    ap.add_argument("--route-root", type=Path, default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--random-trials", type=int, default=RANDOM_TRIALS)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    visions, texts, manifest = load_traces(args.manifest, args.route_root)
    foot, meta = build_footprints(visions, texts)
    foot.to_csv(args.output / "per_request_layer_rank_footprint.csv", index=False)
    _json(args.output / "capture_integrity.json", meta)
    shutil.copy2(args.manifest, args.output / "source_workload_manifest.json")
    grouping, gmeta = summarize_grouping(foot, [t.request_id for t in visions], args.output, args.random_trials)
    modality = modality_summary(foot, pd.DataFrame(), visions, texts, args.output)
    fig_paths = figures(foot, grouping, modality, visions, args.output)
    p = pd.DataFrame(gmeta["configs"])
    p.to_csv(args.output / "grouping_config_summary.csv", index=False)
    oracle_l = p.oracle_l_speedup.to_numpy(float)
    overall = {
        "source_manifest": str(args.manifest), "route_root": str(args.route_root),
        "model": manifest.get("model"), "trace_count": {"vision": len(visions), "text": len(texts)},
        "layers": LAYERS, "ep_ranks": EP_RANKS, "experts": EXPERTS, "top_k": 8,
        "expert_to_rank": "expert_id // 32 (validated linear EP4 placement)",
        "image_token_id": IMAGE_TOKEN_ID, "gpu_execution": False,
        "assignment_invariant_all_rows": bool(foot.total_assignments.eq(foot.vision_assignments + foot.text_assignments).all()),
        "vision_request_order": [t.request_id for t in visions],
        "oracle_l_speedups": {f"{a}/{b}": float(x) for (a,b), x in zip(CONFIGS, oracle_l)},
        "oracle_f_speedups": {f"{a}/{b}": float(x) for (a,b), x in zip(CONFIGS, p.oracle_f_speedup)},
        "best_oracle_l_speedup": float(oracle_l.max()), "median_oracle_l_speedup": float(np.median(oracle_l)),
        "best_config_oracle_l": list(CONFIGS[int(np.argmax(oracle_l))]),
        "gate": "GO if median Oracle-L >= 1.15x; HOLD if 1.07-1.15x; NO-GO if <1.07x or rare",
        "figures": fig_paths,
    }
    _json(args.output / "summary.json", overall)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
