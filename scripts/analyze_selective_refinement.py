#!/usr/bin/env python3
"""Analyze dense-emulated selective refinement and calibrated EP projections.

Evidence levels are intentionally kept separate:

* generation/quality and freeze drift are measured with the reference model;
* EP4/EP8 costs are EP2-calibrated simulations of the recorded routes;
* S1/S2 and perfect balance are substrate/oracle projections, never runtime claims.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.analyze_h1_straggler_deepdive import (
    BatchComputeModel, balance_histogram, load_discovery_trace, ordered_replicas,
    rank_times,
)
from virtual_ep.comm_model import CommunicationScenario


HIDDEN_BYTES = 2048 * 2
SUBSTRATES = ("s0", "s1", "s2")
EP_TARGETS = (4, 8)


def jaccard_last_axis(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Set Jaccard for small fixed-length expert arrays."""
    intersection = (left[..., :, None] == right[..., None, :]).any(axis=-1).sum(axis=-1)
    return intersection / np.maximum(2 * left.shape[-1] - intersection, 1)


def rank_jaccard(left: np.ndarray, right: np.ndarray, ep: int) -> np.ndarray:
    owner = 256 // ep
    a = left // owner
    b = right // owner
    pa = np.stack([(a == rank).any(axis=-1) for rank in range(ep)], axis=-1)
    pb = np.stack([(b == rank).any(axis=-1) for rank in range(ep)], axis=-1)
    return (pa & pb).sum(axis=-1) / np.maximum((pa | pb).sum(axis=-1), 1)


def read_generations(directory: Path) -> dict[int, dict]:
    result = {}
    # A long-tail request can be rebalanced onto an idle GPU without changing
    # its request ID, seed, policy, or trace schema.  Treat those shards exactly
    # like the original worker shards; otherwise the promoted 128-request
    # quality gate silently drops the rebalanced tail.
    for path in sorted(directory.glob("generations_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                result[int(row["sample_id"])] = row
    return result


def extend_reference_baseline(baseline: dict[int, dict], discovery_root: Path,
                              aggregate_trace: Path) -> dict[int, dict]:
    """Add official fixed-0.95 requests 32--127 without rerunning them."""
    result = dict(baseline)
    with np.load(aggregate_trace, allow_pickle=False) as z:
        request = z["request_id"]
        layer = z["layer_id"]
        masked = z["masked_current_block"]
        first_layer = int(layer[0])
        active_by_request = {
            rid: int(masked[(request == rid) & (layer == first_layer)].sum())
            for rid in range(128)
        }
    for pattern in ("generations_worker*.jsonl", "generations_rebalance*.jsonl"):
        for path in discovery_root.glob(pattern):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line); rid = int(row["sample_id"])
                if rid in result:
                    continue
                result[rid] = {
                    "sample_id": rid, "correct": bool(row["correct"]),
                    "parsed_answer": row["parsed_answer"], "output": row.get("output", ""),
                    "output_ids": None, "nfe": int(row["nfe"]),
                    "active_updates": active_by_request[rid],
                }
    return result


def load_npz(path: Path) -> SimpleNamespace:
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files if key != "metadata_json"}
        metadata = json.loads(str(source["metadata_json"]))
    return SimpleNamespace(arrays=arrays, metadata=metadata)


def top_overlap(conf_a: np.ndarray, conf_b: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(len(conf_a) * fraction)))
    a = set(np.argsort(conf_a)[-k:].tolist())
    b = set(np.argsort(conf_b)[-k:].tolist())
    return len(a & b) / k


def predictability(trace_dir: Path) -> dict:
    confidence_a, confidence_b, labels, request_ids, rank_percentile = [], [], [], [], []
    rank_corr, overlaps = [], {"25": [], "50": [], "75": []}
    expert_j, ep4_j, ep8_j, dominant4, dominant8 = [], [], [], [], []
    hidden = defaultdict(lambda: {"cosine": [], "relative_l2": []})
    pair_count = 0
    for path in sorted(trace_dir.glob("request_*.npz")):
        trace = load_npz(path)
        z = trace.arrays
        rid = int(trace.metadata["request_id"])
        if rid >= 16:
            continue
        for index in range(len(z["block_id"]) - 1):
            if (z["block_id"][index + 1] != z["block_id"][index]
                    or z["iteration_id"][index + 1] != z["iteration_id"][index] + 1):
                continue
            positions = np.flatnonzero(z["masked"][index] & z["masked"][index + 1])
            if not len(positions):
                continue
            pair_count += 1
            ca = z["confidence"][index, positions]
            cb = z["confidence"][index + 1, positions]
            confidence_a.extend(ca.tolist()); confidence_b.extend(cb.tolist())
            labels.extend(z["transfer"][index + 1, positions].astype(int).tolist())
            request_ids.extend([rid] * len(positions))
            order = np.argsort(np.argsort(ca)) / max(len(ca) - 1, 1)
            rank_percentile.extend(order.tolist())
            if len(ca) > 1 and np.std(ca) and np.std(cb):
                rank_corr.append(float(spearmanr(ca, cb).statistic))
            for fraction, key in ((.25, "25"), (.5, "50"), (.75, "75")):
                overlaps[key].append(top_overlap(ca, cb, fraction))
            a = z["routes"][index, :, positions, :]
            b = z["routes"][index + 1, :, positions, :]
            expert_j.extend(jaccard_last_axis(a, b).reshape(-1).tolist())
            ep4_j.extend(rank_jaccard(a, b, 4).reshape(-1).tolist())
            ep8_j.extend(rank_jaccard(a, b, 8).reshape(-1).tolist())
            for ep, sink in ((4, dominant4), (8, dominant8)):
                owner = 256 // ep
                aa = a // owner; bb = b // owner
                for token in range(a.shape[1]):
                    va = np.bincount(aa[:, token].reshape(-1), minlength=ep)
                    vb = np.bincount(bb[:, token].reshape(-1), minlength=ep)
                    sink.append(int(np.argmax(va) == np.argmax(vb)))
        if "drift_boundary" in z:
            selected = z["drift_boundary"] == "adjacent_hidden"
            for layer in np.unique(z["drift_layer_id"][selected]):
                mask = selected & (z["drift_layer_id"] == layer)
                hidden[str(int(layer))]["cosine"].extend(z["drift_cosine"][mask].tolist())
                hidden[str(int(layer))]["relative_l2"].extend(
                    z["drift_relative_l2"][mask].tolist())

    ca = np.asarray(confidence_a); cb = np.asarray(confidence_b)
    y = np.asarray(labels); rid = np.asarray(request_ids); rankp = np.asarray(rank_percentile)
    train = rid < 8; held = rid >= 8
    def score_metrics(score):
        if held.sum() == 0 or len(np.unique(y[held])) < 2:
            return {"auroc": float("nan"), "average_precision": float("nan")}
        return {
            "auroc": float(roc_auc_score(y[held], score[held])),
            "average_precision": float(average_precision_score(y[held], score[held])),
            "positive_rate": float(y[held].mean()),
        }
    result = {
        "requests": int(len(set(request_ids))), "adjacent_pairs": pair_count,
        "token_pairs": int(len(ca)),
        "confidence_pearson": float(pearsonr(ca, cb).statistic),
        "confidence_spearman": float(spearmanr(ca, cb).statistic),
        "confidence_rank_spearman_mean": float(np.mean(rank_corr)),
        "top_overlap": {key: float(np.mean(value)) for key, value in overlaps.items()},
        "routing": {
            "expert_jaccard_p50": float(np.median(expert_j)),
            "expert_jaccard_mean": float(np.mean(expert_j)),
            "ep4_rank_jaccard_p50": float(np.median(ep4_j)),
            "ep4_rank_jaccard_mean": float(np.mean(ep4_j)),
            "ep8_rank_jaccard_p50": float(np.median(ep8_j)),
            "ep8_rank_jaccard_mean": float(np.mean(ep8_j)),
            "ep4_dominant_rank_stability": float(np.mean(dominant4)),
            "ep8_dominant_rank_stability": float(np.mean(dominant8)),
        },
        "commit_prediction_heldout_8_15": {
            "previous_confidence": score_metrics(ca),
            "previous_confidence_rank": score_metrics(rankp),
            "split": "request IDs 0-7 discovery, 8-15 held-out",
        },
        "conditional_commit_probability_top_confidence_quartile": float(
            y[ca >= np.quantile(ca, .75)].mean()
        ),
        "hidden": {},
    }
    for layer, metrics in hidden.items():
        result["hidden"][layer] = {
            "samples": len(metrics["cosine"]),
            "cosine_p50": float(np.median(metrics["cosine"])),
            "cosine_p10": float(np.percentile(metrics["cosine"], 10)),
            "relative_l2_p50": float(np.median(metrics["relative_l2"])),
            "relative_l2_p90": float(np.percentile(metrics["relative_l2"], 90)),
        }
    return result


def config_summary(directory: Path, baseline: dict[int, dict]) -> dict:
    rows = read_generations(directory)
    ids = sorted(set(rows) & set(baseline))
    if not ids:
        return {}
    first = rows[ids[0]]
    total_active = sum(rows[i]["active_updates"] for i in ids)
    base_active = sum(baseline[i]["active_updates"] for i in ids)
    total_masked = sum(rows[i]["masked_updates"] for i in ids)
    additional_wrong = sum(baseline[i]["correct"] and not rows[i]["correct"] for i in ids)
    corrected = sum(not baseline[i]["correct"] and rows[i]["correct"] for i in ids)
    discordant = additional_wrong + corrected
    freeze_ages = []
    for request_id in ids:
        path = directory / "traces" / f"request_{request_id:04d}.npz"
        if not path.exists():
            continue
        z = load_npz(path).arrays
        age = np.zeros(32, dtype=np.int16); previous_block = None
        for block, masked, active in zip(z["block_id"], z["masked"], z["active"]):
            if previous_block is None or block != previous_block:
                age.fill(0)
            frozen = masked & ~active
            age[frozen] += 1; age[active | ~masked] = 0
            freeze_ages.extend(age[frozen].tolist())
            previous_block = block
    return {
        "name": directory.name, "path": str(directory), "samples": len(ids),
        "sample_ids": ids, "policy": first["policy"], "semantics": first["semantics"],
        "accuracy": sum(rows[i]["correct"] for i in ids) / len(ids),
        "correct": sum(rows[i]["correct"] for i in ids),
        "baseline_correct": sum(baseline[i]["correct"] for i in ids),
        "additional_wrong": additional_wrong, "baseline_wrong_corrected": corrected,
        "paired_exact_p": (float(binomtest(additional_wrong, discordant, .5).pvalue)
                           if discordant else 1.0),
        "accuracy_delta_pp": 100 * (
            sum(rows[i]["correct"] for i in ids) - sum(baseline[i]["correct"] for i in ids)
        ) / len(ids),
        "parsed_answer_identity": sum(
            rows[i]["parsed_answer"] == baseline[i]["parsed_answer"] for i in ids
        ) / len(ids),
        "exact_sequence_identity": sum(
            (rows[i].get("output_ids") == baseline[i].get("output_ids")
             if baseline[i].get("output_ids") is not None
             else rows[i].get("output", "") == baseline[i].get("output", ""))
            for i in ids
        ) / len(ids),
        "nfe_mean": float(np.mean([rows[i]["nfe"] for i in ids])),
        "nfe_median": float(np.median([rows[i]["nfe"] for i in ids])),
        "baseline_nfe_mean": float(np.mean([baseline[i]["nfe"] for i in ids])),
        "baseline_nfe_median": float(np.median([baseline[i]["nfe"] for i in ids])),
        "nfe_delta_percent": 100 * (
            sum(rows[i]["nfe"] for i in ids) / sum(baseline[i]["nfe"] for i in ids) - 1
        ),
        "w_active": total_active, "baseline_w_active": base_active,
        "current_block_masked_updates": total_masked,
        "w_active_reduction_percent": 100 * (1 - total_active / base_active),
        "same_trajectory_active_reduction_percent": 100 * (1 - total_active / total_masked),
        "forced_updates": sum(rows[i]["forced_updates"] for i in ids),
        "freeze_age_mean": float(np.mean(freeze_ages)) if freeze_ages else 0.0,
        "freeze_age_p95": float(np.percentile(freeze_ages, 95)) if freeze_ages else 0.0,
        "remaining_masks": sum(rows[i]["remaining_masks"] for i in ids),
        "generated_blocks_mean": float(np.mean([rows[i]["blocks"] for i in ids])),
        "termination": dict((key, sum(rows[i]["termination"] == key for i in ids))
                            for key in {rows[i]["termination"] for i in ids}),
        "accepted_per_refinement": float(np.mean([rows[i]["accepted_mean"] for i in ids])),
    }


def drift_summary(directory: Path) -> dict:
    records = defaultdict(lambda: defaultdict(list))
    confidence_error = defaultdict(list)
    disagreement = defaultdict(list)
    expert_j, rank4_j, rank8_j = defaultdict(list), defaultdict(list), defaultdict(list)
    for path in sorted((directory / "traces").glob("request_*.npz")):
        z = load_npz(path).arrays
        if "drift_boundary" in z:
            for boundary in np.unique(z["drift_boundary"]):
                for age in np.unique(z["drift_freeze_age"][z["drift_boundary"] == boundary]):
                    mask = ((z["drift_boundary"] == boundary)
                            & (z["drift_freeze_age"] == age))
                    key = f"{boundary}:age{int(age)}"
                    records[key]["cosine"].extend(z["drift_cosine"][mask].tolist())
                    records[key]["relative_l2"].extend(z["drift_relative_l2"][mask].tolist())
        # Reconstruct staleness relative to last active refresh.
        last_route = np.full(z["routes"].shape[1:], -1, dtype=np.int16)
        ages = np.zeros(32, dtype=np.int16)
        for step in range(len(z["block_id"])):
            if step == 0 or z["block_id"][step] != z["block_id"][step - 1]:
                last_route.fill(-1); ages.fill(0)
            masked = z["masked"][step]; active = z["active"][step]
            frozen = masked & ~active
            for position in np.flatnonzero(frozen):
                age = int(ages[position] + 1)
                if last_route[0, position, 0] >= 0:
                    a = last_route[:, position]
                    b = z["routes"][step, :, position]
                    expert_j[age].extend(jaccard_last_axis(a, b).reshape(-1).tolist())
                    rank4_j[age].extend(rank_jaccard(a, b, 4).reshape(-1).tolist())
                    rank8_j[age].extend(rank_jaccard(a, b, 8).reshape(-1).tolist())
                value = z["confidence_cache_error"][step, position]
                if np.isfinite(value): confidence_error[age].append(float(value))
                disagreement[age].append(float(z["commit_disagreement"][step, position]))
                ages[position] = age
            refreshed = active | ~masked
            for position in np.flatnonzero(refreshed):
                last_route[:, position] = z["routes"][step, :, position]
                ages[position] = 0
    result = {"hidden_or_output": {}, "staleness": {}}
    for key, values in records.items():
        result["hidden_or_output"][key] = {
            "n": len(values["cosine"]), "cosine_p50": float(np.median(values["cosine"])),
            "cosine_p10": float(np.percentile(values["cosine"], 10)),
            "relative_l2_p50": float(np.median(values["relative_l2"])),
            "relative_l2_p90": float(np.percentile(values["relative_l2"], 90)),
        }
    for age in sorted(set(expert_j) | set(confidence_error)):
        result["staleness"][str(age)] = {
            "expert_jaccard_mean": float(np.mean(expert_j[age])) if expert_j[age] else None,
            "ep4_rank_jaccard_mean": float(np.mean(rank4_j[age])) if rank4_j[age] else None,
            "ep8_rank_jaccard_mean": float(np.mean(rank8_j[age])) if rank8_j[age] else None,
            "confidence_error_mean": float(np.mean(confidence_error[age])) if confidence_error[age] else None,
            "commit_disagreement": float(np.mean(disagreement[age])) if disagreement[age] else None,
            "samples": len(disagreement[age]),
        }
    return result


class CostProjector:
    def __init__(self, ep4_model: Path, ep8_model: Path, comm_model: Path,
                 discovery_trace: Path):
        self.models = {4: BatchComputeModel(ep4_model), 8: BatchComputeModel(ep8_model)}
        self.comm = CommunicationScenario.load(comm_model, "ep2_calibrated_base")
        discovery = load_discovery_trace(discovery_trace)
        arrays = discovery.arrays
        hist = arrays["expert_counts_by_class"].sum(axis=1)
        self.replica_maps = {}
        for ep in EP_TARGETS:
            times = rank_times(hist, ep, self.models[ep])
            # Only B1's global-static map is needed. Avoid constructing the
            # much more expensive per-block future/early H1 oracle maps.
            global_map = {}
            for layer in np.unique(arrays["layer_id"]):
                indices = np.flatnonzero(
                    (arrays["request_id"] < 64) & (arrays["layer_id"] == layer)
                ).tolist()
                global_map[int(layer)] = ordered_replicas(indices, hist, times, ep)
            self.replica_maps[ep] = global_map

    def project(self, directory: Path, allowed_requests: set[int] | None = None) -> dict:
        result = {}
        for ep in EP_TARGETS:
            result[f"ep{ep}"] = {}
            for substrate in SUBSTRATES:
                accum = defaultdict(float); vectors = []; stages = []; perfect = []
                replica_histograms = [[] for _ in range(ep)]
                communication_only = []
                paths = sorted((directory / "traces").glob("request_*.npz"))
                if allowed_requests is not None:
                    paths = [path for path in paths
                             if int(path.stem.split("_")[-1]) in allowed_requests]
                for path in paths:
                    trace = load_npz(path); z = trace.arrays
                    layer_ids = trace.metadata["routed_layer_ids"]
                    hist = z[f"hist_{substrate}"].reshape(-1, 256).astype(np.float64)
                    u = z[f"u_{substrate}_ep{ep}"].reshape(-1, ep, ep).astype(np.float64)
                    time_vectors = np.column_stack([
                        self.models[ep].predict_batch(
                            hist[:, rank * (256 // ep):(rank + 1) * (256 // ep)]
                        ) for rank in range(ep)
                    ])
                    vectors.append(time_vectors)
                    for ordinal, (h, unique, tv) in enumerate(zip(hist, u, time_vectors)):
                        layer = int(layer_ids[ordinal % len(layer_ids)])
                        remote = unique.copy(); np.fill_diagonal(remote, 0)
                        outgoing = remote.sum(axis=1) * HIDDEN_BYTES
                        incoming = remote.sum(axis=0) * HIDDEN_BYTES
                        dispatch = self.comm.dispatch.latency(outgoing, incoming)
                        combine = self.comm.combine.latency(incoming, outgoing)
                        expert = float(tv.max())
                        stage = dispatch + expert + combine
                        stages.append(stage)
                        communication_only.append(dispatch + combine)
                        accum["dispatch_ms"] += dispatch; accum["expert_ms"] += expert
                        accum["combine_ms"] += combine; accum["stage_ms"] += stage
                        accum["remote_bytes"] += outgoing.sum()
                        accum["fanout_sum"] += np.count_nonzero(remote, axis=1).mean()
                        accum["invocations"] += 1
                        # Existing global-static budget=8. Communication is held
                        # unchanged: a conservative workload-specific B1 estimate.
                        local, _moves = balance_histogram(
                            h, self.replica_maps[ep].get(layer, [])[:8], ep
                        )
                        for rank, rank_hist in enumerate(local):
                            replica_histograms[rank].append(rank_hist)
                        perfect.append(dispatch + float(tv.mean()) + combine)
                if not stages:
                    continue
                requests = len(paths)
                vec = np.concatenate(vectors)
                maximum = vec.max(axis=1); mean = vec.mean(axis=1)
                second = np.sort(vec, axis=1)[:, -2]
                accum["fanout_mean"] = accum.pop("fanout_sum") / accum["invocations"]
                accum["max_mean_mean"] = float(np.mean(maximum / np.maximum(mean, 1e-12)))
                accum["max_second_mean"] = float(np.mean(maximum / np.maximum(second, 1e-12)))
                accum["cv_mean"] = float(np.mean(vec.std(axis=1) / np.maximum(mean, 1e-12)))
                accum["wait_fraction_mean"] = float(np.mean(
                    (ep * maximum - vec.sum(axis=1)) / np.maximum(ep * maximum, 1e-12)
                ))
                padded_by_rank = []
                for rank in range(ep):
                    width = max(len(row) for row in replica_histograms[rank])
                    padded = np.zeros((len(replica_histograms[rank]), width), dtype=np.float64)
                    for ordinal, row in enumerate(replica_histograms[rank]):
                        padded[ordinal, :len(row)] = row
                    padded_by_rank.append(padded)
                replica_vectors = np.column_stack([
                    self.models[ep].predict_batch(padded_by_rank[rank])
                    for rank in range(ep)
                ])
                accum["global_static_replica_stage_ms"] = float(np.sum(
                    np.asarray(communication_only) + replica_vectors.max(axis=1)
                ))
                accum["perfect_balance_stage_ms"] = float(np.sum(perfect))
                replica_max = replica_vectors.max(axis=1)
                replica_mean = replica_vectors.mean(axis=1)
                replica_second = np.sort(replica_vectors, axis=1)[:, -2]
                accum["global_static_max_mean_mean"] = float(np.mean(
                    replica_max / np.maximum(replica_mean, 1e-12)
                ))
                accum["global_static_max_second_mean"] = float(np.mean(
                    replica_max / np.maximum(replica_second, 1e-12)
                ))
                accum["global_static_cv_mean"] = float(np.mean(
                    replica_vectors.std(axis=1) / np.maximum(replica_mean, 1e-12)
                ))
                accum["global_static_wait_fraction_mean"] = float(np.mean(
                    (ep * replica_max - replica_vectors.sum(axis=1))
                    / np.maximum(ep * replica_max, 1e-12)
                ))
                accum["perfect_balance_max_mean"] = 1.0
                accum["perfect_balance_max_second"] = 1.0
                accum["perfect_balance_cv"] = 0.0
                accum["perfect_balance_wait_fraction"] = 0.0
                accum["requests"] = requests
                for key in ("dispatch_ms", "expert_ms", "combine_ms", "stage_ms",
                            "global_static_replica_stage_ms", "perfect_balance_stage_ms"):
                    accum[f"{key}_per_request"] = accum[key] / requests
                result[f"ep{ep}"][substrate] = dict(accum)
        return result


def oracle_summary(trace_dir: Path) -> dict:
    stats = defaultdict(float)
    generic_maxmean = {4: [], 8: []}; oracle_maxmean = {4: [], 8: []}
    for path in sorted(trace_dir.glob("request_*.npz")):
        z = load_npz(path).arrays
        for step in range(len(z["block_id"]) - 1):
            if (z["block_id"][step + 1] != z["block_id"][step]
                    or z["iteration_id"][step + 1] != z["iteration_id"][step] + 1):
                continue
            masked = np.flatnonzero(z["masked"][step])
            if not len(masked): continue
            commits = np.flatnonzero(z["transfer"][step + 1] & z["masked"][step])
            budget = max(1, int(np.ceil(.75 * len(masked))))
            stats["masked"] += len(masked); stats["o1"] += len(commits)
            stats["o2"] += int(np.ceil(.9 * len(commits)))
            for ep in EP_TARGETS:
                owner = z["routes"][step] // (256 // ep)
                token_load = np.stack([
                    np.bincount(owner[:, pos].reshape(-1), minlength=ep) for pos in range(32)
                ])
                generic = masked[np.argsort(z["confidence"][step, masked])[-budget:]]
                chosen = list(map(int, commits[:budget]))
                remaining = [int(x) for x in masked if x not in chosen]
                loads = token_load[chosen].sum(axis=0) if chosen else np.zeros(ep)
                while len(chosen) < budget and remaining:
                    candidate = min(
                        remaining,
                        key=lambda pos: ((loads + token_load[pos]).max(),
                                         (loads + token_load[pos]).std(), -z["confidence"][step, pos]),
                    )
                    chosen.append(candidate); loads += token_load[candidate]; remaining.remove(candidate)
                gv = token_load[generic].sum(axis=0)
                ov = token_load[chosen].sum(axis=0)
                generic_maxmean[ep].append(gv.max() / max(gv.mean(), 1e-12))
                oracle_maxmean[ep].append(ov.max() / max(ov.mean(), 1e-12))
    return {
        "o1_next_commit_active_fraction": stats["o1"] / stats["masked"],
        "o2_90pct_progress_active_fraction": stats["o2"] / stats["masked"],
        "o3": {f"ep{ep}": {
            "generic_confidence_max_mean": float(np.mean(generic_maxmean[ep])),
            "progress_preserving_balanced_max_mean": float(np.mean(oracle_maxmean[ep])),
            "relative_reduction": 1 - np.mean(oracle_maxmean[ep]) / np.mean(generic_maxmean[ep]),
        } for ep in EP_TARGETS},
        "future_information": True,
    }


def plot_outputs(summary: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    pred = summary["predictability"]
    configs = [row for row in summary["configs"] if row]
    def save(name, x, y, xlabel, ylabel, title):
        plt.figure(figsize=(6, 4)); plt.plot(x, y, "o-"); plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.title(title); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(output / name, dpi=160); plt.close()
    save("01_confidence_persistence.png", [25, 50, 75],
         [pred["top_overlap"][str(x)] for x in (25, 50, 75)], "top budget (%)", "overlap", "Adjacent confidence top-budget overlap")
    save("02_route_jaccard.png", [0, 1, 2], [pred["routing"]["expert_jaccard_mean"], pred["routing"]["ep4_rank_jaccard_mean"], pred["routing"]["ep8_rank_jaccard_mean"]], "expert / EP4 / EP8", "Jaccard", "Adjacent routing persistence")
    drift = summary["drift"]
    ages = sorted({int(age) for item in drift.values() for age in item.get("staleness", {})})
    for number, metric, label in ((3, "cosine", "hidden/output cosine"), (4, "confidence_error_mean", "confidence error"), (5, "commit_disagreement", "commit disagreement")):
        values=[]
        for age in ages:
            vals=[]
            for item in drift.values():
                if metric == "cosine":
                    vals += [v["cosine_p50"] for k,v in item.get("hidden_or_output",{}).items() if k.endswith(f"age{age}")]
                elif str(age) in item.get("staleness",{}) and item["staleness"][str(age)][metric] is not None:
                    vals.append(item["staleness"][str(age)][metric])
            values.append(np.mean(vals) if vals else np.nan)
        save(f"{number:02d}_freeze_age_{metric}.png", ages or [0], values or [np.nan], "freeze age", label, f"Freeze age vs {label}")
    selected = [row for row in configs if row["semantics"] != "none"]
    ratios = [100 * row["policy"]["active_ratio"] for row in selected]
    save("06_active_ratio_accuracy.png", ratios, [100*r["accuracy"] for r in selected], "active ratio (%)", "accuracy (%)", "Active ratio vs accuracy")
    save("07_active_ratio_nfe.png", ratios, [r["nfe_mean"] for r in selected], "active ratio (%)", "mean NFE", "Active ratio vs NFE")
    save("08_active_ratio_work.png", ratios, [r["w_active_reduction_percent"] for r in selected], "active ratio (%)", "W_active reduction (%)", "Active ratio vs work")
    cost = summary.get("costs", {})
    names = [r["name"] for r in selected if r["name"] in cost]
    for number, ep in ((9,4),(10,8)):
        vals=[cost[name][f"ep{ep}"]["s1"]["stage_ms_per_request"] for name in names]
        save(f"{number:02d}_policy_ep{ep}_stage.png", list(range(len(names))), vals, "configuration", "stage sum (ms)", f"EP{ep} policy stage (S1)")
    baseline_name = summary["baseline_name"]
    for number, ep in ((11,4),(12,8)):
        x=[]; y=[]
        for name in [baseline_name] + names:
            if name in cost:
                x.append(len(x)); y.append(cost[name][f"ep{ep}"]["s1"]["max_mean_mean"])
        save(f"{number:02d}_ep{ep}_maxmean.png", x, y, "configuration", "mean max/mean", f"EP{ep} max/mean before/after Stage 1")
    joint=summary.get("joint",{})
    labels=list(joint)
    save("13_joint_ep8_headroom.png", list(range(len(labels))), [joint[k].get("ep8_joint_perfect_gain_percent",0) for k in labels], "configuration", "gain (%)", "Stage1 + perfect balance EP8")
    save("14_quality_ep8_pareto.png", [100*r["accuracy"] for r in selected if r["name"] in cost], [cost[r["name"]]["ep8"]["s1"]["stage_ms_per_request"] for r in selected if r["name"] in cost], "accuracy (%)", "EP8 S1 stage ms/request", "Quality / simulated EP8 cost")


def write_reports(summary: dict, report_dir: Path):
    pred = summary["predictability"]
    config_by_name = {item["name"]: item for item in summary.get("configs", [])}
    (report_dir / "selective_refinement_predictability.md").write_text(f"""# Selective refinement predictability

Targeted heavy trace: {pred['requests']} fixed GSM8K requests, {pred['token_pairs']:,} adjacent still-masked token pairs. The existing 128-request aggregate trace remains the systems reference; this trace only supplies missing token confidence/hidden fields.

| Metric | Value |
|---|---:|
| confidence Pearson | {pred['confidence_pearson']:.4f} |
| confidence Spearman | {pred['confidence_spearman']:.4f} |
| within-block confidence-rank Spearman | {pred['confidence_rank_spearman_mean']:.4f} |
| top-25/50/75 overlap | {pred['top_overlap']['25']:.3f} / {pred['top_overlap']['50']:.3f} / {pred['top_overlap']['75']:.3f} |
| expert Jaccard mean | {pred['routing']['expert_jaccard_mean']:.3f} |
| EP4/EP8 destination Jaccard mean | {pred['routing']['ep4_rank_jaccard_mean']:.3f} / {pred['routing']['ep8_rank_jaccard_mean']:.3f} |
| held-out confidence AUROC | {pred['commit_prediction_heldout_8_15']['previous_confidence']['auroc']:.3f} |

Only previous-step observations are used. Critical-rank persistence was not substituted for token-level predictability.
""")
    rows=[]
    for config in summary["configs"]:
        if config:
            rows.append(f"| {config['name']} | {config['samples']} | {config['correct']}/{config['samples']} | {config['accuracy_delta_pp']:+.2f} | {config['nfe_delta_percent']:+.1f} | {config['w_active_reduction_percent']:.1f} | {config['parsed_answer_identity']:.3f} |")
    age_rows=[]
    for name,item in summary["drift"].items():
        if name not in ("p2_confhigh_f0_r75_age1_sanity8",
                        "p2_confhigh_f0_r75_age2_sanity8",
                        "p2_confhigh_f1_r75_age1_sanity8"):
            continue
        for age,value in item.get("staleness",{}).items():
            age_rows.append(
                f"| {name} | {age} | {value['samples']} | "
                f"{value['expert_jaccard_mean'] if value['expert_jaccard_mean'] is not None else float('nan'):.3f} | "
                f"{value['ep4_rank_jaccard_mean'] if value['ep4_rank_jaccard_mean'] is not None else float('nan'):.3f} | "
                f"{value['ep8_rank_jaccard_mean'] if value['ep8_rank_jaccard_mean'] is not None else float('nan'):.3f} | "
                f"{value['confidence_error_mean'] if value['confidence_error_mean'] is not None else float('nan'):.4f} | "
                f"{value['commit_disagreement'] if value['commit_disagreement'] is not None else float('nan'):.4f} |"
            )
    promoted = config_by_name["p2_confhigh_f0_r75_age1_sanity8"]
    promoted_note = (
        f"\nThe promoted P2 gate used all {promoted['samples']} requests: "
        f"baseline {promoted['baseline_correct']}/{promoted['samples']} versus "
        f"selective {promoted['correct']}/{promoted['samples']}. It corrected "
        f"{promoted['baseline_wrong_corrected']} baseline failures and regressed "
        f"{promoted['additional_wrong']} baseline success (paired exact "
        f"p={promoted['paired_exact_p']:.4f}). The apparent score improvement is "
        "not statistically significant and is not claimed as a quality gain; the "
        "G2 PASS is based on the specified no-more-than-one-additional-wrong safety "
        "criterion, complete termination, and reduced NFE/work.\n"
    )
    (report_dir / "selective_refinement_freeze_safety.md").write_text("""# Selective refinement freeze safety

All timings here are excluded: F0/F1 execute the dense forward and only replace outputs to isolate causal quality risk.

| Configuration | n | correct | accuracy delta (pp) | NFE delta (%) | W_active reduction (%) | parsed identity |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + promoted_note + """

## Freeze staleness

| configuration | age | samples | expert Jaccard | EP4-rank Jaccard | EP8-rank Jaccard | confidence error | commit disagreement |
|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(age_rows) + "\n\nF0 caches the routed-MoE branch output only. F1 caches each decoder layer's post-layer state for frozen current-block positions; attention still executes densely, so this is a semantic emulation rather than a sparse KV implementation.\n")
    base=summary["baseline_name"]; lines=[]
    for name,cost in summary.get("costs",{}).items():
        for ep in EP_TARGETS:
            b=summary["costs"][base][f"ep{ep}"]
            for substrate in SUBSTRATES:
                if substrate in cost[f"ep{ep}"] and substrate in b:
                    gain=100*(1-cost[f"ep{ep}"][substrate]["stage_ms_per_request"]/b[substrate]["stage_ms_per_request"])
                    item=cost[f"ep{ep}"][substrate]
                    lines.append(f"| {name} | EP{ep} | {substrate.upper()} | {item['dispatch_ms_per_request']:.1f} | {item['expert_ms_per_request']:.1f} | {item['combine_ms_per_request']:.1f} | {item['stage_ms_per_request']:.1f} | {gain:+.2f} | {item['max_mean_mean']:.3f} | {item['max_second_mean']:.3f} | {item['cv_mean']:.3f} | {item['wait_fraction_mean']:.3f} | {item['remote_bytes']/item['requests']/1e6:.1f} | {item['fanout_mean']:.2f} |")
    oracle=summary["oracles"]
    oracle_text=f"""

## Future-aware progress oracles

- O1 next-commit active fraction: {100*oracle['o1_next_commit_active_fraction']:.2f}%.
- O2 set covering 90% of next commits: {100*oracle['o2_90pct_progress_active_fraction']:.2f}%.
- O3 75%-budget progress-preserving max/mean reduction: EP4 {100*oracle['o3']['ep4']['relative_reduction']:.2f}%, EP8 {100*oracle['o3']['ep8']['relative_reduction']:.2f}%.

O1--O3 use future baseline commits and are upper bounds, not online policies or quality-validated rollouts.
"""
    (report_dir / "selective_refinement_ep4_ep8_oracle.md").write_text("""# Selective refinement EP4/EP8 projection

All EP4/EP8 numbers are **SIMULATED-EP4/EP8-EP2-CALIBRATED**. S0 keeps full rows; S1 removes frozen current-block MASK rows only; S2 is a live-aware sparse oracle. None is measured sparse-runtime speedup.

| Configuration | Target | substrate | dispatch | expert | combine | stage ms/request | gain vs matched P0 | max/mean | max/second | CV | wait | remote MB/request | fanout |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(lines) + oracle_text + "\n")
    joint_lines=[]
    for name,row in summary.get("joint",{}).items():
        config = config_by_name[name]
        joint_lines.append(
            f"| {name} | {config['samples']} | {config['accuracy_delta_pp']:+.2f} | "
            f"{row.get('ep4_stage1_gain_percent',0):.2f} | "
            f"{row.get('ep4_joint_global_gain_percent',0):.2f} | "
            f"{row.get('ep4_joint_perfect_gain_percent',0):.2f} | "
            f"{row.get('ep8_stage1_gain_percent',0):.2f} | "
            f"{row.get('ep8_joint_global_gain_percent',0):.2f} | "
            f"{row.get('ep8_joint_perfect_gain_percent',0):.2f} |"
        )
    final = summary["final_summary"]
    required = f"""

PREVIOUS_STEP_PREDICTABILITY:
{final['PREVIOUS_STEP_PREDICTABILITY']}

ONE_STEP_MASK_FREEZE_SAFETY:
{final['ONE_STEP_MASK_FREEZE_SAFETY']}

MAX_SAFE_FREEZE_AGE:
{final['MAX_SAFE_FREEZE_AGE']}

BEST_GENERIC_SELECTIVE_POLICY:
{final['BEST_GENERIC_SELECTIVE_POLICY']}

BEST_EP_AWARE_SELECTIVE_POLICY:
{final['BEST_EP_AWARE_SELECTIVE_POLICY']}

GENERIC_ACTIVE_WORK_REDUCTION:
{final['GENERIC_ACTIVE_WORK_REDUCTION']}

EP4_EP_AWARE_INCREMENTAL_STAGE_GAIN:
{final['EP4_EP_AWARE_INCREMENTAL_STAGE_GAIN']}

EP8_EP_AWARE_INCREMENTAL_STAGE_GAIN:
{final['EP8_EP_AWARE_INCREMENTAL_STAGE_GAIN']}

EP4_STAGE1_ONLY_GAIN:
{final['EP4_STAGE1_ONLY_GAIN']}

EP8_STAGE1_ONLY_GAIN:
{final['EP8_STAGE1_ONLY_GAIN']}

EP4_STAGE1_PLUS_BALANCE_GAIN:
{final['EP4_STAGE1_PLUS_BALANCE_GAIN']}

EP8_STAGE1_PLUS_BALANCE_GAIN:
{final['EP8_STAGE1_PLUS_BALANCE_GAIN']}

EP8_JOINT_HEADROOM_GE_15_PERCENT:
{final['EP8_JOINT_HEADROOM_GE_15_PERCENT']}

QUALITY_DELTA_AT_BEST_JOINT_POINT:
{final['QUALITY_DELTA_AT_BEST_JOINT_POINT']}

PRIMARY_BLOCKER:
{final['PRIMARY_BLOCKER']}

VERDICT:
{final['VERDICT']}

DO_NOT_IMPLEMENT_PRODUCTION_SPARSE_RUNTIME_AUTOMATICALLY:
true
"""
    (report_dir / "selective_refinement_joint_balance_summary.md").write_text("""# Stage 1 + residual balance summary

B1 uses the existing train-0..63 global-static replica map with budget 8 and conservatively leaves communication unchanged. B2 lowers expert critical time to rank mean and is an impossible perfect-balance upper bound.

The prior H1 EP8 perfect-balance-only reference was 12.37%. Crossing 15% here is meaningful only when the selective S1 substrate and B2 oracle are both stated; it is not a live runtime result.

P2 quality safety is promoted on n=128. P6 and its EP-aware incremental comparison are paired on n=32 only; P6 was not promoted to 128 because its measured EP-specific increment did not pass the 3-point discovery gate.

| Configuration | n | quality delta (pp) | EP4 S1 | EP4 S1+B1 | EP4 S1+B2 | EP8 S1 | EP8 S1+B1 | EP8 S1+B2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(joint_lines) +
        "\n\nThe required `Stage1+Balance` fields below use B2 perfect balance (oracle); B1 global-static values remain in the table and do not cross the same gate.\n\n" +
        summary["ep_specific_decomposition"]["interpretation"] + "\n" + required)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, default=Path("artifacts/virtual_ep/20260917_selective_refinement/rollouts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--baseline", default="p0_full_none_ratio100_parity8")
    args=parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir=args.rollout_root/args.baseline
    baseline=read_generations(baseline_dir)
    discovery_root = Path("artifacts/virtual_ep/20260917_four_hypothesis_discovery/aggregate_v2")
    discovery_trace = discovery_root / "gsm8k_128_discovery_v2_fixed.npz"
    baseline=extend_reference_baseline(baseline, discovery_root, discovery_trace)
    configs=[]; drifts={}; costs={}
    projector=CostProjector(
        Path("artifacts/virtual_ep/20260917_four_hypothesis_discovery/compute_model/ep4_grouped_mm_discovery.csv"),
        Path("artifacts/virtual_ep/20260916_215010/compute_model/ep8_grouped_mm.csv"),
        Path("artifacts/virtual_ep/20260916_215010/communication/model.json"),
        discovery_trace,
    )
    for directory in sorted(args.rollout_root.iterdir()):
        if (not directory.is_dir()
                or directory.name.endswith("_parity1")
                or not list(directory.glob("generations_worker*.jsonl"))):
            continue
        config=config_summary(directory, baseline)
        if not config: continue
        configs.append(config); drifts[directory.name]=drift_summary(directory)
        # Systems comparisons use the same paired 0--31 routes for every
        # policy. The 128-request extension is a quality/safety validation.
        costs[directory.name]=projector.project(directory, set(range(32)))
    summary={
        "evidence": {
            "model":"inclusionAI/LLaDA2.0-mini", "threshold":.95,
            "measured":"dense-emulated generation quality/freeze drift on GPUs 0,1",
            "simulated":"EP4/EP8 S0/S1/S2 and balance oracles, EP2-calibrated",
            "no_sparse_runtime_speedup_claim":True,
            "quality_reference":"existing threshold=.95 GSM8K 128 discovery cohort (118/128)",
            "systems_projection_cohort":"paired request IDs 0-31",
            "token_predictability_cohort":"fixed request IDs 0-15",
            "communication_model":"EP2-CALIBRATED DeepEP normal median; replicated-state bridge excluded",
            "ep4_compute_model":"20260917_four_hypothesis_discovery/ep4_grouped_mm_discovery.csv",
            "ep8_compute_model":"20260916_215010/ep8_grouped_mm.csv",
            "source_partition_assumption":"preserve original balanced source rank after row selection",
        },
        "baseline_name":args.baseline,
        "predictability":predictability(baseline_dir/"traces"),
        "configs":configs, "drift":drifts, "costs":costs,
        "oracles":oracle_summary(baseline_dir/"traces"),
    }
    joint={}
    for config in configs:
        name=config["name"]
        if name==args.baseline or name not in costs: continue
        row={}
        for ep in EP_TARGETS:
            base=costs[args.baseline][f"ep{ep}"]["s1"]["stage_ms_per_request"]
            item=costs[name][f"ep{ep}"]["s1"]
            row[f"ep{ep}_stage1_gain_percent"]=100*(1-item["stage_ms_per_request"]/base)
            row[f"ep{ep}_joint_global_gain_percent"]=100*(1-item["global_static_replica_stage_ms_per_request"]/base)
            row[f"ep{ep}_joint_perfect_gain_percent"]=100*(1-item["perfect_balance_stage_ms_per_request"]/base)
        joint[name]=row
    summary["joint"]=joint
    by_name={row["name"]:row for row in configs}
    generic_name="p2_confhigh_f0_r75_age1_sanity8"
    aware_name="p6_confep_l05_f0_r75_age1_sanity8"
    generic=by_name[generic_name]; aware=by_name[aware_name]
    generic_joint=joint[generic_name]; aware_joint=joint[aware_name]
    ep4_increment=(aware_joint["ep4_stage1_gain_percent"]
                   - generic_joint["ep4_stage1_gain_percent"])
    ep8_increment=(aware_joint["ep8_stage1_gain_percent"]
                   - generic_joint["ep8_stage1_gain_percent"])
    pred=summary["predictability"]
    predict_verdict=("PASS" if pred["confidence_spearman"] >= .4
                     and pred["commit_prediction_heldout_8_15"]["previous_confidence"]["auroc"] >= .7
                     else "WEAK" if pred["confidence_spearman"] >= .2 else "FAIL")
    freeze_verdict=(
        "PASS" if generic["samples"] >= 128 and generic["additional_wrong"] <= 1
        and (generic["nfe_delta_percent"] <= 10 or generic["w_active_reduction_percent"] >= 20)
        else "WEAK" if generic["accuracy_delta_pp"] >= 0 and generic["samples"] >= 32
        else "FAIL"
    )
    joint_gate=aware_joint["ep8_joint_perfect_gain_percent"] >= 15
    if freeze_verdict == "PASS" and ep8_increment >= 3 and joint_gate:
        verdict="GO"; blocker="none at the discovery gate"
    elif freeze_verdict in ("PASS","WEAK") and joint_gate:
        verdict="HOLD"; blocker=(
            "EP-aware selection adds less than the required 3 percentage points over confidence-only; "
            "the >=15% joint result requires the perfect-balance oracle."
        )
    else:
        verdict="NO-GO"; blocker="quality-safe freeze or joint systems headroom did not pass its gate"
    summary["final_summary"]={
        "PREVIOUS_STEP_PREDICTABILITY":predict_verdict,
        "ONE_STEP_MASK_FREEZE_SAFETY":freeze_verdict,
        "MAX_SAFE_FREEZE_AGE":1,
        "BEST_GENERIC_SELECTIVE_POLICY":"P2 previous-confidence-high, F0, 75% active, max_freeze_age=1",
        "BEST_EP_AWARE_SELECTIVE_POLICY":"P6 confidence+EP8 load, lambda_max=0.5, F0, 75% active, max_freeze_age=1",
        "GENERIC_ACTIVE_WORK_REDUCTION":f"{generic['w_active_reduction_percent']:.2f}%",
        "EP4_EP_AWARE_INCREMENTAL_STAGE_GAIN":f"{ep4_increment:.2f} percentage points",
        "EP8_EP_AWARE_INCREMENTAL_STAGE_GAIN":f"{ep8_increment:.2f} percentage points",
        "EP4_STAGE1_ONLY_GAIN":f"{aware_joint['ep4_stage1_gain_percent']:.2f}% (S1)",
        "EP8_STAGE1_ONLY_GAIN":f"{aware_joint['ep8_stage1_gain_percent']:.2f}% (S1)",
        "EP4_STAGE1_PLUS_BALANCE_GAIN":f"{aware_joint['ep4_joint_perfect_gain_percent']:.2f}% (S1+B2 perfect-balance oracle)",
        "EP8_STAGE1_PLUS_BALANCE_GAIN":f"{aware_joint['ep8_joint_perfect_gain_percent']:.2f}% (S1+B2 perfect-balance oracle)",
        "EP8_JOINT_HEADROOM_GE_15_PERCENT":"YES" if joint_gate else "NO",
        "QUALITY_DELTA_AT_BEST_JOINT_POINT":f"{aware['accuracy_delta_pp']:+.2f} pp on n={aware['samples']}",
        "PRIMARY_BLOCKER":blocker,
        "VERDICT":verdict,
        "DO_NOT_IMPLEMENT_PRODUCTION_SPARSE_RUNTIME_AUTOMATICALLY":True,
    }
    summary["ep_specific_decomposition"]={
        "ep8_total_stage_increment_pp_p6_minus_p2":ep8_increment,
        "ep8_s1_max_mean_p6_minus_p2":(
            costs[aware_name]["ep8"]["s1"]["max_mean_mean"]
            - costs[generic_name]["ep8"]["s1"]["max_mean_mean"]
        ),
        "ep8_s2_max_mean_p6_minus_p2":(
            costs[aware_name]["ep8"]["s2"]["max_mean_mean"]
            - costs[generic_name]["ep8"]["s2"]["max_mean_mean"]
        ),
        "interpretation":(
            "Total stage delta includes P6-induced NFE/trajectory changes. S1 max/mean does not improve; "
            "therefore the delta is not evidence of material EP load shaping."
        ),
    }
    plot_outputs(summary,args.report_dir/"figures/selective_refinement")
    write_reports(summary,args.report_dir)
    (args.report_dir/"selective_refinement_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps({"configs":len(configs),"output":str(args.report_dir/"selective_refinement_summary.json")},indent=2))


if __name__ == "__main__":
    main()
