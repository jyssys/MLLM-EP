#!/usr/bin/env python3
"""Offline expert-placement / rank-saturation characterization.

Only previously captured routed-expert IDs are read.  No CUDA, vLLM, model, or
GPU is initialized by this program.  Placement policies are fixed-size (32
experts per EP rank) heuristic analyses; they never alter the captured routes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EP = 4
EXPERTS = 128
LOCAL_EXPERTS = EXPERTS // EP
LAYERS = 48
TOPK = 8
IMAGE_TOKEN_ID = 151655
SEED = 20260827
JOINT_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Trace:
    pair_id: int
    request_id: str
    modality: str
    category: str
    bucket: str
    route_path: Path
    routes: np.ndarray  # [tokens, layer, topk]
    token_ids: np.ndarray


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def resolve_route(root: Path, rel: str) -> Path:
    for p in (root / rel, root / Path(rel).name):
        if p.exists():
            return p
    raise FileNotFoundError(rel)


def load_traces(manifest_path: Path, route_root: Path) -> tuple[list[Trace], list[Trace], dict]:
    manifest = json.loads(manifest_path.read_text())
    vision, text = [], []
    for pair in manifest["pairs"]:
        for modality, target in (("vision", vision), ("text", text)):
            item = pair[modality]
            path = resolve_route(route_root, item["route_file"])
            with np.load(path, allow_pickle=False) as z:
                routes = np.asarray(z["routed_experts"], dtype=np.int64)
                ids = np.asarray(z["prompt_token_ids"], dtype=np.int64)
            if routes.ndim != 3 or routes.shape[1:] != (LAYERS, TOPK):
                raise ValueError(f"unexpected {path}: {routes.shape}")
            if routes.shape[0] != len(ids) or routes.min() < 0 or routes.max() >= EXPERTS:
                raise ValueError(f"invalid route artifact {path}")
            target.append(Trace(int(pair["pair_id"]), str(item["request_id"]), modality,
                                str(item["category"]), str(pair["token_bucket"]), path,
                                routes, ids))
    if len(vision) != 24 or len(text) != 24:
        raise ValueError(f"expected 24+24 traces, got {len(vision)}+{len(text)}")
    return vision, text, manifest


def linear_placement() -> np.ndarray:
    return np.arange(EXPERTS, dtype=np.int64) // LOCAL_EXPERTS


def apply_placement(routes: np.ndarray, placement: np.ndarray) -> np.ndarray:
    return placement[routes]


def saturation_metrics(routes: np.ndarray, placement: np.ndarray) -> dict:
    """Metrics for route rows [tokens, topk] under an expert->rank map."""
    if routes.size == 0:
        return {"tokens": 0, "assignments": 0, "mean_u": 0.0, "median_u": 0.0,
                "p90_u": 0.0, "p_u4": 0.0, "p_u_ge3": 0.0,
                "rank_load_max": 0, "rank_load_mean": 0.0, "rank_load_cv": 0.0,
                "rank_load_max_mean": 0.0, "critical_rank": -1,
                "remote_volume_proxy": 0.0, "rank_loads": [0] * EP}
    ranks = apply_placement(routes, placement)
    u = np.zeros(len(ranks), dtype=np.int64)
    for k in range(ranks.shape[1]):
        # top-k is unique in the captured router output, but using a bool mask
        # also makes this robust to a malformed duplicate route.
        u += 0
    used = np.zeros((len(ranks), EP), dtype=bool)
    row = np.arange(len(ranks))
    for k in range(ranks.shape[1]):
        used[row, ranks[:, k]] = True
    u = used.sum(axis=1)
    flat = ranks.reshape(-1)
    loads = np.bincount(flat, minlength=EP).astype(np.int64)
    mean = float(loads.mean())
    return {
        "tokens": int(len(routes)), "assignments": int(flat.size),
        "mean_u": float(u.mean()), "median_u": float(np.median(u)),
        "p90_u": float(np.percentile(u, 90)), "p_u4": float(np.mean(u == 4)),
        "p_u_ge3": float(np.mean(u >= 3)), "rank_load_max": int(loads.max()),
        "rank_load_mean": mean, "rank_load_cv": float(loads.std() / mean) if mean else 0.0,
        "rank_load_max_mean": float(loads.max() / mean) if mean else 0.0,
        "critical_rank": int(np.argmax(loads)), "remote_volume_proxy": float(np.sum(u - 1)),
        "rank_loads": loads.tolist(),
    }


def concat_routes(traces: list[Trace], layer: int, modality_mask: str = "all", request_ids: set[str] | None = None) -> np.ndarray:
    rows: list[np.ndarray] = []
    for t in traces:
        if request_ids is not None and t.request_id not in request_ids:
            continue
        if modality_mask == "vision":
            pos = np.flatnonzero(t.token_ids == IMAGE_TOKEN_ID) if t.modality == "vision" else np.empty(0, dtype=np.int64)
        elif modality_mask == "text":
            pos = np.flatnonzero(t.token_ids != IMAGE_TOKEN_ID) if t.modality == "vision" else np.arange(len(t.token_ids))
        else:
            pos = np.arange(len(t.token_ids))
        if len(pos):
            rows.append(t.routes[pos, layer, :])
    return np.concatenate(rows, axis=0) if rows else np.empty((0, TOPK), dtype=np.int64)


def source_positions(t: Trace, source: str) -> np.ndarray:
    if source == "vision":
        return np.flatnonzero(t.token_ids == IMAGE_TOKEN_ID) if t.modality == "vision" else np.empty(0, dtype=np.int64)
    if source == "text":
        return np.flatnonzero(t.token_ids != IMAGE_TOKEN_ID) if t.modality == "vision" else np.arange(len(t.token_ids))
    return np.arange(len(t.token_ids))


def current_characterization(traces_by_mod: dict[str, list[Trace]], out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sat_rows: list[dict] = []
    expert_rows: list[dict] = []
    placement = linear_placement()
    for modality, traces in traces_by_mod.items():
        groups: list[tuple[str, str, list[tuple[Trace, int]]]] = []
        groups.append(("global", "all", [(t, l) for t in traces for l in range(LAYERS)]))
        for l in range(LAYERS):
            groups.append(("layer", str(l), [(t, l) for t in traces]))
        for t in traces:
            groups.append(("request", t.request_id, [(t, l) for l in range(LAYERS)]))
            for l in range(LAYERS):
                groups.append(("request_layer", f"{t.request_id}:L{l}", [(t, l)]))
        for scope, key, items in groups:
            chunks = [t.routes[source_positions(t, "all"), l, :] for t, l in items]
            routes = np.concatenate(chunks, axis=0) if chunks else np.empty((0, TOPK), dtype=np.int64)
            m = saturation_metrics(routes, placement)
            row = {"scope": scope, "key": key, "modality": modality, **{k: v for k, v in m.items() if k != "rank_loads"}}
            sat_rows.append(row)
            for r, value in enumerate(m["rank_loads"]):
                expert_rows.append({"scope": scope, "key": key, "modality": modality, "rank": r, "assignment_count": value})
            counts = np.bincount(routes.reshape(-1), minlength=EXPERTS)
            for e, value in enumerate(counts):
                expert_rows.append({"scope": scope, "key": key, "modality": modality, "expert": e, "assignment_count": int(value)})
    sat = pd.DataFrame(sat_rows)
    exp = pd.DataFrame(expert_rows)
    sat.to_csv(out / "current_saturation.csv", index=False)
    exp.to_csv(out / "current_expert_counts.csv", index=False)
    return sat, exp


def assignment_counts(routes: np.ndarray) -> np.ndarray:
    return np.bincount(routes.reshape(-1), minlength=EXPERTS).astype(np.float64)


def load_ratio(placement: np.ndarray, counts: np.ndarray) -> float:
    loads = np.bincount(placement, weights=counts, minlength=EP)
    return float(loads.max() / loads.mean()) if loads.mean() else 0.0


def coactivation_matrix(routes: np.ndarray) -> np.ndarray:
    w = np.zeros((EXPERTS, EXPERTS), dtype=np.float64)
    if len(routes) == 0:
        return w
    # Co-activation is a deterministic surrogate for minimizing unique rank
    # coverage: reward putting experts selected by the same token together.
    for a in range(TOPK):
        for b in range(a + 1, TOPK):
            np.add.at(w, (routes[:, a], routes[:, b]), 1.0)
            np.add.at(w, (routes[:, b], routes[:, a]), 1.0)
    np.fill_diagonal(w, 0.0)
    return w


def greedy_load(counts: np.ndarray) -> np.ndarray:
    order = sorted(range(EXPERTS), key=lambda e: (-counts[e], e))
    p = np.full(EXPERTS, -1, dtype=np.int64)
    loads = np.zeros(EP, dtype=np.float64)
    sizes = np.zeros(EP, dtype=np.int64)
    for e in order:
        choices = [r for r in range(EP) if sizes[r] < LOCAL_EXPERTS]
        r = min(choices, key=lambda x: (loads[x], sizes[x], x))
        p[e] = r; sizes[r] += 1; loads[r] += counts[e]
    return p


def greedy_coactivation(routes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    w = coactivation_matrix(routes)
    degree = w.sum(axis=1)
    order = sorted(range(EXPERTS), key=lambda e: (-degree[e], -counts[e], e))
    p = np.full(EXPERTS, -1, dtype=np.int64)
    members: list[list[int]] = [[] for _ in range(EP)]
    # Seed and then grow the group that maximizes already assigned coactivation.
    for e in order:
        choices = [r for r in range(EP) if len(members[r]) < LOCAL_EXPERTS]
        score = [(sum(w[e, x] for x in members[r]), -len(members[r]), -r) for r in choices]
        r = choices[max(range(len(choices)), key=lambda i: score[i])]
        p[e] = r; members[r].append(e)
    return p


def surrogate_joint_score(p: np.ndarray, counts: np.ndarray, w: np.ndarray, lam: float) -> float:
    loads = np.bincount(p, weights=counts, minlength=EP)
    load = float(loads.max() / loads.mean()) if loads.mean() else 0.0
    within = 0.0
    for r in range(EP):
        ids = np.flatnonzero(p == r)
        within += float(w[np.ix_(ids, ids)].sum()) / 2.0
    total = float(w.sum() / 2.0)
    sat_proxy = 1.0 - within / total if total else 0.0
    # Load ratio is normalized around 1; saturation proxy is [0,1].
    return lam * load + (1.0 - lam) * sat_proxy


def improve_swaps(p: np.ndarray, counts: np.ndarray, w: np.ndarray, mode: str, lam: float = 0.5) -> np.ndarray:
    p = p.copy()
    def score(q: np.ndarray) -> float:
        if mode == "load":
            return load_ratio(q, counts)
        if mode == "sat":
            # Pairwise coactivation objective is the bounded P_sat surrogate.
            return surrogate_joint_score(q, counts, w, 0.0)
        return surrogate_joint_score(q, counts, w, lam)
    cur = score(p)
    # Bounded deterministic refinement: exhaustive 128^2 swaps at every layer
    # made this artifact-only characterization needlessly expensive.  The
    # largest-load/coactivation experts are the only candidates considered;
    # the resulting policy remains explicitly a heuristic, never an oracle.
    degree = w.sum(axis=1)
    if mode == "load":
        candidate = sorted(range(EXPERTS), key=lambda e: (-counts[e], e))[:40]
    elif mode == "sat":
        candidate = sorted(range(EXPERTS), key=lambda e: (-degree[e], -counts[e], e))[:40]
    else:
        candidate = sorted(range(EXPERTS), key=lambda e: (-(counts[e] / max(counts.mean(), 1.0) + degree[e] / max(degree.mean(), 1.0)), e))[:40]
    for _ in range(3):
        best = cur
        best_pair = None
        for ii, e in enumerate(candidate):
            for f in candidate[ii + 1:]:
                if p[e] == p[f]:
                    continue
                q = p.copy(); q[e], q[f] = q[f], q[e]
                value = score(q)
                if value < best - 1e-12:
                    best, best_pair = value, (e, f)
        if best_pair is None:
            break
        e, f = best_pair; p[e], p[f] = p[f], p[e]; cur = best
    return p


def make_placement(routes: np.ndarray, policy: str) -> tuple[np.ndarray, str]:
    counts = assignment_counts(routes)
    w = coactivation_matrix(routes)
    if policy == "P0":
        return linear_placement(), "fixed_linear"
    if policy == "P_load":
        return improve_swaps(greedy_load(counts), counts, w, "load"), "greedy_load+swap"
    if policy == "P_sat":
        return improve_swaps(greedy_coactivation(routes, counts), counts, w, "sat"), "coactivation_surrogate+swap"
    if policy.startswith("P_joint_"):
        lam = float(policy.split("_")[-1])
        seed = greedy_load(counts) if lam >= 0.5 else greedy_coactivation(routes, counts)
        return improve_swaps(seed, counts, w, "joint", lam), f"joint_surrogate_lambda_{lam:g}"
    if policy == "P_V_load" or policy == "P_T_load":
        return improve_swaps(greedy_load(counts), counts, w, "load"), "modality_greedy_load+swap"
    if policy == "P_V_sat" or policy == "P_T_sat":
        return improve_swaps(greedy_coactivation(routes, counts), counts, w, "sat"), "modality_coactivation_surrogate+swap"
    raise ValueError(policy)


def policies() -> list[str]:
    return ["P0", "P_load", "P_sat"] + [f"P_joint_{x:g}" for x in JOINT_LAMBDAS] + ["P_V_load", "P_T_load", "P_V_sat", "P_T_sat"]


def evaluate_placements(all_traces: list[Trace], visions: list[Trace], texts: list[Trace], out: Path) -> tuple[pd.DataFrame, dict]:
    profiles = {"all": all_traces, "vision": visions, "text": texts}
    rows: list[dict] = []
    assignments: dict = {"placement_definition": "expert_to_rank arrays; each layer has exactly 32 experts/rank", "layers": {}}
    policy_list = policies()
    # Static policies P0/load/sat/joint are trained on all requests. Modality
    # policies are trained only on their source modality. All are evaluated on
    # all three profiles.
    for layer in range(LAYERS):
        assignments["layers"][str(layer)] = {}
        route_for_policy: dict[str, np.ndarray] = {}
        for policy in policy_list:
            if policy.startswith("P_V_"):
                source = "vision"
            elif policy.startswith("P_T_"):
                source = "text"
            else:
                source = "all"
            fit_routes = concat_routes(profiles[source], layer, "all")
            p, optimizer = make_placement(fit_routes, policy)
            if np.bincount(p, minlength=EP).tolist() != [LOCAL_EXPERTS] * EP:
                raise AssertionError(f"capacity violation {policy} L{layer}")
            assignments["layers"][str(layer)][policy] = {"source": source, "optimizer": optimizer, "expert_to_rank": p.tolist()}
            route_for_policy[policy] = p
            for eval_name, eval_traces in profiles.items():
                routes = concat_routes(eval_traces, layer, "all")
                m = saturation_metrics(routes, p)
                rows.append({"scope": "layer", "layer": layer, "placement": policy,
                             "fit_profile": source, "eval_profile": eval_name,
                             "optimizer": optimizer, **{k: v for k, v in m.items() if k != "rank_loads"}})
    result = pd.DataFrame(rows)
    result.to_csv(out / "placement_frontier.csv", index=False)
    write_json(out / "placement_assignments.json", assignments)
    return result, assignments


def modality_cross_eval(frontier: pd.DataFrame, out: Path) -> pd.DataFrame:
    # Average layer-level metrics for each policy when evaluated on Vision/Text.
    cols = ["rank_load_max_mean", "rank_load_cv", "mean_u", "p_u4", "p_u_ge3", "remote_volume_proxy"]
    rows = []
    for policy, g in frontier.groupby("placement"):
        for profile, h in g[g.eval_profile.isin(["vision", "text", "all"])].groupby("eval_profile"):
            row = {"placement": policy, "eval_profile": profile, "fit_profile": h.fit_profile.iloc[0]}
            for c in cols:
                row[c] = float(h[c].mean())
            probs = h.critical_rank.value_counts(normalize=True).to_numpy(dtype=float)
            row["critical_rank_entropy"] = float(-(probs * np.log2(probs)).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "modality_cross_eval.csv", index=False)
    return df


def calibration_transfer(visions: list[Trace], texts: list[Trace], out: Path) -> pd.DataFrame:
    all_traces = visions + texts
    req_v = [t.request_id for t in visions]
    # Paired controls retain the same fold assignment as their real image.
    pair_to_text = {t.pair_id: t for t in texts}
    rows: list[dict] = []
    for fold, cal_idx in (("A_first12", set(req_v[:12])), ("B_last12", set(req_v[12:]))):
        cal_v = [t for t in visions if t.request_id in cal_idx]
        eval_v = [t for t in visions if t.request_id not in cal_idx]
        cal_pairs = {t.pair_id for t in cal_v}
        eval_pairs = {t.pair_id for t in eval_v}
        cal_t = [pair_to_text[p] for p in sorted(cal_pairs)]
        eval_t = [pair_to_text[p] for p in sorted(eval_pairs)]
        cal_profiles = {"all": cal_v + cal_t, "vision": cal_v, "text": cal_t}
        eval_profiles = {"all": eval_v + eval_t, "vision": eval_v, "text": eval_t}
        for policy in policies():
            source = "vision" if policy.startswith("P_V_") else "text" if policy.startswith("P_T_") else "all"
            for layer in range(LAYERS):
                p, optimizer = make_placement(concat_routes(cal_profiles[source], layer, "all"), policy)
                for eval_name, traces in eval_profiles.items():
                    m = saturation_metrics(concat_routes(traces, layer, "all"), p)
                    rows.append({"fold": fold, "policy": policy, "fit_profile": source,
                                 "eval_profile": eval_name, "layer": layer,
                                 "partition": "calibration", "optimizer": optimizer,
                                 **{k: v for k, v in m.items() if k != "rank_loads"}})
                # Evaluate calibration itself for direct transfer deltas.
                m = saturation_metrics(concat_routes(cal_profiles[source], layer, "all"), p)
                rows.append({"fold": fold, "policy": policy, "fit_profile": source,
                             "eval_profile": source, "layer": layer,
                             "partition": "calibration_fit", "optimizer": optimizer,
                             **{k: v for k, v in m.items() if k != "rank_loads"}})
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration_transfer.csv", index=False)
    return df


def figures(current: pd.DataFrame, frontier: pd.DataFrame, cross: pd.DataFrame, transfer: pd.DataFrame, visions: list[Trace], texts: list[Trace], out: Path) -> list[str]:
    fd = out / "figures"; fd.mkdir(parents=True, exist_ok=True); paths = []
    # 1. Current linear placement saturation distribution.
    vals = {}
    for modality, traces in (("Vision", visions), ("Text", texts)):
        rs = np.concatenate([concat_routes(traces, l, "all") for l in range(LAYERS)], axis=0)
        ranks = apply_placement(rs, linear_placement()); used = np.zeros((len(ranks), EP), dtype=bool); ii = np.arange(len(ranks))
        for k in range(TOPK): used[ii, ranks[:, k]] = True
        vals[modality] = used.sum(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4)); bins = np.arange(.5, 4.6, 1)
    ax.hist(vals["Vision"], bins=bins, alpha=.65, label="Vision", rwidth=.8)
    ax.hist(vals["Text"], bins=bins, alpha=.65, label="Text", rwidth=.55)
    ax.set_xticks([1,2,3,4]); ax.set_xlabel("unique destination ranks per token (u)"); ax.set_ylabel("token count"); ax.legend(); ax.set_title("Current linear placement: rank coverage")
    fig.tight_layout(); p=fd/"plot1_unique_rank_token_histogram.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    # 2. Pareto-like placement frontier on all routes, averaged across layers.
    a = frontier[frontier.eval_profile == "all"].groupby("placement", as_index=False).agg(rank_cv=("rank_load_cv","mean"), sat=("mean_u","mean"), max_mean=("rank_load_max_mean","mean"))
    fig, ax = plt.subplots(figsize=(8,5))
    for _, r in a.iterrows():
        ax.scatter(r.rank_cv, r.sat/EP, s=45); ax.annotate(str(r.placement), (r.rank_cv, r.sat/EP), fontsize=7, xytext=(3,3), textcoords="offset points")
    ax.set_xlabel("rank-load CV (lower is better)"); ax.set_ylabel("mean unique-rank saturation u/4 (lower is better)"); ax.set_title("Placement load–saturation trade-off (all profiles)"); fig.tight_layout(); p=fd/"plot2_load_vs_saturation_frontier.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    # 3. Cross-evaluation heatmap for major policies.
    names = ["P0","P_load","P_sat","P_joint_0","P_joint_0.5","P_joint_1","P_V_load","P_T_load","P_V_sat","P_T_sat"]
    names = [x for x in names if x in set(cross.placement)]
    mat = []
    labels=[]
    for n in names:
        row=[]
        for prof in ("vision","text"):
            q=cross[(cross.placement==n)&(cross.eval_profile==prof)].iloc[0]
            row += [q.rank_load_cv, q.mean_u/EP]
        mat.append(row); labels.append(n)
    fig, ax=plt.subplots(figsize=(8,5)); im=ax.imshow(np.asarray(mat), aspect="auto", cmap="magma_r"); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=8); ax.set_xticks(range(4)); ax.set_xticklabels(["V load CV","V sat","T load CV","T sat"]); fig.colorbar(im,ax=ax,label="lower is better"); ax.set_title("Vision/Text placement cross-evaluation"); fig.tight_layout(); p=fd/"plot3_modality_cross_evaluation.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    # 4. Calibration transfer (saturation) plus current per-layer signal.
    q=transfer[(transfer.partition=="calibration") & (transfer.eval_profile.isin(["vision","text"]))].groupby(["policy","eval_profile","partition"],as_index=False).mean(numeric_only=True)
    fig, axes=plt.subplots(1,2,figsize=(13,4.5))
    for prof, sub in q.groupby("eval_profile"):
        axes[0].scatter(sub.policy, sub.mean_u/EP, label=prof, s=25)
    axes[0].tick_params(axis="x",rotation=70,labelsize=7); axes[0].set_ylabel("held-out mean u/4"); axes[0].set_title("Calibration → held-out saturation"); axes[0].legend()
    for prof, traces in (("Vision",visions),("Text",texts)):
        ys=[]
        for l in range(LAYERS): ys.append(saturation_metrics(concat_routes(traces,l,"all"),linear_placement())["mean_u"]/EP)
        axes[1].plot(range(LAYERS),ys,label=prof)
    axes[1].set_xlabel("decoder layer"); axes[1].set_ylabel("mean u/4"); axes[1].set_title("Current per-layer saturation"); axes[1].legend(); fig.tight_layout(); p=fd/"plot4_calibration_transfer_saturation.png"; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    return paths


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609/workload_manifest.json")); ap.add_argument("--route-root",type=Path,default=Path("poc_flashvep/deepep_revalidation/results/live_prefill_execution_regime_20260821_111609")); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    visions,texts,manifest=load_traces(args.manifest,args.route_root); all_traces=visions+texts
    shutil.copy2(args.manifest, args.output / "source_workload_manifest.json")
    write_json(args.output / "analysis_policy.json", {
        "ep": EP, "experts": EXPERTS, "local_experts_per_rank": LOCAL_EXPERTS,
        "layers": LAYERS, "top_k": TOPK, "image_token_id": IMAGE_TOKEN_ID,
        "placement_capacity": "exactly 32 experts per rank",
        "policies": policies(), "joint_lambdas": list(JOINT_LAMBDAS),
        "placement_fit": "per-layer; P0/load/sat/joint fit on all 48 traces, P_V/P_T fit on source modality",
        "saturation": "unique destination ranks among captured Top-8; remote proxy=sum(u-1)",
        "optimizer_note": "deterministic heuristic; no policy is called an exact oracle",
        "calibration": "Fold A first 12 image pairs -> last 12; Fold B reversed; paired text follows image pair",
        "gpu_execution": False,
    })
    current,_=current_characterization({"vision":visions,"text":texts},args.output)
    frontier,assignments=evaluate_placements(all_traces,visions,texts,args.output)
    cross=modality_cross_eval(frontier,args.output); transfer=calibration_transfer(visions,texts,args.output)
    fig_paths=figures(current,frontier,cross,transfer,visions,texts,args.output)
    # Aggregate policy metrics used by the fixed gate and report.
    agg=frontier[frontier.eval_profile=="all"].groupby("placement").agg(rank_cv=("rank_load_cv","mean"), sat=("mean_u","mean"), max_mean=("rank_load_max_mean","mean")).reset_index()
    p0=float(agg.loc[agg.placement=="P0","sat"].iloc[0]); pl=float(agg.loc[agg.placement=="P_load","sat"].iloc[0]); ps=float(agg.loc[agg.placement=="P_sat","sat"].iloc[0])
    load_improvement=float((agg.loc[agg.placement=="P0","max_mean"].iloc[0]-agg.loc[agg.placement=="P_load","max_mean"].iloc[0])/agg.loc[agg.placement=="P0","max_mean"].iloc[0])
    sat_change=float((ps-p0)/p0)
    # Stable direct summary for modality and current state.
    cv=(current[(current.scope=="global")].set_index("modality"))
    summary={"gpu_execution":False,"model_execution":False,"source_manifest":str(args.manifest),"route_root":str(args.route_root),"trace_counts":{"vision":len(visions),"text":len(texts)},"layers":LAYERS,"experts":EXPERTS,"ep_ranks":EP,"top_k":TOPK,"current_placement":"expert_id // 32","assignment_invariant":"route artifacts are unchanged; placement only remaps expert ids to rank labels","joint_lambda_grid":list(JOINT_LAMBDAS),"window_policy":"all 48 traces evaluated; modality policies fit only their source profile; calibration folds are first/last 12 image pairs with paired text controls","aggregate_all":{"P0_mean_u":p0,"P_load_mean_u":pl,"P_sat_mean_u":ps,"P0_rank_cv":float(agg.loc[agg.placement=="P0","rank_cv"].iloc[0]),"P_load_rank_cv":float(agg.loc[agg.placement=="P_load","rank_cv"].iloc[0]),"P_sat_rank_cv":float(agg.loc[agg.placement=="P_sat","rank_cv"].iloc[0]),"P_load_max_mean_relative_improvement":load_improvement,"P_sat_saturation_relative_change":sat_change},"current_global":{m:{k:float(cv.loc[m,k]) for k in ["mean_u","median_u","p90_u","p_u4","p_u_ge3","rank_load_cv","rank_load_max_mean"]} for m in cv.index},"figures":fig_paths}
    write_json(args.output/"summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
