#!/usr/bin/env python3
"""Analyze temporal routing/branch persistence and conservative EP2 oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


MASK_ID = 156895


def rel_l2(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.vector_norm((a - b).float(), dim=dim) / torch.linalg.vector_norm(
        b.float(), dim=dim
    ).clamp_min(1e-12)


def destination_code(ids: torch.Tensor, experts_per_rank: int = 32) -> torch.Tensor:
    ranks = ids // experts_per_rank
    return ((ranks == 0).any(dim=-1).to(torch.int64) +
            2 * (ranks == 1).any(dim=-1).to(torch.int64))


def load_artifacts(root: Path):
    paths = sorted(root.glob("temporal/*/temporal_*.pt"))
    paths = [p for p in paths if not p.name.endswith("summary.pt")]
    if not paths:
        raise FileNotFoundError(f"no temporal artifacts below {root}")
    return [(p, torch.load(p, map_location="cpu", weights_only=False)) for p in paths]


def stage_inputs(root: Path, prior_scaling: Path):
    clean = []
    instrumented = []
    for path in sorted((root / "stage_timing").glob("*/*_rank0.json")):
        data = json.loads(path.read_text())
        (instrumented if data.get("instrument") else clean).append(data)
    scaling = pd.read_csv(prior_scaling)
    scaling = scaling[scaling.backend == "deepep_high_throughput"].sort_values("local_m")
    return clean, instrumented, scaling


def interpolate_stage(scaling: pd.DataFrame, effective_m: float) -> float:
    if effective_m <= 0:
        return 0.0
    m = scaling.local_m.to_numpy(float)
    routed = scaling[["dispatch_ms", "expert_ms", "combine_ms"]].sum(axis=1).to_numpy(float)
    if effective_m < m.min():
        return float(routed[0])
    return float(np.interp(effective_m, m, routed))


def analyze_one(path: Path, art: dict):
    ids = art["top_ids"]
    weights = art["top_weights"]
    logits = art["router_logits"].float()
    calls, layers, physical_m, top_k = ids.shape
    batch = int(art.get("batch_size", 1))
    prompt = int(art["prompt_length"])
    gen = int(art["gen_length"])
    seq = prompt + gen
    gen_rows = torch.cat([
        torch.arange(b * seq + prompt, b * seq + prompt + gen)
        for b in range(batch)
    ])
    metadata = art["call_metadata"]
    token_rows = []
    branch_frames = []
    cell_rows = []

    branch_outputs = {int(k): v for k, v in art.get("branch_outputs", {}).items()}
    hidden = {int(k): v for k, v in art.get("hidden", {}).items()}
    moe_output = {int(k): v for k, v in art.get("moe_output", {}).items()}

    for t in range(1, calls):
        live = (art["tokens_before"][t].reshape(-1) == MASK_ID)
        previous_live = (art["tokens_before"][t - 1].reshape(-1) == MASK_ID)
        newly_decoded = previous_live & ~live
        epoch_fresh_generation = (
            torch.ones_like(live) if t % 5 == 0 else (live | newly_decoded)
        )
        for layer in range(layers):
            prev_ids = ids[t - 1, layer]
            cur_ids = ids[t, layer]
            ordered = (prev_ids == cur_ids).all(dim=-1)
            set_same = (prev_ids.sort(dim=-1).values == cur_ids.sort(dim=-1).values).all(dim=-1)
            dest_same = destination_code(prev_ids) == destination_code(cur_ids)
            intersect = (cur_ids.unsqueeze(-1) == prev_ids.unsqueeze(-2)).any(dim=-1)
            route_weight_l1 = (weights[t, layer] - weights[t - 1, layer]).abs().sum(dim=-1)
            router_change = rel_l2(logits[t, layer], logits[t - 1, layer])
            hidden_change = None
            moe_change = None
            if layer in hidden:
                hidden_change = rel_l2(hidden[layer][t], hidden[layer][t - 1])
                moe_change = rel_l2(moe_output[layer][t], moe_output[layer][t - 1])

            for pos in range(physical_m):
                local_pos = pos % seq
                is_gen = local_pos >= prompt
                gen_index = (pos // seq) * gen + (local_pos - prompt) if is_gen else -1
                token_rows.append({
                    "run_id": art["run_id"],
                    "transition": t,
                    "layer": layer,
                    "position": pos,
                    "is_generation": is_gen,
                    "is_live": bool(live[gen_index]) if is_gen else False,
                    "is_newly_decoded": bool(newly_decoded[gen_index]) if is_gen else False,
                    "is_epoch_fresh": bool(epoch_fresh_generation[gen_index]) if is_gen else False,
                    "ordered_route_same": bool(ordered[pos]),
                    "route_set_same": bool(set_same[pos]),
                    "destination_set_same": bool(dest_same[pos]),
                    "branch_persistence": float(intersect[pos].float().mean()),
                    "router_rel_l2": float(router_change[pos]),
                    "topk_weight_l1": float(route_weight_l1[pos]),
                    "hidden_rel_l2": float(hidden_change[pos]) if hidden_change is not None else np.nan,
                    "moe_output_rel_l2": float(moe_change[pos]) if moe_change is not None else np.nan,
                })

            cell = {
                "run_id": art["run_id"],
                "transition": t,
                "layer": layer,
                "masked_before": int(metadata[t]["masked_before"]),
                "masked_after": int(metadata[t]["masked_after"]),
                "physical_m": physical_m,
                "ordered_route_same": float(ordered.float().mean()),
                "route_set_same": float(set_same.float().mean()),
                "destination_set_same": float(dest_same.float().mean()),
                "branch_persistence": float(intersect.float().mean()),
                "router_rel_l2": float(router_change.mean()),
                "topk_weight_l1": float(route_weight_l1.mean()),
            }

            if layer in branch_outputs:
                prev_b = branch_outputs[layer][t - 1].float()
                cur_b = branch_outputs[layer][t].float()
                prev_gen_ids = prev_ids.index_select(0, gen_rows)
                cur_gen_ids = cur_ids.index_select(0, gen_rows)
                prev_w = weights[t - 1, layer].index_select(0, gen_rows).float()
                cur_w = weights[t, layer].index_select(0, gen_rows).float()
                current_exact = (cur_b * cur_w.unsqueeze(-1)).sum(dim=1)
                matches = cur_gen_ids.unsqueeze(-1) == prev_gen_ids.unsqueeze(-2)
                present = matches.any(dim=-1)
                previous_branch = matches.to(torch.int64).argmax(dim=-1)
                gather_index = previous_branch.unsqueeze(-1).expand(-1, -1, cur_b.shape[-1])
                old = prev_b.gather(dim=1, index=gather_index)
                old_w = prev_w.gather(dim=1, index=previous_branch)
                branch_rel = rel_l2(old, cur_b)
                weighted_rel = rel_l2(old * old_w.unsqueeze(-1), cur_b * cur_w.unsqueeze(-1))
                weight_change = (cur_w - old_w).abs()
                live_matrix = live.unsqueeze(-1).expand_as(present)
                fresh_matrix = epoch_fresh_generation.unsqueeze(-1).expand_as(present)
                eligible_matrix = present
                flat = eligible_matrix.reshape(-1)
                if flat.any():
                    positions = torch.arange(batch * gen).unsqueeze(-1).expand(-1, top_k)
                    branch_frames.append(pd.DataFrame({
                        "run_id": art["run_id"],
                        "transition": t,
                        "layer": layer,
                        "generation_position": positions.reshape(-1)[flat].numpy(),
                        "expert": cur_gen_ids.reshape(-1)[flat].numpy(),
                        "is_live": live_matrix.reshape(-1)[flat].numpy(),
                        "is_epoch_fresh": fresh_matrix.reshape(-1)[flat].numpy(),
                        "branch_rel_l2": branch_rel.reshape(-1)[flat].numpy(),
                        "weighted_branch_rel_l2": weighted_rel.reshape(-1)[flat].numpy(),
                        "router_weight_change": weight_change.reshape(-1)[flat].numpy(),
                    }))
                counts = {}
                for threshold in (0.0, 0.001, 0.01, 0.05):
                    # Candidate semantics reuse E_e(h) but still recompute and
                    # apply the current router weight.  Therefore cache safety
                    # is determined by the unweighted expert-output change;
                    # weighted_rel is recorded separately as a diagnostic.
                    safe_numerical = branch_rel == 0.0 if threshold == 0.0 else branch_rel <= threshold
                    safe = present & fresh_matrix & safe_numerical
                    counts[threshold] = int(safe.sum())
                    approx_branches = torch.where(safe.unsqueeze(-1), old, cur_b)
                    approx = (approx_branches * cur_w.unsqueeze(-1)).sum(dim=1)
                    fresh_approx = approx[epoch_fresh_generation]
                    fresh_exact = current_exact[epoch_fresh_generation]
                    error = (float(rel_l2(fresh_approx.reshape(1, -1), fresh_exact.reshape(1, -1)).item())
                             if fresh_exact.numel() else 0.0)
                    key = "exact" if threshold == 0.0 else f"{threshold:g}"
                    cell[f"epoch_fresh_reusable_branches_{key}"] = counts[threshold]
                    cell[f"epoch_fresh_combined_error_{key}"] = error
                cell["epoch_fresh_generation_branches"] = int(epoch_fresh_generation.sum()) * top_k
                cell["persistent_generation_branches"] = int(eligible_matrix.sum())
            cell_rows.append(cell)

    branches = pd.concat(branch_frames, ignore_index=True) if branch_frames else pd.DataFrame()
    return pd.DataFrame(token_rows), branches, pd.DataFrame(cell_rows)


def plot_heatmap(frame: pd.DataFrame, value: str, output: Path, title: str):
    pivot = frame.pivot_table(index="layer", columns="transition", values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("denoising transition")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--prior-scaling", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.result_root / "analysis"
    plots = analysis / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    token_parts, branch_parts, cell_parts = [], [], []
    artifacts = load_artifacts(args.result_root)
    for path, art in artifacts:
        token, branch, cell = analyze_one(path, art)
        token_parts.append(token)
        branch_parts.append(branch)
        cell_parts.append(cell)
    token = pd.concat(token_parts, ignore_index=True)
    branch = pd.concat(branch_parts, ignore_index=True)
    cell = pd.concat(cell_parts, ignore_index=True)
    token.to_csv(analysis / "transition_token_metrics.csv", index=False)
    branch.to_csv(analysis / "branch_metrics.csv", index=False)
    cell.to_csv(analysis / "layer_iteration_metrics.csv", index=False)

    clean, instrumented, scaling = stage_inputs(args.result_root, args.prior_scaling)
    if clean:
        clean_ms = float(np.median([x["request_ms"] for x in clean]))
    else:
        # Pinned prior-run median; replaced once fresh clean runs exist.
        clean_ms = 501.98432989418507
    critical_stage = {}
    for name in ("router_linear", "dispatch", "expert", "combine", "moe_total"):
        values = [max(
            json.loads(p.read_text())["stage_sums_ms"].get(name, 0.0)
            for p in sorted(run.glob("*_rank*.json"))
        ) for run in sorted((args.result_root / "stage_timing").glob("instrumented_*"))]
        if values:
            critical_stage[name] = float(np.median(values))
    base_m = int(artifacts[0][1]["top_ids"].shape[2])
    base_routed_ms = interpolate_stage(scaling, base_m)

    physical_assignments = 0
    epoch_assignments = 0
    epoch_by_run = {}
    for _, art in artifacts:
        calls, layers, m, k = art["top_ids"].shape
        batch = int(art.get("batch_size", 1))
        gen = int(art["gen_length"])
        prompt = int(art["prompt_length"])
        run_phys = calls * layers * m * k
        run_epoch = 0
        for t, meta in enumerate(art["call_metadata"]):
            if t == 0 or t % 5 == 0:
                fresh_tokens = m
            else:
                newly_decoded = max(
                    0,
                    int(art["call_metadata"][t - 1]["masked_before"])
                    - int(meta["masked_before"]),
                )
                fresh_tokens = int(meta["masked_before"]) + newly_decoded
            run_epoch += layers * fresh_tokens * k
        physical_assignments += run_phys
        epoch_assignments += run_epoch
        epoch_by_run[art["run_id"]] = (run_phys, run_epoch)

    candidate_rows = []
    stock_to_epoch_work = 1 - epoch_assignments / physical_assignments
    candidate_rows.append({
        "candidate": "Epoch-equivalent live/new/refresh fresh lane",
        "scope": "already claimed by Epoch; excluded from novelty",
        "removable_assignment_fraction_stock": stock_to_epoch_work,
        "projected_e2e_upper_bound_pct": np.nan,
        "correctness_risk": "bounded decoded-cache approximation",
    })

    router_e2e = 100 * critical_stage.get("router_linear", 0.0) / clean_ms
    candidate_rows.append({
        "candidate": "perfect router recomputation elimination",
        "scope": "post-Epoch upper bound; no executable exact predictor",
        "removable_assignment_fraction_stock": 0.0,
        "projected_e2e_upper_bound_pct": router_e2e,
        "correctness_risk": "high: live gate values are iteration-clock state",
    })

    # The profiled get_dispatch_layout kernel is ~4 us/layer at M=64.  Reusing
    # every stable plan gives a deliberately optimistic all-in upper bound.
    stable_route = float(token.ordered_route_same.mean())
    calls_total = sum(int(a["top_ids"].shape[0]) for _, a in artifacts)
    layers = int(artifacts[0][1]["top_ids"].shape[1])
    layout_max_ms = calls_total * layers * 0.004 * stable_route
    candidate_rows.append({
        "candidate": "unchanged routing/layout delta update",
        "scope": "post-Epoch structural metadata only",
        "removable_assignment_fraction_stock": 0.0,
        "projected_e2e_upper_bound_pct": 100 * layout_max_ms / (clean_ms * len(artifacts)),
        "correctness_risk": "low for metadata; gate/top-k still recomputed",
    })

    for key, label in (("exact", "bit-identical stable branch reuse"),
                       ("0.001", "0.1%-expert-output-change branch reuse"),
                       ("0.01", "1%-expert-output-change branch reuse"),
                       ("0.05", "5%-expert-output-change branch reuse")):
        col = f"epoch_fresh_reusable_branches_{key}"
        if col not in cell:
            continue
        subset = cell.dropna(subset=[col]).copy()
        removable = int(subset[col].sum())
        live_branches = int(subset["epoch_fresh_generation_branches"].sum())
        # Measured-cost calibration: per cell compare Epoch fresh M with M
        # after removing only future-known reusable live branches.
        saving = 0.0
        epoch_cost = 0.0
        for _, row in subset.iterrows():
            # Prompt/stable decoded positions are omitted here. This isolates
            # residual reuse inside Epoch's genuinely fresh live lane.
            generation_fresh_m = float(row["epoch_fresh_generation_branches"]) / 8.0
            # Non-generation rows are not covered by the branch capture. On a
            # refresh iteration they remain present and unmodified.
            prompt_fresh_m = max(0.0, float(row["physical_m"]) - generation_fresh_m) if int(row["transition"]) % 5 == 0 else 0.0
            fresh_m = generation_fresh_m + prompt_fresh_m
            remaining_m = max(0.0, fresh_m - float(row[col]) / 8.0)
            before = interpolate_stage(scaling, fresh_m)
            after = interpolate_stage(scaling, remaining_m)
            epoch_cost += before
            saving += max(0.0, before - after)
        e2e = 100 * saving / (clean_ms * len(artifacts))
        candidate_rows.append({
            "candidate": label,
            "scope": "post-Epoch fresh-lane residual, perfect-future oracle",
            "removable_assignment_fraction_stock": removable / physical_assignments,
            "removable_fraction_of_observed_fresh_branches": removable / max(live_branches, 1),
            "projected_moe_local_gain_pct": 100 * saving / max(epoch_cost, 1e-12),
            "projected_e2e_upper_bound_pct": e2e,
            "combined_error_median": float(subset[f"epoch_fresh_combined_error_{key}"].median()),
            "combined_error_p95": float(subset[f"epoch_fresh_combined_error_{key}"].quantile(0.95)),
            "correctness_risk": "none observed" if key == "exact" else "approximate; no rollout/benchmark validation",
        })

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(analysis / "candidate_oracles.csv", index=False)

    summary = {
        "artifacts": [str(p) for p, _ in artifacts],
        "runs": len(artifacts),
        "clean_request_median_ms": clean_ms,
        "instrumented_stage_critical_ms": critical_stage,
        "instrumented_observer_overhead_pct": (
            100 * (np.median([x["request_ms"] for x in instrumented]) / clean_ms - 1)
            if instrumented else None
        ),
        "physical_assignments": physical_assignments,
        "epoch_fresh_assignments_m5": epoch_assignments,
        "epoch_assignment_reduction_pct": 100 * stock_to_epoch_work,
        "ordered_route_same_pct": 100 * float(token.ordered_route_same.mean()),
        "route_set_same_pct": 100 * float(token.route_set_same.mean()),
        "destination_set_same_pct": 100 * float(token.destination_set_same.mean()),
        "branch_persistence_pct": 100 * float(token.branch_persistence.mean()),
        "hidden_changed_route_same_pct": 100 * float(((token.hidden_rel_l2 > 0.001) & token.route_set_same).mean()),
        "branch_weighted_rel_l2": {
            "median": float(branch.weighted_branch_rel_l2.median()),
            "p10": float(branch.weighted_branch_rel_l2.quantile(0.1)),
            "p90": float(branch.weighted_branch_rel_l2.quantile(0.9)),
        },
        "branch_unweighted_rel_l2": {
            "median": float(branch.branch_rel_l2.median()),
            "p10": float(branch.branch_rel_l2.quantile(0.1)),
            "p90": float(branch.branch_rel_l2.quantile(0.9)),
        },
        "branch_reconstruction_validation": {
            a["run_id"]: a["branch_validation"] for _, a in artifacts
        },
        "best_post_epoch_candidate_e2e_pct": float(
            candidates[candidates.scope.str.contains("post-Epoch")]["projected_e2e_upper_bound_pct"].max()
        ),
    }
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plot_heatmap(cell, "route_set_same", plots / "routing_stability_heatmap.png",
                 "Top-k set stability across denoising transitions")
    plot_heatmap(cell, "branch_persistence", plots / "branch_persistence_heatmap.png",
                 "Token-expert branch persistence")

    # Live state and physical M.
    meta_rows = []
    for _, art in artifacts:
        for row in art["call_metadata"]:
            meta_rows.append({"run_id": art["run_id"], **row,
                              "physical_m": int(art["top_ids"].shape[2])})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(analysis / "iteration_work.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for run, group in meta.groupby("run_id"):
        ax.plot(group.iteration_id, group.masked_before, marker="o", label=f"{run}: live")
        ax.plot(group.iteration_id, group.physical_m, linestyle="--", alpha=0.6,
                label=f"{run}: physical M")
    ax.set_xlabel("denoising iteration")
    ax.set_ylabel("positions")
    ax.set_title("Semantic live state versus physical MoE M")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / "mask_live_vs_physical_m.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ratio = epoch_assignments / physical_assignments
    ax.bar(["physical executed", "Epoch-equivalent fresh"], [1.0, ratio], color=["#777777", "#377eb8"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("work / stock physical assignments")
    ax.set_title("Logical fresh work versus physical executed work")
    fig.tight_layout()
    fig.savefig(plots / "logical_vs_physical_ratio.png", dpi=180)
    plt.close(fig)

    plot_candidates = candidates.dropna(subset=["projected_e2e_upper_bound_pct"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(plot_candidates.candidate, plot_candidates.projected_e2e_upper_bound_pct, color="#4daf4a")
    ax.axvline(5, color="black", linestyle="--", linewidth=1)
    ax.axvline(10, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("optimistic request E2E upper bound (%)")
    ax.set_title("Residual candidate upper bounds")
    fig.tight_layout()
    fig.savefig(plots / "candidate_projected_e2e.png", dpi=180)
    plt.close(fig)

    removable = candidates.dropna(subset=["removable_assignment_fraction_stock"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(removable.candidate, 100 * removable.removable_assignment_fraction_stock, color="#984ea3")
    ax.set_xlabel("removable assignments / stock physical work (%)")
    ax.set_title("Candidate removable-work upper bounds")
    fig.tight_layout()
    fig.savefig(plots / "candidate_removable_work.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
