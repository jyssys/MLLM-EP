#!/usr/bin/env python3
"""Trace-driven critical-rank coalescing analysis.

This analysis never changes a route.  It joins the measured EP8 route capture
to the read-only CUDA-event trace and reports an explicit oracle estimate.  A
missing hidden/output sample is reported as scarcity rather than replaced by
synthetic similarity.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def schedule_index(result: Path) -> dict[int, dict[str, Any]]:
    rows = read_json(result / "schedule.json", [])
    return {int(x["wave"]): x for x in rows}


def load_routes(result: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted((result / "raw_routes").glob("route_*.npz")):
        try:
            z = np.load(p, allow_pickle=False)
            match = re.search(r"_ep(-?\d+)", p.stem)
            out.append({"path": p, "z": z, "wave": int(z["wave"]), "layer": int(z["layer"]),
                        "ep_rank": int(match.group(1)) if match else -1})
        except Exception:
            continue
    return out


def load_timing(result: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted((result / "timing_raw").glob("rank*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                x = json.loads(line)
                x["timing_file"] = p.name
                rows.append(x)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def route_features(items: list[dict[str, Any]], sched: dict[int, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    inv: list[dict[str, Any]] = []
    tok: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    # A replicated measurement request is intentionally present on all four
    # DP engines.  dp0 is the canonical route view; the other files are kept
    # as a replication audit, not summed as four independent user requests.
    # Prefer one non-overlapping worker view (DP0/EP0) when available.  The
    # capture names include EP rank because two TP workers share each DP rank.
    # Keep both TP shards of canonical DP0; route rows are concatenated below
    # so the invocation metrics represent the full sequence rather than one
    # sequence-parallel half.
    dp0 = [x for x in items if "_dp0" in x["path"].stem]
    if dp0:
        items = dp0
    for item in items:
        z = item["z"]; wave, layer = item["wave"], item["layer"]
        ids = np.asarray(z["topk_ids"], dtype=np.int64)
        weights = np.asarray(z.get("topk_weights", np.zeros_like(ids, dtype=np.float32)))
        token_ids = np.asarray(z.get("token_ids", np.full(ids.shape[0], -1)), dtype=np.int64)
        token_positions = np.asarray(z.get("token_positions", np.arange(ids.shape[0])), dtype=np.int64)
        modality = np.asarray(z.get("modality", token_ids == 151655), dtype=np.int8)
        exp = ids.reshape(-1)
        mods = np.repeat(modality, ids.shape[1])
        ranks = exp // 16
        eh = np.bincount(exp[(exp >= 0) & (exp < 128)], minlength=128).astype(float)
        rh = np.bincount(ranks[(ranks >= 0) & (ranks < 8)], minlength=8).astype(float)
        vh = np.bincount(ranks[(ranks >= 0) & (ranks < 8) & (mods == 1)], minlength=8).astype(float)
        th = np.bincount(ranks[(ranks >= 0) & (ranks < 8) & (mods == 0)], minlength=8).astype(float)
        active = eh[eh > 0]
        total = float(eh.sum()); mean = float(rh.mean())
        entropy = float(-np.sum((active / max(total, 1)) * np.log((active / max(total, 1)) + 1e-12))) if active.size else 0.0
        rec = sched.get(wave, {})
        row = {
            "wave": wave, "layer": layer, "request_id": rec.get("request_id", f"wave{wave}"),
            "rep": rec.get("rep", -1), "category": rec.get("category", "unknown"),
            "image_count": rec.get("image_count", -1), "prompt_tokens": rec.get("prompt_tokens", len(token_ids)),
            "vision_tokens": int(modality.sum()), "text_tokens": int((modality == 0).sum()),
            "token_rows": int(ids.shape[0]), "total_assignments": total,
            "active_experts": int((eh > 0).sum()), "effective_experts": float(np.exp(entropy)),
            "expert_entropy": entropy, "expert_hhi": float(np.sum((eh / max(total, 1)) ** 2)),
            "rank_max_assignments": float(rh.max()), "rank_mean_assignments": mean,
            "rank_ratio": float(rh.max() / max(mean, 1e-9)), "rank_cv": float(rh.std() / max(mean, 1e-9)),
            "critical_rank": int(np.argmax(rh)), "critical_excess": float(rh.max() - mean),
            "vision_rank_max": float(vh.max()), "vision_rank_mean": float(vh.mean()),
            "vision_critical_excess": float(vh[int(np.argmax(rh))] - vh.mean()),
            "text_rank_max": float(th.max()), "text_rank_mean": float(th.mean()),
            "text_critical_excess": float(th[int(np.argmax(rh))] - th.mean()),
            "expert_histogram": json.dumps(eh.astype(int).tolist()),
            "rank_histogram": json.dumps(rh.astype(int).tolist()),
            "vision_rank_histogram": json.dumps(vh.astype(int).tolist()),
            "text_rank_histogram": json.dumps(th.astype(int).tolist()),
            "route_file": str(item["path"]),
        }
        inv.append(row)
        for t in range(ids.shape[0]):
            tok.append({"wave": wave, "layer": layer, "token_index": int(token_positions[t]), "token_id": int(token_ids[t]),
                        "modality": "Vision" if modality[t] else "Text", "topk_expert_ids": json.dumps(ids[t].tolist()),
                        "topk_dest_ranks": json.dumps((ids[t] // 16).tolist()),
                        "topk_weights": json.dumps(weights[t].tolist())})
        if "hidden_states_fp16" in z.files and "hidden_positions" in z.files:
            pos = np.asarray(z["hidden_positions"], dtype=int)
            hs = np.asarray(z["hidden_states_fp16"], dtype=np.float32)
            for j, t in enumerate(pos.tolist()):
                for slot, expert in enumerate(ids[t].tolist()):
                    hidden.append({"wave": wave, "layer": layer, "token_index": int(token_positions[t]), "expert_id": int(expert),
                                   "dest_rank": int(expert // 16), "router_weight": float(weights[t, slot]),
                                   "hidden": hs[j].astype(np.float32)})
    if not inv:
        return pd.DataFrame(), pd.DataFrame(tok), hidden
    # Aggregate the two TP sequence shards into one wave/layer invocation.
    # Histogram columns are exact integer assignment counts; all derived
    # imbalance statistics are recomputed from the summed histograms.
    merged: list[dict[str, Any]] = []
    frame = pd.DataFrame(inv)
    for (wave, layer), g in frame.groupby(["wave", "layer"], sort=True):
        row = g.iloc[0].to_dict()
        eh = np.sum(np.stack([np.asarray(json.loads(x), dtype=float) for x in g.expert_histogram]), axis=0)
        rh = np.sum(np.stack([np.asarray(json.loads(x), dtype=float) for x in g.rank_histogram]), axis=0)
        vh = np.sum(np.stack([np.asarray(json.loads(x), dtype=float) for x in g.vision_rank_histogram]), axis=0)
        th = np.sum(np.stack([np.asarray(json.loads(x), dtype=float) for x in g.text_rank_histogram]), axis=0)
        total = float(eh.sum()); mean = float(rh.mean()); active = eh[eh > 0]
        entropy = float(-np.sum((active / max(total, 1)) * np.log((active / max(total, 1)) + 1e-12))) if active.size else 0.0
        crit = int(np.argmax(rh))
        row.update({"wave": int(wave), "layer": int(layer),
                    "vision_tokens": int(g.vision_tokens.sum()), "text_tokens": int(g.text_tokens.sum()),
                    "token_rows": int(g.token_rows.sum()), "total_assignments": total,
                    "active_experts": int((eh > 0).sum()), "effective_experts": float(np.exp(entropy)),
                    "expert_entropy": entropy, "expert_hhi": float(np.sum((eh / max(total, 1)) ** 2)),
                    "rank_max_assignments": float(rh.max()), "rank_mean_assignments": mean,
                    "rank_ratio": float(rh.max() / max(mean, 1e-9)), "rank_cv": float(rh.std() / max(mean, 1e-9)),
                    "critical_rank": crit, "critical_excess": float(rh.max() - mean),
                    "vision_rank_max": float(vh.max()), "vision_rank_mean": float(vh.mean()),
                    "vision_critical_excess": float(vh[crit] - vh.mean()),
                    "text_rank_max": float(th.max()), "text_rank_mean": float(th.mean()),
                    "text_critical_excess": float(th[crit] - th.mean()),
                    "expert_histogram": json.dumps(eh.astype(int).tolist()),
                    "rank_histogram": json.dumps(rh.astype(int).tolist()),
                    "vision_rank_histogram": json.dumps(vh.astype(int).tolist()),
                    "text_rank_histogram": json.dumps(th.astype(int).tolist()),
                    "route_file": ";".join(g.route_file.astype(str).tolist())})
        merged.append(row)
    return pd.DataFrame(merged), pd.DataFrame(tok), hidden


def attach_timing(inv: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    if inv.empty or timing.empty:
        return inv
    rows = []
    # Raw timing is per EP rank.  Select the largest local call per rank and
    # then aggregate eight ranks for each measured wave/layer.
    for (wave, layer, ep), g in timing.groupby(["wave", "layer", "ep_rank"], dropna=False):
        size_col = g.get("local_rows", g.get("dispatched_rows", pd.Series(0, index=g.index))).astype(float)
        x = g.loc[[size_col.idxmax()]].iloc[0]
        def val(k: str) -> float:
            q = x.get(k, {}) or {}; return float(q.get("ms", np.nan)) if isinstance(q, dict) else float("nan")
        rows.append({"wave": int(wave), "layer": int(layer), "ep_rank": int(ep), "assignments": float(x.get("total_assignments", 0)),
                     "expert_ms": val("expert"), "dispatch_ms": val("dispatch"), "combine_ms": val("combine"),
                     "expert_hist": x.get("expert_histogram", [])})
    if not rows: return inv
    t = pd.DataFrame(rows); agg = []
    for (wave, layer), g in t.groupby(["wave", "layer"]):
        loads = g.assignments.to_numpy(float); ex = g.expert_ms.to_numpy(float)
        ds = g.dispatch_ms.to_numpy(float); co = g.combine_ms.to_numpy(float)
        agg.append({"wave": int(wave), "layer": int(layer), "timing_ep_ranks": int(len(g)),
                    "actual_rank_max_assignments": float(np.nanmax(loads)), "actual_rank_mean_assignments": float(np.nanmean(loads)),
                    "actual_rank_ratio": float(np.nanmax(loads) / max(np.nanmean(loads), 1e-9)),
                    "expert_max_ms": float(np.nanmax(ex)), "expert_mean_ms": float(np.nanmean(ex)),
                    "expert_max_mean_ratio": float(np.nanmax(ex) / max(np.nanmean(ex), 1e-9)),
                    "dispatch_max_ms": float(np.nanmax(ds)), "dispatch_mean_ms": float(np.nanmean(ds)),
                    "combine_max_ms": float(np.nanmax(co)), "combine_mean_ms": float(np.nanmean(co)),
                    "critical_rank_timing": int(g.iloc[int(np.nanargmax(ex))].ep_rank)})
    return inv.merge(pd.DataFrame(agg), on=["wave", "layer"], how="left")


def pair_candidates(hidden: list[dict[str, Any]], inv: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not hidden:
        return pd.DataFrame(), {"hidden_sample_assignments": 0, "pair_count": 0, "status": "NO_HIDDEN_SAMPLES"}
    rows = []
    hdf = pd.DataFrame(hidden)
    crit = {(int(r.wave), int(r.layer)): int(r.critical_rank) for _, r in inv.iterrows()}
    for (wave, layer, expert), g in hdf.groupby(["wave", "layer", "expert_id"]):
        g = g.reset_index(drop=True)
        x = np.stack(g.hidden.to_numpy()); x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        sim = x @ x.T; used: set[int] = set()
        order = np.dstack(np.unravel_index(np.argsort(-sim.ravel()), sim.shape))[0]
        for a, b in order:
            a, b = int(a), int(b)
            if a >= b or a in used or b in used or float(sim[a, b]) < 0.90: continue
            used |= {a, b}
            rows.append({"wave": int(wave), "layer": int(layer), "expert_id": int(expert), "token_a": int(g.iloc[a].token_index),
                         "token_b": int(g.iloc[b].token_index), "dest_rank_a": int(g.iloc[a].dest_rank), "dest_rank_b": int(g.iloc[b].dest_rank),
                         "cosine": float(sim[a, b]), "critical_pair": int(crit.get((int(wave), int(layer)), -1) in {int(g.iloc[a].dest_rank), int(g.iloc[b].dest_rank)})})
    p = pd.DataFrame(rows)
    summary = {"hidden_sample_assignments": int(len(hdf)), "pair_count": int(len(p)),
               "pair_fraction_cosine_ge_0_90": float(len(p) / max(len(hdf) / 2, 1)),
               "median_cosine": float(p.cosine.median()) if len(p) else None,
               "critical_pair_fraction": float(p.critical_pair.mean()) if len(p) else None,
               "threshold": 0.90}
    return p, summary


def coalescing(inv: pd.DataFrame, pairs: pd.DataFrame, routes: list[dict[str, Any]]) -> pd.DataFrame:
    if inv.empty: return pd.DataFrame()
    rng = np.random.default_rng(20260904); output = []
    pmap = {(int(r.wave), int(r.layer)): r for _, r in pairs.iterrows()} if not pairs.empty else {}
    for _, row in inv.iterrows():
        key = (int(row.wave), int(row.layer))
        parts = [x for x in routes if x["wave"] == key[0] and x["layer"] == key[1] and "_dp0" in x["path"].stem]
        if not parts:
            continue
        parts.sort(key=lambda x: int(x.get("ep_rank", 0)))
        id_parts = []; mod_parts = []; pos_parts = []
        for item in parts:
            z = item["z"]; local_ids = np.asarray(z["topk_ids"], dtype=int)
            local_mod = np.asarray(z.get("modality", np.zeros(local_ids.shape[0], dtype=int)), dtype=int)
            local_pos = np.asarray(z.get("token_positions", np.arange(local_ids.shape[0])), dtype=int)
            valid = local_pos >= 0
            id_parts.append(local_ids[valid]); mod_parts.append(local_mod[valid])
            pos_parts.append(local_pos[valid])
        ids = np.concatenate(id_parts, axis=0).reshape(-1)
        mods = np.repeat(np.concatenate(mod_parts, axis=0), 8)
        full_ids = np.concatenate(id_parts, axis=0)
        row_positions = np.concatenate(pos_parts, axis=0)
        dest = ids // 16; n = len(ids); base_rank = np.bincount(dest, minlength=8).astype(float); old_max = float(base_rank.max())
        eligible = np.flatnonzero(mods == 1)
        pair_rows = pairs[(pairs.wave == key[0]) & (pairs.layer == key[1])] if not pairs.empty else pd.DataFrame()
        remove_orders: dict[str, np.ndarray] = {}
        remove_orders["RANDOM"] = rng.permutation(np.arange(n))
        if len(pair_rows):
            red = []
            for _, q in pair_rows.sort_values("cosine", ascending=False).iterrows():
                rows = np.flatnonzero((row_positions == int(q.token_b)) & (np.concatenate(mod_parts) == 1))
                for r in rows:
                    slots = np.flatnonzero(full_ids[r] == int(q.expert_id))
                    if len(slots):
                        red.append(int(r * 8 + slots[-1])); break
            critical = pair_rows.sort_values(["critical_pair", "cosine"], ascending=[False, False])
            cr = []
            for _, q in critical.iterrows():
                rows = np.flatnonzero((row_positions == int(q.token_b)) & (np.concatenate(mod_parts) == 1))
                for r in rows:
                    slots = np.flatnonzero(full_ids[r] == int(q.expert_id))
                    if len(slots):
                        cr.append(int(r * 8 + slots[-1])); break
            remove_orders["REDUNDANCY_ONLY"] = np.asarray(list(dict.fromkeys(red)), dtype=int)
            remove_orders["CRITICAL_RANK_AWARE"] = np.asarray(list(dict.fromkeys(cr)), dtype=int)
        else:
            remove_orders["REDUNDANCY_ONLY"] = np.asarray([], dtype=int); remove_orders["CRITICAL_RANK_AWARE"] = np.asarray([], dtype=int)
        for budget in (0.05, 0.10, 0.20, 0.30):
            target = int(np.ceil(budget * n))
            for strategy, order in remove_orders.items():
                chosen = order[:min(target, len(order))]
                new = base_rank.copy();
                for d in dest[chosen]: new[int(d)] -= 1
                old_pred = old_max; new_pred = float(new.max())
                output.append({"wave": key[0], "layer": key[1], "request_id": row.request_id, "budget": budget, "strategy": strategy,
                               "base_assignments": n, "removed_assignments": int(len(chosen)), "actual_removed_fraction": float(len(chosen)/max(n,1)),
                               "base_critical_rank": int(np.argmax(base_rank)), "new_critical_rank": int(np.argmax(new)),
                               "critical_rank_load_reduction": float(1-new[int(row.critical_rank)]/max(base_rank[int(row.critical_rank)],1)) if int(row.critical_rank) < 8 else float(1-new_pred/max(old_pred,1)),
                               "max_rank_load_reduction": float(1-new_pred/max(old_pred,1)),
                               "predicted_expert_latency_reduction": float(1-new_pred/max(old_pred,1)),
                               "pair_cosine_median": float(pair_rows.cosine.median()) if len(pair_rows) else np.nan,
                               "selected_critical_pair_fraction": float(pair_rows.critical_pair.mean()) if len(pair_rows) else np.nan,
                               "quality_error_proxy_1_minus_cosine": float(1-pair_rows.cosine.median()) if len(pair_rows) else np.nan})
    return pd.DataFrame(output)


def matched_pair_budget(inv: pd.DataFrame, pairs: pd.DataFrame,
                        routes: list[dict[str, Any]]) -> pd.DataFrame:
    """Compare strategies at the same *available-pair* budget.

    The requested 5--30% assignment budgets can exceed the sampled hidden
    pair pool.  In that case the two similarity strategies are both capped
    and an apparent tie is not informative.  This diagnostic keeps the
    number of selected eligible pairs fixed (25/50/75/100% of each
    invocation's available pool), without changing any route.
    """
    if inv.empty or pairs.empty:
        return pd.DataFrame()
    out: list[dict[str, Any]] = []
    for _, row in inv.iterrows():
        key = (int(row.wave), int(row.layer))
        pair_rows = pairs[(pairs.wave == key[0]) & (pairs.layer == key[1])]
        if pair_rows.empty:
            continue
        parts = [x for x in routes if x["wave"] == key[0] and x["layer"] == key[1]
                 and "_dp0" in x["path"].stem]
        if not parts:
            continue
        parts.sort(key=lambda x: int(x.get("ep_rank", 0)))
        id_parts = []; mod_parts = []; pos_parts = []
        for item in parts:
            z = item["z"]
            local_ids = np.asarray(z["topk_ids"], dtype=int)
            local_mod = np.asarray(z.get("modality", np.zeros(local_ids.shape[0], dtype=int)), dtype=int)
            local_pos = np.asarray(z.get("token_positions", np.arange(local_ids.shape[0])), dtype=int)
            valid = local_pos >= 0
            id_parts.append(local_ids[valid]); mod_parts.append(local_mod[valid]); pos_parts.append(local_pos[valid])
        full_ids = np.concatenate(id_parts, axis=0)
        full_mod = np.concatenate(mod_parts, axis=0)
        full_pos = np.concatenate(pos_parts, axis=0)
        dest = full_ids.reshape(-1) // 16
        base_rank = np.bincount(dest, minlength=8).astype(float)
        old_max = float(base_rank.max())
        pair_slots: list[int] = []
        for _, q in pair_rows.iterrows():
            rows = np.flatnonzero((full_pos == int(q.token_b)) & (full_mod == 1))
            slot = None
            for r in rows:
                hits = np.flatnonzero(full_ids[r] == int(q.expert_id))
                if len(hits):
                    slot = int(r * 8 + hits[-1]); break
            if slot is not None:
                pair_slots.append(slot)
        if not pair_slots:
            continue
        # Pair rows and slots remain aligned; preserve the deterministic
        # ordering for the two policy variants and add a matched random arm.
        available = len(pair_slots)
        order = pair_rows.iloc[:available].copy()
        order["slot"] = pair_slots
        orders = {
            "RANDOM_MATCHED": np.random.default_rng(20260904 + key[0] * 97 + key[1]).permutation(available),
            "REDUNDANCY_ONLY": np.argsort(-order.cosine.to_numpy(float), kind="stable"),
            "CRITICAL_RANK_AWARE": np.lexsort((-order.cosine.to_numpy(float), -order.critical_pair.to_numpy(int))),
        }
        for fraction in (0.25, 0.50, 0.75, 1.0):
            k = max(1, int(np.ceil(fraction * available)))
            for strategy, indices in orders.items():
                chosen_rows = order.iloc[np.asarray(indices[:k], dtype=int)]
                chosen_slots = chosen_rows.slot.to_numpy(dtype=int)
                new_rank = base_rank.copy()
                for d in dest[chosen_slots]:
                    new_rank[int(d)] -= 1
                out.append({
                    "wave": key[0], "layer": key[1], "request_id": row.request_id,
                    "pair_budget_fraction": fraction, "strategy": strategy,
                    "available_pairs": available, "removed_assignments": int(len(chosen_slots)),
                    "actual_removed_fraction": float(len(chosen_slots) / max(len(dest), 1)),
                    "max_rank_load_reduction": float(1 - new_rank.max() / max(old_max, 1)),
                    "critical_pair_fraction": float(chosen_rows.critical_pair.mean()),
                    "pair_cosine_median": float(chosen_rows.cosine.median()),
                })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--trace", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); sched = schedule_index(args.trace); routes = load_routes(args.trace)
    inv, tok, hidden = route_features(routes, sched); timing = load_timing(args.trace); inv = attach_timing(inv, timing)
    args.output.mkdir(parents=True, exist_ok=True)
    tok.to_csv(args.output / "per_token_routes.csv.gz", index=False, compression="gzip"); inv.to_csv(args.output / "invocation_features.csv", index=False)
    if not timing.empty: timing.to_csv(args.output / "timing_raw_flat.csv", index=False)
    pairs, ps = pair_candidates(hidden, inv); pairs.to_csv(args.output / "hidden_similarity_pairs.csv", index=False)
    coal = coalescing(inv, pairs, routes); coal.to_csv(args.output / "coalescing_results.csv", index=False)
    matched = matched_pair_budget(inv, pairs, routes)
    matched.to_csv(args.output / "matched_pair_budget_results.csv", index=False)
    # Figures are intentionally simple and trace-driven.
    fig, ax = plt.subplots(figsize=(7,4));
    if not coal.empty:
        q = coal.groupby(["budget","strategy"]).predicted_expert_latency_reduction.median().unstack(); q.plot(ax=ax, marker="o")
    ax.set(xlabel="coalescing budget", ylabel="predicted critical expert latency reduction"); fig.tight_layout(); fig.savefig(args.output/"coalescing_gain.png", dpi=160); plt.close(fig)
    if not inv.empty:
        fig, ax = plt.subplots(figsize=(7,4)); ax.scatter(inv.rank_ratio, inv.get("expert_max_mean_ratio", pd.Series(np.nan,index=inv.index)), c=inv.layer, s=14, alpha=.7); ax.set(xlabel="route rank max/mean",ylabel="measured expert CUDA max/mean"); fig.tight_layout(); fig.savefig(args.output/"straggler_scatter.png", dpi=160); plt.close(fig)
    ratios = inv.get("expert_max_mean_ratio", pd.Series(dtype=float)).dropna().to_numpy(float)
    gate = {"trace_routes": int(len(routes)), "invocations": int(len(inv)), "timing_rows": int(len(timing)),
            "hidden_pair_summary": ps, "expert_output_similarity": "NOT_CAPTURED_EP8", "stage_b_straggler": {
                "median": float(np.median(ratios)) if len(ratios) else None, "p90": float(np.quantile(ratios,.9)) if len(ratios) else None,
                "max": float(np.max(ratios)) if len(ratios) else None}, "trace_portable": True,
            "measurement_mode": "same real request submitted to all four DP engines to ensure multimodal EP collective participation"}
    (args.output / "analysis_summary.json").write_text(json.dumps(gate, indent=2)+"\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__": main()
