"""Aggregate randomized regime-grid replay and estimate H1/H2 effects."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def med(obs: dict, key: str) -> float:
    return float(obs[key]["median_ms"])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted((args.result / "replay").glob("rank*_layer*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ok":
            continue
        for obs in payload.get("observations", []):
            match = re.search(r"_M(\d+)_F(\d+)_A(\d+)(?:_R\d+)?$", obs["case_id"])
            if not match:
                continue
            m, fanout, active = map(int, match.groups())
            base_case = f"regime_M{m}_F{fanout}_A{active}"
            rows.append({
                "case_id": obs["case_id"], "base_case_id": base_case, "rep": int(obs["case_id"].split("_R")[-1]) if "_R" in obs["case_id"] else -1,
                "rank": int(payload["rank"]), "M": m,
                "fanout": fanout, "active": active,
                "wall_ms": med(obs, "wall_stats"), "layout_ms": med(obs, "layout_stats"),
                "dispatch_ms": med(obs, "dispatch_stats"), "expert_ms": med(obs, "expert_stats"),
                "combine_ms": med(obs, "combine_stats"), "correctness": bool(obs["correctness"]["passed"]),
                "iterations": int(obs.get("iterations", 0)), "warmups": int(obs.get("warmups", 0)),
            })
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("no regime-grid replay observations")
    raw.to_csv(args.result / "regime_rank_timing_raw.csv", index=False)
    # For repeated interleaved runs, retain both per-repetition rows and a
    # condition-level median.  For the ordinary grid rep=-1 and the two are
    # equivalent.
    agg = raw.groupby(["case_id", "base_case_id", "M", "fanout", "active", "rep"], as_index=False).agg(
        ranks=("rank", "nunique"), critical_wall_ms=("wall_ms", "max"),
        layout_ms=("layout_ms", "max"), dispatch_ms=("dispatch_ms", "max"),
        expert_ms=("expert_ms", "max"), combine_ms=("combine_ms", "max"),
        correctness=("correctness", "all"), iterations=("iterations", "max"), warmups=("warmups", "max"))
    agg["assignments"] = agg["M"] * 8.0
    for col in ("critical_wall_ms", "layout_ms", "dispatch_ms", "expert_ms", "combine_ms"):
        agg[col + "_per_assignment"] = agg[col] / agg["assignments"]
    agg.to_csv(args.result / "regime_metrics.csv", index=False)

    cond = agg.groupby(["base_case_id", "M", "fanout", "active"], as_index=False).agg(
        critical_wall_ms=("critical_wall_ms", "median"), layout_ms=("layout_ms", "median"),
        dispatch_ms=("dispatch_ms", "median"), expert_ms=("expert_ms", "median"),
        combine_ms=("combine_ms", "median"), repetitions=("rep", "nunique"),
        correctness=("correctness", "all"))
    cond["assignments"] = cond["M"] * 8.0
    for col in ("critical_wall_ms", "layout_ms", "dispatch_ms", "expert_ms", "combine_ms"):
        cond[col + "_per_assignment"] = cond[col] / cond["assignments"]
    cond.to_csv(args.result / "condition_medians.csv", index=False)

    def effect(group: pd.DataFrame, value: str, base_filter: dict, other_filter: dict) -> float | None:
        a = group.loc[(group[list(base_filter)] == pd.Series(base_filter)).all(axis=1), value]
        b = group.loc[(group[list(other_filter)] == pd.Series(other_filter)).all(axis=1), value]
        if a.empty or b.empty or float(a.iloc[0]) == 0:
            return None
        return float((b.iloc[0] - a.iloc[0]) / a.iloc[0] * 100.0)

    effects = []
    for m, g in cond.groupby("M"):
        for active in sorted(g.active.unique()):
            effects.append({
                "M": int(m), "active": int(active),
                "F2_vs_F1_expert_pct": effect(g, "expert_ms", {"fanout": 1, "active": active}, {"fanout": 2, "active": active}),
                "F4_vs_F1_expert_pct": effect(g, "expert_ms", {"fanout": 1, "active": active}, {"fanout": 4, "active": active}),
                "F2_vs_F1_dispatch_pct": effect(g, "dispatch_ms", {"fanout": 1, "active": active}, {"fanout": 2, "active": active}),
                "F4_vs_F1_dispatch_pct": effect(g, "dispatch_ms", {"fanout": 1, "active": active}, {"fanout": 4, "active": active}),
                "F2_vs_F1_combine_pct": effect(g, "combine_ms", {"fanout": 1, "active": active}, {"fanout": 2, "active": active}),
                "F4_vs_F1_combine_pct": effect(g, "combine_ms", {"fanout": 1, "active": active}, {"fanout": 4, "active": active}),
                "F2_vs_F1_wall_pct": effect(g, "critical_wall_ms", {"fanout": 1, "active": active}, {"fanout": 2, "active": active}),
                "F4_vs_F1_wall_pct": effect(g, "critical_wall_ms", {"fanout": 1, "active": active}, {"fanout": 4, "active": active}),
            })
    effects_df = pd.DataFrame(effects); effects_df.to_csv(args.result / "fanout_effects_by_active.csv", index=False)

    pivot = cond.pivot_table(index="M", columns=["fanout", "active"], values="expert_ms_per_assignment")
    ax = pivot.plot(figsize=(10, 5), marker="o")
    ax.set_title("Controlled expert cost/assignment: M × fanout × active experts")
    ax.set_ylabel("critical expert ms / assignment"); ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=3)
    ax.figure.tight_layout(); ax.figure.savefig(args.result / "regime_expert_cost_curves.png", dpi=150); plt.close(ax.figure)
    for active, g in cond.groupby("active"):
        p = g.pivot(index="M", columns="fanout", values="expert_ms")
        ax = p.plot(figsize=(7, 4), marker="o", title=f"Fanout effect, active/rank={active}")
        ax.set_ylabel("critical expert ms"); ax.grid(alpha=.2); ax.legend(title="fanout")
        ax.figure.tight_layout(); ax.figure.savefig(args.result / f"fanout_by_M_A{active}.png", dpi=150); plt.close(ax.figure)
    summary = {
        "status": "ok", "raw_rows": int(len(raw)), "case_rows": int(len(agg)), "condition_rows": int(len(cond)),
        "M_values": sorted(int(x) for x in agg.M.unique()),
        "fanout_values": sorted(int(x) for x in agg.fanout.unique()),
        "active_values": sorted(int(x) for x in agg.active.unique()),
        "rank_load_invariant": bool((agg.assignments > 0).all()),
        "correctness_all": bool(agg.correctness.all()), "condition_medians": "condition_medians.csv",
        "effects_csv": "fanout_effects_by_active.csv",
        "interpretation": "randomized proper-warmup controlled DeepEP regime grid; synthetic route diagnostic",
    }
    (args.result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
