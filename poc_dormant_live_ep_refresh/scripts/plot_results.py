#!/usr/bin/env python3
"""Create compact evidence figures from the derived dormant-live tables."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180)
    plt.close()


census = pd.read_csv(ROOT / "DORMANT_TOKEN_CENSUS.csv")
rows = []
for (task, horizon, state), group in census.groupby(["task", "horizon", "state"]):
    rows.append({"task": task, "horizon": horizon, "state": state,
                 "fraction": 100 * group.token_rows.sum() /
                 census[(census.task == task) & (census.horizon == horizon)].token_rows.sum()})
frame = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, task in zip(axes, ("gsm8k", "humaneval")):
    pivot = frame[frame.task == task].pivot(index="horizon", columns="state", values="fraction")
    pivot[["ACTIVE", "DORMANT", "DEAD"]].plot(kind="bar", stacked=True, ax=ax,
        color=["#4daf4a", "#ffb000", "#999999"])
    ax.set_title(task); ax.set_ylabel("logical token-row share (%)"); ax.legend(fontsize=7)
save("01_dormant_census.png")

dynamics = pd.read_csv(ROOT / "DORMANT_ROUTING_DYNAMICS.csv")
p = dynamics[(dynamics.horizon == 1) & dynamics.state.isin(["ACTIVE", "DORMANT"])]
metrics = ["exact_topk_set_pct", "exact_destination_set_pct"]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, metric, title in zip(axes, metrics, ["Exact top-k set", "Exact destination set"]):
    q = p.pivot(index="task", columns="state", values=metric)
    q[["ACTIVE", "DORMANT"]].plot(kind="bar", ax=ax, color=["#4daf4a", "#ffb000"])
    ax.set_title(title); ax.set_ylabel("lag-1 agreement (%)"); ax.set_xlabel("")
save("02_active_vs_dormant_stability.png")

oracle = pd.read_csv(ROOT / "DORMANT_ORACLES.csv")
q = oracle[(oracle.horizon == 1) & oracle.policy.isin([
    "perfect_remove", "periodic_k2", "periodic_k4", "periodic_k8", "route_trigger_k8"])]
labels = {"perfect_remove": "perfect", "periodic_k2": "K2", "periodic_k4": "K4",
          "periodic_k8": "K8", "route_trigger_k8": "route-K8"}
fig, ax = plt.subplots(figsize=(7, 3.6))
for task, group in q.groupby("task"):
    order = ["perfect_remove", "periodic_k2", "periodic_k4", "periodic_k8", "route_trigger_k8"]
    group = group.set_index("policy").loc[order]
    ax.plot([labels[x] for x in order], group.feasible_e2e_pct, marker="o", label=task)
ax.axhline(8, ls="--", color="black", lw=1, label="HOLD gate")
ax.set_ylabel("feasible post-Epoch E2E oracle (%)"); ax.legend(fontsize=8)
save("03_oracle_gate.png")

causal = pd.read_csv(ROOT / "REFRESH_POLICY_RESULTS.csv")
causal = causal[causal.policy != "baseline"]
fig, ax = plt.subplots(figsize=(7, 4))
for task, group in causal.groupby("task"):
    ax.scatter(group.feasible_e2e_oracle_pct,
               100 * group.final_sequence_exact / group.final_sequence_total,
               label=task, s=45)
    for _, row in group.iterrows():
        ax.annotate(row.policy, (row.feasible_e2e_oracle_pct,
                    100 * row.final_sequence_exact / row.final_sequence_total), fontsize=6)
ax.axvline(8, ls="--", color="black", lw=1)
ax.axhline(100, ls=":", color="black", lw=1)
ax.set_xlabel("feasible routed-EP E2E oracle (%)")
ax.set_ylabel("final-sequence exact rate (%)")
ax.legend(fontsize=8)
save("04_refresh_quality_pareto.png")
