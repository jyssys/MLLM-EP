#!/usr/bin/env python3
"""Mechanical checks for the dormant-live evidence tables."""

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def close(a, b, eps=1e-6):
    assert abs(float(a) - float(b)) <= eps, (a, b)


def main():
    census = pd.read_csv(ROOT / "DORMANT_TOKEN_CENSUS.csv")
    for _, group in census.groupby(["task", "horizon", "phase"]):
        close(group.state_fraction_pct.sum(), 100.0, 2e-5)
        assert set(group.state) == {"ACTIVE", "DORMANT", "DEAD"}

    dynamics = pd.read_csv(ROOT / "DORMANT_ROUTING_DYNAMICS.csv")
    primary = dynamics[dynamics.horizon.eq(1)]
    for task in ("gsm8k", "humaneval"):
        active = primary[(primary.task == task) & (primary.state == "ACTIVE")].iloc[0]
        dormant = primary[(primary.task == task) & (primary.state == "DORMANT")].iloc[0]
        assert dormant.exact_topk_set_pct < active.exact_topk_set_pct
        assert dormant.exact_destination_set_pct < active.exact_destination_set_pct
        assert dormant.routed_output_rel_l2_p50 > active.routed_output_rel_l2_p50

    oracle = pd.read_csv(ROOT / "DORMANT_ORACLES.csv")
    assert (oracle.feasible_e2e_pct <= oracle.optimistic_e2e_pct + 1e-9).all()
    primary = oracle[(oracle.horizon == 1) & (oracle.policy == "perfect_remove")]
    assert len(primary) == 2 and (primary.feasible_e2e_pct < 8.0).all()
    route = oracle[(oracle.horizon == 1) & (oracle.policy == "route_trigger_k8")]
    assert len(route) == 2 and (route.feasible_e2e_pct < 1.0).all()

    causal = pd.read_csv(ROOT / "REFRESH_POLICY_RESULTS.csv")
    changed = causal[causal.policy != "baseline"]
    assert (changed.final_sequence_exact < changed.final_sequence_total).all()
    assert (~changed.nfe_equal).all()

    proxy = pd.read_csv(ROOT / "CLASSIFIER_RESULTS.csv")
    assert (proxy.post_epoch_e2e_pct < 5.0).all()
    print("all dormant-live evidence invariants passed")


if __name__ == "__main__":
    main()
