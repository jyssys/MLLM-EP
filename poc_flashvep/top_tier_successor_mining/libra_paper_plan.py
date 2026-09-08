"""Bounded, deterministic port of Libra Appendix B/C planning semantics.

Not the unavailable official Cython implementation. This computes count/placement
diagnostics, NEVER serving speed or CPU-overlap claims. Underspecified initial
remote ownership uses each expert's linear home; ties use lowest numeric ID.
"""
import numpy as np


def initial_work(demand, placement):
    g, e = demand.shape
    homes = np.arange(e) // (e // g)
    local = np.zeros(g, dtype=np.int64)
    remote = np.zeros((g,e), dtype=np.int64)
    for source in range(g):
        for expert in range(e):
            if placement[expert, source]:
                local[source] += demand[source, expert]
            else:
                remote[homes[expert], expert] += demand[source, expert]
    assert local.sum() + remote.sum() == demand.sum()
    return local, remote


def shard(local, remote, placement, epsilon=0.01):
    """Appendix C Algorithm 2, integer assignment quantities."""
    remote = remote.copy()
    loads = local + remote.sum(axis=1)
    target = loads.mean()
    moves = []
    for _ in range(int(remote.sum()) + 1):
        if loads.max() <= (1+epsilon)*target:
            break
        moved = False
        for source in np.argsort(-loads, kind="stable"):
            if loads[source] <= target:
                continue
            for expert in np.argsort(-remote[source], kind="stable"):
                if remote[source, expert] == 0:
                    break
                candidates = np.flatnonzero(placement[expert])
                candidates = candidates[candidates != source]
                if not len(candidates):
                    continue
                dest = candidates[np.argmin(loads[candidates])]
                count = min(int(remote[source,expert]), int(np.floor(target-loads[dest])))
                if count <= 0:
                    continue
                remote[source,expert] -= count
                remote[dest,expert] += count
                loads[source] -= count
                loads[dest] += count
                moves.append((int(source),int(dest),int(expert),count))
                moved = True
                break
            if moved:
                break
        if not moved:
            break
    assert remote.sum() + local.sum() == loads.sum()
    assert (remote >= 0).all()
    assert not (remote[~placement.T] > 0).any()
    return {"local":local,"remote":remote,"loads":loads,"moves":moves}


def plan(demand, replicas=8, alpha=0.5, phase2_balance=False):
    demand = np.asarray(demand, dtype=np.int64)
    g, e = demand.shape
    assert e % g == 0 and (demand >= 0).all()
    homes = np.arange(e) // (e//g)
    placement = np.zeros((e,g), dtype=bool)
    placement[np.arange(e),homes] = True
    capacity = np.zeros(g,dtype=int)
    local_budget = int(replicas*alpha)
    phase1, phase2 = [], []
    for source in range(g):
        candidates = np.flatnonzero((homes != source) & (demand[source] > 0))
        chosen = sorted(candidates, key=lambda x:(-demand[source,x],x))[:local_budget]
        placement[chosen,source] = True
        capacity[source] += len(chosen)
        phase1.extend((int(x),source) for x in chosen)
    for _ in range((replicas-local_budget)*g):
        local,remote = initial_work(demand,placement)
        # Appendix B only updates newly localized computations between replica
        # choices. Algorithm 2 is a subsequent stage, not explicitly part of
        # Algorithm 1. Keep the earlier, stronger interpretation as a sensitivity
        # control, never silently substitute it for the literal paper algorithm.
        if phase2_balance:
            balanced = shard(local,remote,placement)
            loads,remote = balanced["loads"],balanced["remote"]
        else:
            loads=local+remote.sum(axis=1)
        target = loads.mean()
        source = int(np.argmax(loads))
        if loads[source] <= target:
            break
        expert = int(np.argmax(remote[source]))
        if remote[source,expert] == 0:
            break
        candidates = np.flatnonzero((capacity < replicas) & ~placement[expert])
        if not len(candidates):
            break
        dest = int(candidates[np.argmin(loads[candidates])])
        # The pseudocode requires post-duplication destination load <= B.
        # Local demand becomes local immediately; remaining remote work can be
        # split by Algorithm 2, not necessarily moved as an indivisible expert.
        newly_local = demand[dest,expert]
        if loads[dest] + newly_local > target:
            break
        placement[expert,dest] = True
        capacity[dest] += 1
        phase2.append((expert,dest))
    assert (placement.sum(axis=0) - e//g <= replicas).all()
    assert placement[np.arange(e),homes].all()
    return {"placement":placement,"phase1":phase1,"phase2":phase2,
            "underspecified_choices":"linear-home initial remote assignment; stable ID ties; integer transfers; phase2 local demand guard",
            "phase2_balance_before_replication":phase2_balance}


def evaluate_plan(demand, placement):
    local,remote = initial_work(np.asarray(demand),placement)
    return shard(local,remote,placement)


if __name__ == "__main__":
    rng = np.random.default_rng(7317)
    for count in [1,8,64,512]:
        for _ in range(30):
            d = rng.multinomial(count*8,rng.dirichlet(np.ones(128)),size=4)
            p = plan(d)
            r = evaluate_plan(d,p["placement"])
            assert r["loads"].sum() == d.sum()
    print("120 demand/placement/sharding invariants PASS")
