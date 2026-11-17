"""Search modeled variation before spending serving runs on new policies."""
import collections
import fcntl
import random
import time
from .common import ROOT, QUIET_LOCK, resources, write
from .replay import load_model, groups, static_frontier, oracle, evaluate
from ..duetserve_agg_decode.configuration import offsets


def main():
    results = []
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        assert not resources().processes()
        resources().restore(list(range(8)))
        from .cost_trace import fit
        fit('v2')
        for size in (4, 8):
            for kind in ('fixed16', 'fixed64', 'mixed16_64_200', 'bursty_mixed'):
                count = size * 16
                cap_groups = [16] * 16 if kind == 'fixed16' else [64] * 16 if kind == 'fixed64' else [16] * 8 + [64] * 5 + [200] * 3
                random.Random(2026111730).shuffle(cap_groups)
                caps = [cap for cap in cap_groups for _ in range(size)]
                for qps in (3.2, 4.8, 6.4):
                    arrival = offsets(count, qps, 2026111731)
                    if kind == 'bursty_mixed':
                        # Piecewise-rate Poisson trace, normalized to the same
                        # total horizon as its homogeneous control.
                        delta = [(b - a) * (.25 if (i // (size * 2)) % 2 == 0 else 1.75)
                                 for i, (a, b) in enumerate(zip(arrival, arrival[1:]))]
                        scale = arrival[-1] / sum(delta)
                        arrival = [0.]
                        for gap in delta:
                            arrival.append(arrival[-1] + gap * scale)
                    model = load_model(size)
                    workload = groups(arrival, caps, size)
                    start = time.time()
                    best, frontier = static_frontier(model, workload)
                    dynamic = oracle(model, workload, width=64)
                    actual = evaluate(dynamic)
                    ratio = actual['SLO_output_tokens_s'] / best[0]['SLO_output_tokens_s'] if best[0]['SLO_output_tokens_s'] else None
                    record = dict(group_size=size, kind=kind, QPS=qps, count=count,
                        output_tokens=sum(caps), static=best[0], static_policy=best[1],
                        oracle=actual, oracle_to_static_SLO_ratio=ratio,
                        oracle_cuts=dict(collections.Counter(r['layers'] for r in dynamic.rows)),
                        oracle_decisions=dynamic.decisions, static_decisions=best[2].decisions,
                        borrowed_cost_cells=sorted(model.borrowed), seconds=time.time()-start,
                        arrivals=arrival, caps=caps, evidence='OFFLINE_ORACLE',
                        limitation='Finite-cohort bounded beam, not a proof of the optimum. Modeled host costs. Fixed FIFO groups and whole-cohort D guard. The score can fall below the separately tested static frontier because pruning is approximate. No measured serving gain or physical idle claim.')
                    results.append(record)
                    write(ROOT / 'analysis/oracle_variance_screen_v1.json', results)
                    print('ORACLE', size, kind, qps, 'ratio', ratio, 'cuts', record['oracle_cuts'], flush=True)


if __name__ == '__main__':
    main()
