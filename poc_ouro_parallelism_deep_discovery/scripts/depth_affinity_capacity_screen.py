"""Screen one depth-pinned layout. Capacity arithmetic is not GPU speedup."""
import argparse
import csv
import itertools
import random
from poc_ouro_parallelism_deep_discovery.contract import ROOT, write


def visits(rows):
    total = sum(row['N_decode_events'] for row in rows)
    result = [total, total,
              sum(row['R3_events'] + row['R4_events'] for row in rows),
              sum(row['R4_events'] for row in rows)]
    assert sum(result) == sum(row['executed_depth_steps'] for row in rows)
    return [value/total for value in result]


def unique_partitions(P):
    # Device-label permutations have identical load in this zero-transfer screen.
    choices = []
    for allocation in itertools.product(range(min(P, 4)), repeat=4):
        if allocation[0] != 0:
            continue
        if any(allocation[i] > max(allocation[:i], default=-1) + 1 for i in range(4)):
            continue
        choices.append(allocation)
    return choices


def screen(v, P):
    options = []
    for assignment in unique_partitions(P):
        loads = [sum(value for value, owner in zip(v, assignment) if owner == rank) for rank in range(P)]
        options.append((max(loads), assignment, loads))
    maximum, assignment, loads = min(options)
    total = sum(v)
    return dict(P=P, visits_per_committed_token=v, best_depth_owner=list(assignment),
                normalized_rank_work=loads, best_static_rank_work_max=maximum,
                pooled_work_capacity=P/total, pinned_work_capacity=1/maximum,
                pinned_over_pooled_saturated_capacity=total/(P*maximum),
                finite_C=[dict(C=C, pinned_capacity=min(C/total, 1/maximum),
                               pooled_capacity=min(C, P)/total,
                               pinned_over_pooled_capacity=min(C/total, 1/maximum)/(min(C, P)/total))
                          for C in [1, 2, 4, 8]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    output = ROOT/'analysis'/args.label
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT/'analysis/adaptive_profile_01/request_profiles.csv'
    raw = list(csv.DictReader(source.open()))
    for row in raw:
        for key in ['N_decode_events', 'R2_events', 'R3_events', 'R4_events', 'executed_depth_steps']:
            row[key] = int(row[key])
    records = []
    rng = random.Random(202610191330)
    for size in ['1.4B', '2.6B']:
        for domain in ['gsm8k', 'math500']:
            rows = [row for row in raw if row['model'] == size and row['dataset'] == domain]
            assert len(rows) == 8
            for P in [2, 4, 8]:
                observation = screen(visits(rows), P)
                bootstrap = sorted(screen(visits(rng.choices(rows, k=len(rows))), P)[
                    'pinned_over_pooled_saturated_capacity'] for _ in range(4000))
                observation.update(model=size, dataset=domain, mode='original_adaptive', N_requests=len(rows),
                                   request_bootstrap_95_CI=[bootstrap[100], bootstrap[3899]],
                                   P8_status='UNMEASURED_PROJECTION_ONLY' if P == 8 else 'CPU_STRUCTURAL_SCREEN')
                records.append(observation)
    fixed = [screen([1, 1, 1, 1], P) for P in [2, 4, 8]]
    write(output/'screen.json', dict(evidence_class='ANALYTIC_BOUND',
          meaning='Service-capacity limits inside a declared unit-cost, unbatched, zero-transfer queueing model. Not a physical latency bound.',
          records=records, fixed_R4=fixed, source=str(source.relative_to(ROOT)),
          oracle='The static owner search uses the complete observed calibration depth histogram. It is not a causal scheduler.',
          assumptions=['One normalized core task costs one unit at every depth',
                       'All ranks retain complete shared weights',
                       'Each logical depth has one fixed KV and compute owner',
                       'Hidden transfers, gates, output head, and publication cost zero',
                       'No operator sharding or batch-efficiency benefit',
                       'The pool is an ideal work-conserving comparison, not measured CDB'],
          limitations=['No measured throughput, E2E, or candidate quality',
                       'Batched core costs, real communication, and heterogeneous request readiness can change actual results',
                       'No claim against layouts that split a depth, relocate KV, or use another exact representation']))
    for row in records:
        print(row['model'], row['dataset'], row['P'], round(row['pinned_over_pooled_saturated_capacity'], 5))


if __name__ == '__main__':
    main()
