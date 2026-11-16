"""Separate fixed active-arrival windows from finite-cohort drain tails."""
import argparse
import fcntl
import json
import random
import statistics
import time

from .analyze import records
from .campaign import run_job
from .common import QUIET_LOCK, ROOT, bind_previous, resources, write
from .configuration import base, standard_capture_sizes


def configuration(method, qps, seed):
    name = f'long_{method}_O200_q{qps}_s{seed}_W600_v1'
    arm = 'PD_G1' if method == 'Native' else 'DP2'
    budget, slots = (8192, 1024) if method == 'DP2_B8192_S1024' else (512, 16)
    config = base(name, arm=arm, budget=budget, slots=slots, count=1, rate=qps, seed=seed,
        dataset='fresh_confirmation.json', reference_name='fresh_reference_Native_N128_O200_repeat1_v1',
        intern_immutable_prompt_pool=True, phase='FIXED_ACTIVE_ARRIVAL_WINDOW_NOT_GLOBAL_MAXIMUM')
    if method == 'DP2_B8192_S1024':
        config['explicit_cudagraph_capture_sizes'] = standard_capture_sizes(1024)
    rng = random.Random(seed)
    arrivals, at = [], 0.
    while at < 750:
        arrivals.append(at)
        at += rng.expovariate(qps)
    config.update(count=len(arrivals), request_caps=[200] * len(arrivals), arrival_offsets=arrivals,
        timeout=3000, wall_timeout=3600,
        fixed_measurement=dict(warm_seconds=60., measurement_seconds=600., follow_through_seconds=90.,
            arrival_horizon_seconds=750., target_joint_attainment=.95, completion_rate_fraction=.98,
            outstanding_slope_max=.01, outstanding_slope_upper_max=.025,
            queue_block_seconds=30., finite_operational_screen=True))
    pool = json.loads((ROOT / 'data/fresh_confirmation.json').read_text())
    rows = [pool[i % len(pool)] for i in range(len(arrivals))]
    dataset = f'fresh_declared_cycle_long_N{len(arrivals)}.json'
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        path = ROOT / 'data' / dataset
        if path.exists():
            assert json.loads(path.read_text()) == rows
        else:
            write(path, rows)
    config.update(dataset=dataset,
        prompt_reuse_scope='Cycle128 declared unseen real documents. Prefix caching is disabled. Repeated prompts do not add independent text or independent server starts.')
    return config


def assess(name):
    bind_previous()
    from ..dense_ar_slo_layer_pd.sustainable import assess as inherited
    result = inherited(name)
    folder = ROOT / 'raw' / name
    rows = records(folder / 'requests.jsonl')
    selected = [r for r in rows if result['window_begin'] <= r['arrival'] < result['window_end']]
    intervals = [1000 * (b['time'] - a['time']) for r in selected
                 for a, b in zip(r['token_events'], r['token_events'][1:])]
    result.update(mean_TBT_ms=statistics.mean(intervals) if intervals else None,
        mean_TTFT_ms=statistics.mean(r['ttft_ms'] for r in selected) if selected else None,
        latency_scope='Follow every arrival in the600-second measurement window until completion. Include its later drain intervals. Completion-window throughput has a separate denominator.',
        measurement_class='FIXED_ACTIVE_ARRIVAL_WINDOW', sustainable_capacity_claim=False)
    # Also expose intervals physically observed during active arrivals. This
    # secondary statistic must not replace the fully followed arrival cohort.
    active = [1000 * (b['time'] - a['time']) for r in rows
              for a, b in zip(r['token_events'], r['token_events'][1:])
              if result['window_begin'] <= a['time'] and b['time'] < result['window_end']]
    result['active_window_observed_mean_TBT_ms'] = statistics.mean(active) if active else None
    result['active_window_interval_count'] = len(active)
    result['actually_emitted_output_tokens_s'] = sum(
        result['window_begin'] <= event['time'] < result['window_end']
        for row in rows for event in row['token_events']) / (result['window_end'] - result['window_begin'])
    write(ROOT / 'analysis' / (name + '_fixed_window.json'), result)
    print('FIXED_WINDOW', name, result['mean_TBT_ms'], result['active_window_observed_mean_TBT_ms'],
          result['raw_completed_output_tokens_s'], result['stability']['outstanding_delta'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qps', type=float, nargs='+', default=[2., 6.])
    parser.add_argument('--methods', nargs='+', default=['DP2_B8192_S1024', 'DP2_B512_S16', 'Native'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026111631])
    args = parser.parse_args()
    assert json.loads((ROOT / 'analysis/fresh_reference_gate.json').read_text())['passed']
    try:
        for seed in args.seeds:
            for qps in args.qps:
                for method in args.methods:
                    config = configuration(method, qps, seed)
                    run_job(config)
                    assess(config['name'])
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
