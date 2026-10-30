"""Time-limited longer W1 windows. Reuse 200 distinct texts without prefix caching."""
import csv
import json
import math
import time

from .common import ROOT, START, GPU_STOP, write, remaining
from .jobs import configuration, run, parity
from .analyze import summarize, main as analyze
from . import resources


def complete_scale_controls():
    """Optional weak controls only after the full-population primary windows."""
    references = []
    for path in (ROOT / 'configs/jobs').glob('*.json'):
        job = json.loads(path.read_text())
        if job['workload'] == 'W1' and job['arm'] == 'DP_Mixed' and len(job['gpus']) in (4, 8):
            status = ROOT / 'raw' / job['name'] / 'status.json'
            if status.exists() and json.loads(status.read_text())['status'] == 'COMPLETE':
                references.append(job)
    for reference in sorted(references, key=lambda row: (len(row['gpus']), row['model'])):
        for arm in ('DP', 'PD'):
            if remaining() < 5400:
                return
            job = configuration(reference['model'], arm, 'W1', reference['count'],
                                reference['rate'], reference['seed'], gpus=reference['gpus'],
                                dataset=reference['dataset'], cap=reference['cap'], capacity=reference['capacity'])
            run(job)
            analyze()


def check_stronger_plain_DP():
    """Do not assume Mixed is the strongest colocated Huginn scheduler."""
    # Initial matched W1: plain DP96.5%, DP_Mixed93.5% at0.2 requests/s.
    # Probe the next registered rate without retuning its SLO or model.
    if remaining() < 7200:
        return
    plan = dict(model='huginn', next_rate=.4, count=200,
                reason='Plain DP exceeds DP_Mixed on the initial matched0.2 point. Its higher-rate capacity must not be assumed.',
                budget_rule='Start no new probe with fewer than two GPU hours. Primary windows precede this probe. Scale extras follow it.')
    write(ROOT / 'configs/additional_plain_DP_plan.json', plan)
    passed = True
    for seed, gpus in ((2026103001, (0, 1)), (2026106001, (2, 3)), (2026107001, (4, 5))):
        if remaining() < 7200:
            return
        job = configuration('huginn', 'DP', 'W1', 200, .4, seed, gpus=gpus, dataset='W1')
        run(job)
        analyze()
        row = summarize(job)[0]
        passed &= row['status'] == 'COMPLETE' and float(row.get('joint_attainment_percent', 0)) >= 90 and row.get('weak_queue_check', False)
    if passed and remaining() >= 7200:
        job = configuration('huginn', 'DP', 'W1', 200, .8, 2026103001, dataset='W1')
        run(job)
        analyze()
    elif not passed:
        # Confirm the lower plain-DP point with the same complete source population.
        for seed in (2026106001, 2026107001):
            if remaining() < 7200:
                return
            job = configuration('huginn', 'DP', 'W1', 200, .2, seed,
                                gpus=(6, 7), dataset='W1_long')
            run(job)
            analyze()


def main():
    # Prioritize a longer primary window over more scale discovery.
    # The resumed original controller can hold an older analyzer import.
    # Rebuild original-epoch tables before selecting its extension rates.
    analyze()
    if remaining() < 2.0 * 3600:
        write(ROOT / 'state/extended_windows.json', dict(status='NOT_MEASURED_TIME_BUDGET', time=time.time()))
        return
    source = json.loads((ROOT / 'data/W1.json').read_text())
    expanded = []
    for occurrence in range(24):
        for row in source:
            expanded.append(dict(row, source_prompt_id=row['prompt_id'], occurrence=occurrence,
                                 prompt_id=f"{row['prompt_id']}::occurrence{occurrence}"))
    write(ROOT / 'data/W1_long.json', expanded)
    rows = list(csv.DictReader((ROOT / '03_primary_w1_capacity.csv').open()))
    plan = []
    for model in ('ouro', 'huginn'):
        measured = [row for row in rows if row['model'] == model and row['GPU_count'] == '2']
        by_arm = {}
        for arm in ('DP_Mixed', 'PD_G1', 'B'):
            rates_for_arm = sorted({float(row['offered_rate']) for row in measured if row['arm'] == arm})
            by_arm[arm] = [rate for rate in rates_for_arm if all(
                row['status'] == 'COMPLETE' and float(row.get('joint_attainment_percent') or 0) >= 90
                and row.get('weak_queue_check') == 'True'
                for row in measured if row['arm'] == arm and float(row['offered_rate']) == rate)]
        enabled = not (ROOT / 'state' / (model + '_B_serving_parity_failure.json')).exists()
        all_rates = sorted({float(row['offered_rate']) for row in rows if row['model'] == model and row['GPU_count'] == '2' and row['arm'] == 'PD_G1'})
        base = by_arm['PD_G1'][-1] if by_arm['PD_G1'] else (min(all_rates) / 2 if all_rates else .1 if model == 'huginn' else .5)
        rates = {base}
        # One independently defined refinement point. Never select on a favorable E2E metric.
        greater = [rate for rate in all_rates if rate > base]
        if by_arm['PD_G1']:
            rates.add((base + min(greater)) / 2 if greater else 1.5 * base)
        dp_rates = sorted({float(row['offered_rate']) for row in rows if row['model'] == model
                           and row['GPU_count'] == '2' and row['arm'] == 'DP_Mixed'})
        if by_arm['DP_Mixed']:
            rates.add(by_arm['DP_Mixed'][-1])
        elif dp_rates:
            # A failed colocated ladder needs a lower-load feasibility point.
            # Do not infer zero capacity from the absence of a tested passing rate.
            rates.add(min(dp_rates) / 2)
        for rate in sorted(rates):
            # Keep the complete empirical W1 distribution, including its long prompts.
            # Whole cycles avoid an easier low-rate prefix or a biased partial final cycle.
            count = min(4800, max(200, 200 * math.ceil(rate * 360 / 200)))
            for repeat in (0, 1):
                plan.append(dict(model=model, rate=rate, count=count, distinct_texts=min(count, 200),
                                 seed=2026106001 + repeat * 1000, enabled=enabled))
    write(ROOT / 'configs/long_window_plan.json', dict(
        created=time.time(), plan=plan, target_seconds=360, source_distinct_texts=200,
        seed_base=2026106001, seed_rule='Disjoint from initial ladder and main independent-confirmation seeds.',
        colocated_rule='Include the best passing DP_Mixed rate, or half its lowest tested rate when no passing point exists.',
        low_rate_rule='Preserve all200 source prompts. Test half the lowest failed PD_G1 rate, not an automatic quarter-rate point with fewer prompts. Omit extra points if the hard budget cannot support the full population.',
        repeated_arrival_rule='Cycle the frozen 200 distinct W1 prompts. Use unique occurrence IDs and independent request seeds. No cross-request prefix reuse.',
        time_rule='Skip this optional extension if fewer than two GPU hours remain. Stop planning new jobs with fewer than 45 GPU minutes.'))
    resources.stop()
    try:
        for item in plan:
            if remaining() < 2700:
                break
            model, rate, seed = item['model'], item['rate'], item['seed']
            enabled = item['enabled'] and not (ROOT / 'state' / (model + '_B_serving_parity_failure.json')).exists()
            arms = ['DP_Mixed', 'PD_G1'] + (['B'] if enabled else [])
            # The main seed increments by1000 preserve parity for two arms.
            # Alternate the extension explicitly after the B gate closes.
            offset = (seed // 1000) % len(arms)
            arms = arms[offset:] + arms[:offset]
            jobs = {}
            for arm in arms:
                if remaining() < 1800:
                    break
                job = configuration(model, arm, 'W1', item['count'], rate, seed,
                                    gpus=(6, 7), dataset='W1_long',
                                    name=f'{model}_W1_long_{arm}_rate{rate:g}_seed{seed}',
                                    timeout=min(2400, max(1200, item['count'] / rate * 2.5)))
                run(job)
                analyze()
                jobs[arm] = job
            if 'PD_G1' in jobs and 'B' in jobs:
                result = parity(jobs['PD_G1']['name'], jobs['B']['name'])
                write(ROOT / 'state' / ('parity_' + jobs['B']['name'] + '.json'), result)
                if result['N'] != item['count'] or result['exact'] != item['count']:
                    write(ROOT / 'state' / (model + '_B_serving_parity_failure.json'), result)
        check_stronger_plain_DP()
        complete_scale_controls()
        write(ROOT / 'state/extended_windows.json', dict(status='COMPLETE_WITH_BUDGET_LIMITS', time=time.time()))
    finally:
        resources.restore()


if __name__ == '__main__':
    main()
