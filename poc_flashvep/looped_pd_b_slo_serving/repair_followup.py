"""Expand a promising diagnostic without reopening the original B performance gate."""
import json
import argparse
import os
import time

from .common import ROOT, REPO, GPU_STOP, write
from .jobs import configuration, parity, output_rows
from .repair_priority import enqueue, alive


def await_cohort(names, count, model):
    while time.time() + 600 < GPU_STOP:
        statuses = []
        for name in names:
            path = ROOT / 'raw' / name / 'status.json'
            statuses.append(json.loads(path.read_text()).get('status') if path.exists() else 'PENDING')
        if all(status in ('COMPLETE', 'FAILED', 'TIME_BUDGET_SKIPPED') for status in statuses):
            results = [parity(names[0], names[index]) for index in (1, 2)]
            depth = 4 if model == 'ouro' else 32
            full_depth = all(all(value == depth for value in row['exit_depths'])
                             and len(row['exit_depths']) == row['output_tokens']
                             for name in names for row in output_rows(name))
            passed = statuses == ['COMPLETE'] * 3 and full_depth and all(
                value['N'] == count and value['exact'] == count and value['complete_ids']
                and value['input_equal'] == count and value['depth_equal'] == count for value in results)
            return dict(statuses=statuses, comparisons=results, passed=passed, full_depth=full_depth)
        write(ROOT / 'state/repair_heartbeat.json', dict(pid=os.getpid(), waiting_for=names, time=time.time()))
        time.sleep(5)
    return dict(passed=False, reason='Original absolute GPU deadline')


def main():
    state_path = ROOT / 'state/repair_followup.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('pid') != os.getpid() and alive(previous.get('pid'), 'repair_followup'):
            raise RuntimeError('A repair qualification/serving watcher is already active')
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('all_linear_fixed', 'canonical_all'), default='canonical_all')
    variant = parser.parse_args().variant
    version = 'v4' if variant == 'canonical_all' else 'v3'
    initial = [f'repair_{version}_ouro_{variant}_N7_cap126_{arm}_repeat{index}'
               for index, arm in enumerate(('PD_G1', 'B', 'PD_G1'))]
    stages = [dict(model='ouro', count=7, cap=126, runs=initial)]
    checks = []
    previous = await_cohort(initial, 7, 'ouro')
    checks.append(dict(stage=stages[0], result=previous))
    write(ROOT / 'configs/b_repair_expansion.json', checks)
    if not previous['passed']:
        return
    rejected = set()
    for order, (model, count, cap, rate) in enumerate((('ouro', 8, 128, 1.),
                                                     ('ouro', 32, 128, 1.),
                                                     ('ouro', 200, 128, 1.),
                                                     ('huginn', 33, 128, .2),
                                                     ('huginn', 200, 128, .2))):
        if model in rejected:
            continue
        names = []
        for index, arm in enumerate(('PD_G1', 'B', 'PD_G1')):
            name = f'repair_expand_{variant}_{model}_N{count}_cap{cap}_{arm}_repeat{index}'
            enqueue(configuration(model, arm, 'diagnostic', count, rate, 2026103001,
                                  gpus=(0, 1), dataset='W1', cap=cap, capacity=8, diagnostic=True,
                                  timeout=1800, repair_order=6. + order * .01 + index * .001, name=name,
                                  python=str(REPO / 'poc_flashvep/looped_pd_b_slo_serving/repair_python.py'),
                                  env=dict(B_REPAIR_VARIANT=variant, B_REPAIR_RUN_DIR=str(ROOT / 'raw' / name))))
            names.append(name)
        write(ROOT / 'state/repair_followup.json', dict(pid=os.getpid(), stage='QUALIFICATION',
                                                       model=model, count=count, runs=names,
                                                       original_B_gate='CLOSED', time=time.time()))
        result = await_cohort(names, count, model)
        checks.append(dict(stage=dict(model=model, count=count, cap=cap, runs=names), result=result))
        write(ROOT / 'configs/b_repair_expansion.json', checks)
        if not result['passed']:
            rejected.add(model)
            write(ROOT / 'state/repair_followup.json', dict(pid=os.getpid(), stage='REPAIR_NOT_QUALIFIED',
                                                           failed_stage=checks[-1], original_B_gate='CLOSED', time=time.time()))
    write(ROOT / 'state/repair_followup.json', dict(pid=os.getpid(), stage='MODEL_QUALIFICATIONS_FINISHED_REVIEW_REQUIRED',
                                                   rejected_models=sorted(rejected),
                                                   original_B_gate='CLOSED', accepted_serving_speedup=False,
                                                   next='Review KV identity, source hash, and a separate serving epoch before accepting performance.',
                                                   time=time.time()))
    from .repair_epoch import main as serving_epoch
    serving_epoch(checks, variant)


if __name__ == '__main__':
    main()
