"""Budget-limited B-only serving campaign. Resume never repeats a terminal job."""
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .common import ROOT, START, GPU_STOP, HARD_STOP, setup, write, remaining
from .jobs import configuration, run, output_rows, parity
from .analyze import main as analyze, summarize
from . import resources


def heartbeat(stage, **values):
    write(ROOT / 'state/campaign.json', dict(stage=stage, pid=os.getpid(), time=time.time(),
                                            elapsed_hours=(time.time() - START) / 3600,
                                            remaining_GPU_hours=remaining() / 3600, **values))


def qualification():
    file = ROOT / 'configs/qualification.json'
    while not file.exists():
        controller = ROOT / 'state/qualification_controller.json'
        state = json.loads(controller.read_text()) if controller.exists() else {}
        if state.get('status') == 'FAILED':
            raise RuntimeError('Qualification controller failed')
        if remaining() < 3600:
            raise TimeoutError('Qualification did not finish within the experiment budget')
        time.sleep(10)
    # Wait for the heavy byte checks and reference jobs to release their GPU sessions.
    while True:
        state = json.loads((ROOT / 'state/qualification_controller.json').read_text())
        if state['status'] == 'COMPLETE_REVIEW_REQUIRED':
            # COMPLETE is written before the qualification controller restores burners.
            # Wait for that finally block before acquiring the quiet lane.
            pid = state.get('pid')
            proc = Path(f'/proc/{pid}') if pid else None
            try:
                alive = proc is not None and proc.exists() and 'State:\tZ' not in (proc / 'status').read_text()
            except FileNotFoundError:
                alive = False
            if not alive:
                break
        pid = state.get('pid')
        if pid and not Path(f'/proc/{pid}').exists():
            raise RuntimeError('Qualification controller stopped before releasing its pool')
        time.sleep(10)
    result = json.loads(file.read_text())
    for model in ('ouro', 'huginn'):
        checks = list((ROOT / 'raw' / f'{model}_KV_identity_B').glob('KV_checks-*.jsonl'))
        values = [json.loads(line) for path in checks for line in path.read_text().splitlines()]
        result[model]['KV_passed'] = len(values) >= 2 and all(row['passed'] for row in values)
        result[model]['passed'] = result[model]['passed'] and result[model]['KV_passed']
    write(ROOT / 'configs/final_correctness_gate.json', result)
    return {model: row['passed'] for model, row in result.items()}


def execute(job):
    heartbeat('running', current=job['name'])
    result = run(job)
    analyze()
    return result


def calibrate():
    lock = ROOT / 'configs/slo_lock.json'
    if lock.exists():
        return json.loads(lock.read_text())
    values = {}
    for model in ('ouro', 'huginn'):
        measurements = []
        for arm in ('DP_Mixed', 'PD_G1'):
            job = configuration(model, arm, 'calibration', 24, .3 if model == 'ouro' else .08,
                                2026102990, cap=128, name=f'{model}_calibration_{arm}')
            result = execute(job)
            if result['status'] != 'COMPLETE':
                raise RuntimeError('Baseline-only calibration failed: ' + job['name'])
            rows = output_rows(job['name'])
            measurements.append(dict(run=job['name'], N=len(rows),
                                     TTFT_p90_ms=float(np.percentile([r['ttft_ms'] for r in rows], 90)),
                                     TPOT_p90_ms=float(np.percentile([r['tpot_ms'] for r in rows], 90)),
                                     ITL_p90_ms=float(np.percentile([1000 * (event['time'] - event['previous_time']) for r in rows for event in r['token_events'] if not event['first']], 90))))
        values[model] = dict(TTFT_ms=math.ceil(1.25 * max(row['TTFT_p90_ms'] for row in measurements)),
                             TPOT_ms=math.ceil(1.25 * max(row['TPOT_p90_ms'] for row in measurements)),
                             secondary_ITL_ms=math.ceil(1.25 * max(row['ITL_p90_ms'] for row in measurements)),
                             calibration=measurements, rule='ceil(1.25 * max DP_Mixed/PD_G1 low-load p90)',
                             created=time.time(), B_heldout_seen=False)
    write(lock, values)
    return values


def matched_point(model, rate, seed, count, enabled, gpus=(0, 1), workload='W1', arms=None):
    if remaining() < 1800:
        return {}
    arms = list(arms or ('DP_Mixed', 'PD_G1', 'B'))
    if not enabled:
        arms = [arm for arm in arms if arm != 'B']
    # Rotate order without assigning one policy permanently to one device pair.
    offset = seed % len(arms)
    arms = arms[offset:] + arms[:offset]
    jobs = {}
    for arm in arms:
        if remaining() < 1200:
            break
        job = configuration(model, arm, workload, count, rate, seed, gpus=gpus)
        execute(job)
        jobs[arm] = job
    if 'B' in jobs and 'PD_G1' in jobs:
        check = parity(jobs['PD_G1']['name'], jobs['B']['name'])
        check['passed'] = check['N'] == count and check['exact'] == count and check['complete_ids']
        write(ROOT / 'state' / ('parity_' + jobs['B']['name'] + '.json'), check)
        if not check['passed']:
            enabled = False
            write(ROOT / 'state' / (model + '_B_serving_parity_failure.json'), check)
    return dict(jobs=jobs, enabled=enabled)


def main():
    setup()
    heartbeat('wait_for_qualification')
    try:
        enabled = qualification()
        resources.stop()
        calibrate()
        heartbeat('W1_primary', enabled=enabled)
        best = {}
        ladders = dict(ouro=(1., 2., 4., 8.), huginn=(.2, .4, .8, 1.6))
        for index in range(4):
            for model in ('ouro', 'huginn'):
                rate = ladders[model][index]
                if remaining() < 3.5 * 3600:
                    break
                seed = 2026103001
                result = matched_point(model, rate, seed, 200, enabled[model])
                enabled[model] = result.get('enabled', enabled[model])
                for arm, job in result.get('jobs', {}).items():
                    row = summarize(job)[0]
                    if float(row.get('joint_attainment_percent', 0)) >= 90 and row.get('weak_queue_check', False):
                        best[model, arm] = rate
        for model in ('ouro', 'huginn'):
            selected = best.get((model, 'PD_G1'), ladders[model][0])
            # Weaker controls receive a declared anchor rather than consume the whole budget.
            matched_point(model, selected, 2026103001, 200, False, arms=('DP', 'PD'))
        heartbeat('heldout_confirmation', enabled=enabled)
        for model in ('ouro', 'huginn'):
            rates = sorted({best.get((model, arm), ladders[model][0]) for arm in ('DP_Mixed', 'PD_G1', 'B')})
            for rate in rates:
                for repeat in (1, 2):
                    if remaining() < 2.5 * 3600:
                        break
                    pair = (2, 3) if repeat == 1 else (4, 5)
                    result = matched_point(model, rate, 2026103001 + 1000 * repeat, 200, enabled[model], gpus=pair)
                    enabled[model] = result.get('enabled', enabled[model])
        heartbeat('supplementary', enabled=enabled)
        for workload in ('W2', 'W3', 'W4'):
            for model in ('ouro', 'huginn'):
                if remaining() < 1.5 * 3600:
                    break
                rate = best.get((model, 'PD_G1'), .2 if model == 'huginn' else 1.)
                result = matched_point(model, rate, 2026103101, 80 if workload == 'W4' else 100,
                                       enabled[model], workload=workload)
                enabled[model] = result.get('enabled', enabled[model])
        heartbeat('optional_scale', enabled=enabled)
        for count in (4, 8):
            for model in ('ouro', 'huginn'):
                if remaining() < 3600:
                    break
                rate = best.get((model, 'PD_G1'), .2 if model == 'huginn' else 1.) * count / 2
                result = matched_point(model, rate, 2026103201, 100, enabled[model],
                                       gpus=tuple(range(count)), arms=('DP_Mixed', 'PD_G1', 'B'))
                enabled[model] = result.get('enabled', enabled[model])
        analyze()
        heartbeat('COMPLETE_REVIEW_REQUIRED', enabled=enabled, pending=['Final scientific review', 'Commit/push'])
    except BaseException as error:
        heartbeat('FAILED_REVIEW_REQUIRED', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        resources.restore()


if __name__ == '__main__':
    main()
