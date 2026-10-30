"""Prioritize B diagnosis without aborting the healthy in-flight baseline."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from .common import ROOT, GPU_STOP, write
from .jobs import configuration, run, output_rows, parity
from . import resources


def alive(pid, module):
    try:
        proc = Path(f'/proc/{pid}')
        return proc.stat().st_uid == os.getuid() and module.encode() in (proc / 'cmdline').read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False


def prefix_plan():
    plans = {}
    for model, rate in (('ouro', 1.), ('huginn', .2)):
        left = f'{model}_W1_PD_G1_N2_rate{rate:g}_seed2026103001'
        right = f'{model}_W1_B_N2_rate{rate:g}_seed2026103001'
        check = parity(left, right)
        a = {r['sample_id']: r for r in output_rows(left)}
        b = {r['sample_id']: r for r in output_rows(right)}
        times = []
        for difference in check['differences']:
            relative = []
            for name, rows in ((left, a), (right, b)):
                start = json.loads((ROOT / 'raw' / name / 'window.json').read_text())['start']
                row = rows[difference['sample_id']]
                count = 0
                event_time = row['finish']
                for event in row['token_events']:
                    count += event['new_tokens']
                    if count > difference['first_divergence']:
                        event_time = event['time']
                        break
                relative.append(event_time - start)
            times.append((max(relative), difference))
        if not times:
            raise RuntimeError('The frozen counterexample no longer exists')
        when, first = min(times, key=lambda item: item[0])
        trace = json.loads((ROOT / 'raw' / left / 'arrival_trace.json').read_text())
        count = min(200, max(8, sum(offset <= when for offset in trace['offsets']) + 1))
        plans[model] = dict(model=model, left=left, right=right, original=check,
                            first_observed_divergence=first, relative_divergence_s=when,
                            causal_prefix_count=count, rate=rate, cap=128, seed=2026103001,
                            reason='Retain every scheduled arrival through the earliest observed divergence, plus one margin request. Preserve all eight warmup prompts.')
    write(ROOT / 'configs/b_repair_prefix_plan.json', plans)
    return plans


def enqueue(job):
    queue = ROOT / 'state/repair_queue'
    queue.mkdir(exist_ok=True)
    path = queue / (job['name'] + '.json')
    if not path.exists():
        write(path, job)


def enqueue_initial(plans):
    for model in ('ouro', 'huginn'):
        plan = plans[model]
        for index, arm in enumerate(('PD_G1', 'B', 'PD_G1')):
            enqueue(configuration(model, arm, 'diagnostic', plan['causal_prefix_count'], plan['rate'],
                                  plan['seed'], gpus=(0, 1), dataset='W1', cap=128, capacity=8,
                                  diagnostic=True, timeout=1200, repair_order=(0 if model == 'ouro' else 3) + index,
                                  name=f'repair_r0_{model}_prefix{plan["causal_prefix_count"]}_{arm}_repeat{index}'))


def summary(plans):
    results = {}
    for model, plan in plans.items():
        names = [f'repair_r0_{model}_prefix{plan["causal_prefix_count"]}_{arm}_repeat{index}'
                 for index, arm in enumerate(('PD_G1', 'B', 'PD_G1'))]
        results[model] = dict(control_vs_B=parity(names[0], names[1]),
                              control_self_repeat=parity(names[0], names[2]), runs=names,
                              scope='Diagnostic natural generation. No clean latency or benchmark accuracy claim.')
    write(ROOT / 'configs/b_repair_initial_results.json', results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plans = prefix_plan()
    if args.plan_only:
        print(json.dumps({m: {k: v[k] for k in ('causal_prefix_count', 'relative_divergence_s', 'first_observed_divergence')}
                          for m, v in plans.items()}, indent=2))
        return
    state_path = ROOT / 'state/repair_priority.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('pid') != os.getpid() and alive(previous.get('pid'), 'repair_priority'):
            raise RuntimeError('A repair-priority controller is already active')
    campaign = json.loads((ROOT / 'state/campaign.json').read_text())
    parent = campaign['pid']
    current = campaign.get('current')
    active_path = ROOT / 'state' / (current + '.json') if current else None
    active = json.loads(active_path.read_text()) if active_path and active_path.exists() else {}
    paused = False
    try:
        if alive(parent, 'poc_flashvep.looped_pd_b_slo_serving.campaign'):
            os.kill(parent, signal.SIGSTOP)
            paused = True
        write(state_path, dict(pid=os.getpid(), stage='WAIT_CURRENT_HEALTHY_RUN', paused_CPU_controller=parent,
                               in_flight_worker=active.get('pid'), time=time.time(), GPU_stop=GPU_STOP))
        while active.get('pid') and alive(active['pid'], 'poc_flashvep.looped_pd_b_slo_serving.worker'):
            if time.time() >= GPU_STOP:
                return
            time.sleep(1)
        resources.stop()
        enqueue_initial(plans)
        last_work = time.time()
        while time.time() + 600 < GPU_STOP:
            pending = sorted((ROOT / 'state/repair_queue').glob('*.json'),
                             key=lambda path: (json.loads(path.read_text()).get('repair_order', 100), path.name))
            if pending:
                path = pending[0]
                job = json.loads(path.read_text())
                write(state_path, dict(pid=os.getpid(), stage='DIAGNOSTIC_RUNNING', current=job['name'],
                                       paused_CPU_controller=parent, time=time.time(), GPU_stop=GPU_STOP))
                result = run(job)
                write(path.with_suffix('.done'), dict(job=job, result=result))
                path.unlink()
                last_work = time.time()
                summary(plans)
                continue
            heartbeat = ROOT / 'state/repair_heartbeat.json'
            recent = max(last_work, heartbeat.stat().st_mtime if heartbeat.exists() else 0)
            write(state_path, dict(pid=os.getpid(), stage='WAIT_REPAIR_NEXT_TEST', paused_CPU_controller=parent,
                                   idle_auto_resume_s=600, time=time.time(), GPU_stop=GPU_STOP))
            if time.time() - recent >= 600 or (ROOT / 'state/repair_release.json').exists():
                break
            time.sleep(2)
    finally:
        if paused and alive(parent, 'poc_flashvep.looped_pd_b_slo_serving.campaign'):
            os.kill(parent, signal.SIGCONT)
        write(state_path, dict(pid=os.getpid(), stage='ORIGINAL_CONTROLLER_RESUMED', time=time.time(),
                               note='B serving gates remain closed. A repair needs a separate qualification epoch.'))


if __name__ == '__main__':
    main()
