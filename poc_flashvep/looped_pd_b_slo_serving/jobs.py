"""Completed-run resume, explicit GPU sessions, and wall-clock limits."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .common import ROOT, REPO, PYTHON, GPU_STOP, setup, write, append, env


def configuration(model, arm, workload, count, rate, seed, gpus=(0, 1), **extra):
    name = extra.pop('name', f'{model}_{workload}_{arm}_N{len(gpus)}_rate{rate:g}_seed{seed}')
    value = dict(name=name, model=model, arm=arm, workload=workload,
                 dataset=extra.pop('dataset', workload), count=count, rate=rate, seed=seed,
                 gpus=list(gpus), cap=extra.pop('cap', {'W1': 128, 'W2': 64, 'W3': 128, 'W4': 256}.get(workload, 32)),
                 capacity=extra.pop('capacity', 8), timeout=extra.pop('timeout', 2400), **extra)
    path = ROOT / 'configs/jobs' / (name + '.json')
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('An existing run name has a different contract: ' + name)
    if not path.exists():
        write(path, value)
    return value


def clean_session(pid):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid == os.getuid() and os.getsid(int(proc.name)) == pid:
                # The isolated Popen session contains only this job and inherited workers.
                os.killpg(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def run(job):
    setup()
    folder = ROOT / 'raw' / job['name']
    status = folder / 'status.json'
    if status.exists():
        old = json.loads(status.read_text())
        if old.get('status') in ('COMPLETE', 'FAILED', 'TIME_BUDGET_SKIPPED'):
            print('REUSE', job['name'], old['status'], flush=True)
            return old
    if time.time() + 600 >= GPU_STOP:
        value = dict(status='TIME_BUDGET_SKIPPED', time=time.time())
        write(status, value)
        return value
    path = ROOT / 'configs/jobs' / (job['name'] + '.json')
    command = [job.get('python', PYTHON), '-m', 'poc_flashvep.looped_pd_b_slo_serving.worker', str(path)]
    environment = env(job['gpus'])
    environment.update(job.get('env', {}))
    environment['ABC_LANE'] = 'parallel_correctness' if job.get('diagnostic') else 'isolated_quiet'
    log = (ROOT / 'logs' / (job['name'] + '.log')).open('a')
    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    append(ROOT / 'commands.jsonl', dict(command=command, name=job['name'], pid=process.pid,
                                       gpus=job['gpus'], diagnostic=job.get('diagnostic', False), time=time.time()))
    write(ROOT / 'state' / (job['name'] + '.json'), dict(pid=process.pid, job=job, status='RUNNING', time=time.time()))
    print('START', job['name'], process.pid, flush=True)
    limit = min(GPU_STOP, time.time() + job.get('timeout', 2400) + 600)
    killed = False
    while process.poll() is None:
        if time.time() >= limit:
            clean_session(process.pid)
            killed = True
            break
        time.sleep(1)
    process.wait(timeout=30)
    clean_session(process.pid)
    log.close()
    if killed or not status.exists():
        write(status, dict(status='FAILED', error='Whole-process watchdog or missing terminal state',
                           returncode=process.returncode, time=time.time()))
    value = json.loads(status.read_text())
    write(ROOT / 'state' / (job['name'] + '.json'), dict(job=job, **value))
    print('END', job['name'], value['status'], flush=True)
    return value


def output_rows(name):
    path = ROOT / 'raw' / name / 'requests.jsonl'
    if not path.exists():
        return []
    return [r for line in path.read_text().splitlines() if line.strip()
            for r in [json.loads(line)] if not r.get('warmup', False)]


def parity(left, right):
    a = {r['sample_id']: r for r in output_rows(left)}
    b = {r['sample_id']: r for r in output_rows(right)}
    ids = sorted(a.keys() & b.keys())
    differences = []
    for i in ids:
        if a[i]['output_ids'] != b[i]['output_ids']:
            pos = next((j for j, (x, y) in enumerate(zip(a[i]['output_ids'], b[i]['output_ids'])) if x != y),
                       min(len(a[i]['output_ids']), len(b[i]['output_ids'])))
            differences.append(dict(sample_id=i, first_divergence=pos,
                                    left_length=len(a[i]['output_ids']), right_length=len(b[i]['output_ids'])))
    return dict(left=left, right=right, N=len(ids), exact=len(ids) - len(differences),
                input_equal=sum(a[i]['input_ids'] == b[i]['input_ids'] for i in ids),
                length_equal=sum(len(a[i]['output_ids']) == len(b[i]['output_ids']) for i in ids),
                stop_equal=sum(a[i].get('finish_reason') == b[i].get('finish_reason') for i in ids),
                depth_equal=sum(a[i].get('exit_depths') == b[i].get('exit_depths') for i in ids),
                complete_ids=bool(ids) and len(ids) == len(a) == len(b), differences=differences,
                evidence='MEASURED_GENERATED_SEQUENCE_COMPARISON_NOT_ANSWER_ACCURACY')
