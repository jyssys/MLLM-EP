"""Finish the campaign without resetting its absolute wall-clock allowance."""
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback

from .common import ROOT, REPO, START, GPU_STOP, HARD_STOP, setup, write, remaining
from .jobs import configuration, run, clean_session
from . import resources


def alive(pid, module):
    proc = Path('/proc') / str(pid)
    try:
        return proc.stat().st_uid == os.getuid() and module.encode() in (proc / 'cmdline').read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return False


def deadline_guard(stop):
    while not stop.wait(2):
        if time.time() < GPU_STOP:
            continue
        terminated = []
        for path in (ROOT / 'state').glob('*.json'):
            try:
                value = json.loads(path.read_text())
                pid = value.get('pid')
                if pid and alive(pid, 'poc_flashvep.looped_pd_b_slo_serving.worker'):
                    clean_session(pid)
                    terminated.append(pid)
            except (ValueError, FileNotFoundError):
                continue
        state = ROOT / 'state/campaign.json'
        previous = json.loads(state.read_text()) if state.exists() else {}
        pid = previous.get('pid')
        if pid and alive(pid, 'poc_flashvep.looped_pd_b_slo_serving.campaign'):
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        write(state, dict(stage='TIME_BUDGET_STOPPED', time=time.time(),
                          prior_state=previous, terminated_owned_PIDs=terminated))
        return


def diagnostics():
    if remaining() < 1200:
        return
    resources.stop()
    try:
        for model in ('ouro', 'huginn'):
            for arm in ('PD_G1', 'B'):
                if remaining() < 600:
                    return
                job = configuration(model, arm, 'diagnostic', 4, 0, 2026105001,
                                    dataset='W2', cap=8, capacity=1,
                                    diagnostic=True, env=dict(B_EVENT_TRACE='1'),
                                    name=f'{model}_diagnostic_{arm}')
                run(job)
        from .triage import main as triage
        triage()
    finally:
        resources.restore()


def wait_for_owned_inference():
    """A dead launcher does not authorize stopping a healthy inference session."""
    while time.time() < GPU_STOP + 30:
        active = []
        for path in (ROOT / 'state').glob('*.json'):
            try:
                pid = json.loads(path.read_text()).get('pid')
                if pid and alive(pid, 'poc_flashvep.looped_pd_b_slo_serving.worker'):
                    active.append(pid)
            except (ValueError, FileNotFoundError):
                continue
        if not active:
            return
        time.sleep(5)


def main():
    setup()
    lock = ROOT / 'state/supervisor.json'
    if lock.exists():
        old = json.loads(lock.read_text())
        if old.get('pid') != os.getpid() and alive(old.get('pid'), 'poc_flashvep.looped_pd_b_slo_serving.supervisor'):
            raise RuntimeError('Another supervisor is active')
    write(lock, dict(pid=os.getpid(), stage='WAITING_FOR_SERVING', time=time.time(), hard_stop=HARD_STOP))
    stop = threading.Event()
    guard = threading.Thread(target=deadline_guard, args=(stop,), daemon=True)
    guard.start()
    try:
        while True:
            path = ROOT / 'state/campaign.json'
            state = json.loads(path.read_text()) if path.exists() else {}
            if state.get('stage') in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED', 'TIME_BUDGET_STOPPED'):
                pid = state.get('pid')
                if not pid or not alive(pid, 'poc_flashvep.looped_pd_b_slo_serving.campaign'):
                    break
            if state.get('pid') and not alive(state['pid'], 'poc_flashvep.looped_pd_b_slo_serving.campaign'):
                write(path, dict(stage='FAILED_REVIEW_REQUIRED', error='Controller exited without a terminal stage', previous=state, time=time.time()))
                break
            time.sleep(10)
        if state.get('stage') == 'COMPLETE_REVIEW_REQUIRED' and remaining() >= 2 * 3600:
            write(lock, dict(pid=os.getpid(), stage='LONG_W1_WINDOWS', time=time.time()))
            from .extend import main as extend
            extend()
        diagnostics()
    except BaseException as error:
        write(ROOT / 'state/supervisor_error.json', dict(error=repr(error), traceback=traceback.format_exc(), time=time.time()))
    finally:
        wait_for_owned_inference()
        stop.set()
        guard.join(5)
        resources.restore()
    write(lock, dict(pid=os.getpid(), stage='FINAL_ANALYSIS', time=time.time()))
    try:
        from .finish import report, publish
        state_path = ROOT / 'state/campaign.json'
        state = json.loads(state_path.read_text())
        if state.get('stage') not in ('COMPLETE_REVIEW_REQUIRED', 'FAILED_REVIEW_REQUIRED', 'TIME_BUDGET_STOPPED'):
            write(state_path, dict(stage='FAILED_REVIEW_REQUIRED', previous=state, reason='Supervisor closeout', time=time.time()))
        report(final=True)
        files = {}
        for path in sorted((ROOT / 'raw').rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                files[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
        write(ROOT / 'evidence_manifest.json', files)
        write(ROOT / 'state/final_closeout.json', dict(status='FINALIZED_WITH_LIMITATIONS',
                                                     elapsed_hours=(time.time() - START) / 3600,
                                                     hard_stop=HARD_STOP, within_budget=time.time() < HARD_STOP,
                                                     time=time.time()))
        publish()
        write(lock, dict(pid=os.getpid(), stage='COMPLETE_PUSH_VERIFIED', time=time.time()))
    except BaseException as error:
        write(lock, dict(pid=os.getpid(), stage='FINAL_REVIEW_OR_PUBLICATION_FAILED', error=repr(error),
                         traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    main()
