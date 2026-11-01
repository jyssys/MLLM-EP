"""Advance bounded extension diagnostics without interrupting active GPU jobs."""
import fcntl
import subprocess
import importlib
from .common import *
from .guard import processes


def pending_jobs():
    from .campaign import terminal
    values=[]
    for filename in json.loads((ROOT/'configs/study_plan.json').read_text())['configs']:
        c=json.loads((ROOT/'configs'/filename).read_text())
        if terminal(c['name'])=='PENDING':values.append(c)
    return values


def resume_if_needed():
    if processes('campaign') or processes('dispatch') or not pending_jobs():return
    path=ROOT/'state/search_stage.json';stage=json.loads(path.read_text())
    if stage.get('stage')=='GPU_SEARCH_COMPLETE_ANALYSIS_PENDING':
        write(path,dict(time=time.time(),stage='EXTENSION_WORK',reason='The original GPU stage ended. Continue the separately gated user extension.'))
    with (ROOT/'logs/extension_campaign_resume_20261102.log').open('a') as log:
        child=subprocess.Popen([str(PYTHON),'-m','poc_flashvep.looped_pd_atomic_prefill.campaign'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    append(ROOT/'state/controller_reloads.jsonl',dict(time=time.time(),new_controller=child.pid,
        reason='Resume pending immutable extension namespaces after the original GPU stage. An exclusive controller lease prevents duplicate queue owners.'))


def main():
    lease=(ROOT/'state/extension_manager.lock').open('a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    budget=json.loads((ROOT/'configs/extension_budget_20261102.json').read_text())
    from .campaign import terminal
    while not (ROOT/'state/CLOSED.json').exists():
        now=time.time()
        if now>=budget['new_GPU_job_cutoff_epoch']:
            p=ROOT/'state/skipped_jobs.json';skips=json.loads(p.read_text()) if p.exists() else {}
            for c in pending_jobs():skips[c['name']]=dict(time=now,status='NOT_RUN_USER_11H_DEADLINE')
            write(p,skips)
            write(ROOT/'state/extension_terminal_20261102.json',dict(time=now,status='TIME_BUDGET_CLOSEOUT',
                reason='No new GPU job after 10:00 KST. Active owned jobs have a 10:15 stop. Preserve incomplete questions as unmeasured.'))
            write(ROOT/'state/search_stage.json',dict(time=now,stage='STOPPED_USER_11H_DEADLINE'))
            break
        try:
            from . import extension_discovery,extension_runtime_controls,ar_recovery,extension_confirmation,extension_transport
            importlib.reload(extension_discovery);importlib.reload(extension_runtime_controls);importlib.reload(ar_recovery);importlib.reload(extension_confirmation);importlib.reload(extension_transport)
            extension_discovery.followups();extension_runtime_controls.memory_initial();extension_runtime_controls.context16k32_initial()
            runtime=extension_runtime_controls.advance()
            AR=ar_recovery.advance()
            wire=extension_transport.advance()
            confirmation=extension_confirmation.advance()
            status=dict(time=now,pid=os.getpid(),status='MONITORING_AND_ADVANCING',
                remaining_to_final_deadline_hours=(budget['final_deadline_epoch']-now)/3600,
                pending_configurations=len(pending_jobs()),controllers=processes('campaign'),dispatchers=processes('dispatch'),
                runtime_control_stages={key:value['stage'] for key,value in runtime.items()},AR_stage=AR['stage'],
                confirmation_stages={key:value['stage'] for key,value in confirmation.items()})
            status['wire_credit_stages']={key:value['stage'] for key,value in wire.items()}
            write(ROOT/'state/extension_manager_status.json',status)
            resume_if_needed()
            confirmations=ROOT/'state/extension_confirmation_terminal_20261102.json'
            ar=ROOT/'state/ar_debug_terminal_20261102.json'
            if confirmations.exists() and ar.exists() and not pending_jobs() and not processes('campaign') and not processes('dispatch'):
                write(ROOT/'state/extension_terminal_20261102.json',dict(time=time.time(),status='EXTENSION_TERMINAL',
                    confirmation=json.loads(confirmations.read_text()),AR=json.loads(ar.read_text())))
                break
        except Exception as error:
            import traceback
            append(ROOT/'state/extension_manager_errors.jsonl',dict(time=time.time(),error=repr(error),traceback=traceback.format_exc(),
                scope='Do not overwrite failed evidence or claim the stage passed. Existing managed GPU jobs remain active.'))
        time.sleep(15)


if __name__=='__main__':main()
