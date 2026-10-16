"""Cleanup only the job session that this guard directly created."""
import os
import signal
import subprocess
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def registered_launch(original,expected,output,groups):
    def launch(command,*args,**kwargs):
        process=original(command,*args,**kwargs)
        environment=kwargs.get('env') or {}
        if (list(command)==expected and kwargs.get('start_new_session') is True
                and environment.get('LOOPED_KV_OUTPUT')==str(output)):
            groups.append(process.pid)
        return process
    return launch


def belongs(pid,group):
    try:
        return ((Path('/proc')/str(pid)).stat().st_uid==os.getuid()
                and os.getpgid(pid)==group and os.getsid(pid)==group)
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return False


def pending_restores(output):
    status=output/'guard_status.json'
    if not status.exists():
        return set()
    pending=set()
    for row in read(status).get('restored_burner_workers',[]):
        try:
            process=Path('/proc')/str(row['pid'])
            if process.stat().st_uid!=os.getuid():
                continue
            words=[part.decode() for part in (process/'cmdline').read_bytes().split(b'\0') if part]
            if not any(path in words for path in ['/home/esjung/vllm-ep/utilize.py',
                                                   '/home/esjung/vllm-ep/run_utilize.sh']):
                continue
            devices=row['physical_gpu']
            pending.update(devices if isinstance(devices,list) else [devices])
        except (FileNotFoundError,PermissionError):
            continue
    return pending


def finish(groups,gpus,output,contexts):
    killed=[]
    reclaimed=set()
    for group in groups:
        affected={gpu for gpu in gpus if any(belongs(pid,group) for pid in contexts([gpu]))}
        if affected:
            try:
                os.killpg(group,signal.SIGKILL)
                killed.append(group)
                reclaimed.update(affected)
            except ProcessLookupError:
                pass
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        active=contexts(gpus)
        owned=any(belongs(pid,group) for pid in active for group in groups)
        stale=any(not (Path('/proc')/str(pid)).exists() for pid in active)
        if not owned and not stale:
            break
        time.sleep(.25)
    remaining=[pid for pid in contexts(gpus) if any(belongs(pid,group) for group in groups)]
    # A restored CPU-loading worker can have no GPU context yet.
    # A late model-context release can also leave a genuinely idle GPU.
    pending=pending_restores(output)
    eligible=set(gpus) if (output/'guard_status.json').exists() else reclaimed
    idle=[gpu for gpu in sorted(eligible-pending) if not contexts([gpu])]
    launcher=None
    if idle:
        directory=ASSETS/'burner_logs'
        directory.mkdir(parents=True,exist_ok=True)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,idle)),
                         VLLM_UTILIZE_GPUS=','.join(map(str,idle)))
        environment.pop('PYTHONPATH',None)
        environment.pop('LOOPED_KV_OUTPUT',None)
        with (directory/f'{output.name}.post_cleanup.log').open('a') as log:
            process=subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'],
                cwd='/home/esjung/vllm-ep',env=environment,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        launcher=process.pid
    write(output/'guard_cleanup_validation.json',dict(target_GPUs=gpus,registered_job_groups=groups,
        force_stopped_owned_groups=killed,owned_GPU_contexts_remaining=remaining,
        reclaimed_GPUs=sorted(reclaimed),
        pending_original_restore_GPUs=sorted(pending&set(gpus)),
        scoped_idle_burner_restore=idle,launcher_PID=launcher,
        unrelated_process_groups_untouched=True,
        scope='Post-exit verification. It does not replace the final burner utilization check.'))
    if remaining:
        raise RuntimeError('An owned GPU context survived scoped cleanup. No foreign process was signalled.')
