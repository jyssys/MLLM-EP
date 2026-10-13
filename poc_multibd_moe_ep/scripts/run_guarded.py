"""Reuse the validated scoped watchdog/burner restoration contract."""
import argparse
import os
import json
import signal
import subprocess
import time
from pathlib import Path
from poc_multibd_moe_ep.contract import *


def finish_scoped_cleanup(output):
    import poc_native_lp_ep4.scripts.gpu_session as old
    old.MASK=MASK
    # Parent exit can leave rank 0 inside llm.exit() after other ranks die.
    # Match only this PoC's process environment/argv, never a foreign user.
    def owns(pid):
        try:
            proc=Path(f'/proc/{pid}')
            cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
            env=(proc/'environ').read_bytes()
            return proc.stat().st_uid==os.getuid() and (
                b'poc_multibd_moe_ep.scripts.' in cmd or
                ('POC_MULTIBD_OUTPUT='+str(ROOT)+'/').encode() in env)
        except (FileNotFoundError,PermissionError):return False
    remaining=[pid for pid in old.compute_pids() if owns(pid)]
    for pid in remaining:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass
    deadline=time.monotonic()+15
    while any(owns(pid) for pid in old.compute_pids()) and time.monotonic()<deadline:
        time.sleep(1)
    policy=output/'gpu_policy.json'
    if policy.exists() and json.loads(policy.read_text())['original'] and not old.compute_pids():
        # Do not duplicate an already-restored supervisor still loading on CPU.
        present=False
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:
                args=(entry/'cmdline').read_bytes().replace(b'\0',b' ')
                present |= (entry.stat().st_uid==os.getuid() and
                    ('utilize.py --gpus '+MASK+' ').encode() in args)
            except (FileNotFoundError,PermissionError):pass
        if not present:
            env=dict(os.environ,VLLM_UTILIZE_GPUS=MASK,CUDA_VISIBLE_DEVICES=MASK)
            env.pop('PYTHONPATH',None)
            with (output/'restored_burner_fallback.log').open('a') as log:
                proc=subprocess.Popen(['bash','/home/esjung/vllm-ep/run_utilize.sh'],
                    env=env,cwd='/home/esjung/vllm-ep',stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(output/'restored_burner_fallback.json',dict(pid=proc.pid,mask=MASK))
    write(output/'cleanup_status.json',dict(terminated_owned_orphans=remaining,
        target_GPU_processes=sorted(old.compute_pids()),foreign_users_untouched=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    import poc_dllm_attn_moe_pipeline.scripts.gpu_session as guard
    # Override only this imported guard instance, not the prior PoC on disk.
    guard.PROJECT = PROJECT
    guard.MASK = MASK
    output = ROOT/a.output
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=MASK, TORCHDYNAMO_DISABLE='1',
        PYTHONPATH=str(PROJECT)+':'+str(UPSTREAM)+':/home/esjung/external/multibd-poc-deps',
        OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    with (output/'console.log').open('w') as log:
        try:
            code = guard.guarded_run(command,env,log,output,a.timeout)
            status = dict(exit_code=code, success=code==0)
        except Exception as e:
            status = dict(exit_code=None, success=False, error=repr(e))
        except KeyboardInterrupt:
            status = dict(exit_code=None, success=False, interrupted=True)
        finally:
            finish_scoped_cleanup(output)
    write(output/'guard_status.json',status)
    print(status, flush=True)
    raise SystemExit(0 if status['success'] else 1)


if __name__ == '__main__':
    main()
