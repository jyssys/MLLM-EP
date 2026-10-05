"""Scoped process-group watchdog; restore only a pre-existing burner policy."""
import os
import signal
import subprocess
import time
from pathlib import Path
from poc_dllm_attn_moe_pipeline.contract import *


def guarded_run(command, env, log, output, timeout):
    import poc_native_lp_ep4.scripts.gpu_session as old
    old.MASK = MASK
    old.PROJECT_ROOT = PROJECT
    burners = []
    restore_environments = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [x.decode() for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
            if (entry.stat().st_uid == os.getuid() and
                '/home/esjung/vllm-ep/utilize.py' in argv and '--gpus' in argv and
                argv[argv.index('--gpus')+1] == MASK):
                pid = int(entry.name)
                burners.append(dict(pid=pid, owner_uid=os.getuid(), args=argv, mask=MASK))
                restore_environments[pid] = dict(x.decode().split('=',1) for x in
                    (entry/'environ').read_bytes().split(b'\0') if b'=' in x)
        except (FileNotFoundError,PermissionError):
            continue
    write(output/'gpu_policy.json',dict(original=burners,restore_only_original=True))
    proc = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        # Restoration also covers exceptions during shutdown/competing checks.
        for burner in burners:
            os.kill(burner['pid'],signal.SIGTERM)
        deadline = time.monotonic()+40
        while old.compute_pids() and burners and time.monotonic()<deadline:
            time.sleep(1)
        if old.compute_pids():
            raise RuntimeError(f'Competing target GPU processes: {old.compute_pids()}')
        proc = subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return proc.wait(timeout=timeout)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait(timeout=15)
        # Never invoke the default launcher or create a service when initialOFF.
        # Use the exact pre-existing arguments/environment, not a guessed policy.
        if burners and not old.compute_pids():
            for burner in burners:
                with (output/'restored_burner.log').open('a') as restored_log:
                    subprocess.Popen(burner['args'], env=restore_environments[burner['pid']],
                        cwd='/home/esjung/vllm-ep', stdout=restored_log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True)
        elif burners:
            write(output/'restoration_blocked.json',dict(reason='Assigned GPU processes remain; no duplicate service',
                                                        original=burners))
