"""Scope burner shutdown/restoration to the explicitly authorized GPU UUIDs."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

MASK="0,1,4,5"
PROJECT_ROOT=Path(__file__).resolve().parents[2]


def compute_pids():
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    uuids={line.split(",")[1].strip() for line in rows.splitlines() if line.split(",")[0].strip() in MASK.split(",")}
    apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
    return {int(line.split(",")[1]) for line in apps.splitlines() if line.split(",")[0].strip() in uuids}


def owned_experiment(pid):
    try:
        proc=Path(f"/proc/{pid}")
        ancestor=pid
        for _ in range(8):
            current=Path(f'/proc/{ancestor}')
            cmd=(current/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if '/home/esjung/vllm-ep/utilize.py --gpus '+MASK+' ' in cmd:
                return False
            status=(current/'status').read_text()
            ancestor=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
            if ancestor<=1:break
        env=(proc/"environ").read_bytes().split(b"\0")
        # vLLM's process-title update can erase early environment strings.
        prefix=('NATIVE_LP_OUTPUT='+str(PROJECT_ROOT/'poc_native_lp_ep4')+'/').encode()
        scoped=any(value.startswith(prefix) for value in env)
        scoped |= (proc/"cwd").resolve()==PROJECT_ROOT
        return proc.stat().st_uid==os.getuid() and scoped
    except (FileNotFoundError,ProcessLookupError,PermissionError):return False


def stop_burner():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit(): continue
        try:
            cmd=(entry/"cmdline").read_bytes().replace(b"\0",b" ").decode()
            if "/home/esjung/vllm-ep/utilize.py --gpus "+MASK+" " in cmd and entry.stat().st_uid==os.getuid():
                os.kill(int(entry.name),signal.SIGTERM)
        except (ProcessLookupError,FileNotFoundError,PermissionError): pass
    deadline=time.monotonic()+40
    while compute_pids() and time.monotonic()<deadline: time.sleep(1)
    for pid in compute_pids():
        if owned_experiment(pid):
            os.kill(pid,signal.SIGKILL)
    if compute_pids(): time.sleep(5)
    if compute_pids(): raise RuntimeError(f"competing GPU processes: {compute_pids()}")


def restore():
    if compute_pids():
        print("RESTORE_BLOCKED: GPU processes remain",flush=True)
        return
    env=os.environ.copy();env["VLLM_UTILIZE_GPUS"]=MASK
    env.pop("PYTHONPATH",None)
    with open("/tmp/native_lp_burner.log","a") as log:
        proc=subprocess.Popen(["bash","/home/esjung/vllm-ep/run_utilize.sh"],env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            cwd='/home/esjung/vllm-ep')
    print(f"BURNER_RESTORED {proc.pid} {MASK}",flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--timeout",type=int,default=1800)
    p.add_argument("command",nargs=argparse.REMAINDER);args=p.parse_args()
    if not args.command: raise ValueError("command required")
    # Serialize this PoC's stop/run/restore transactions, including teardown.
    session_lock=open('/tmp/native_lp_gpu_session.lock','a')
    deadline=time.monotonic()+90
    while True:
        try:
            fcntl.flock(session_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic()>=deadline:raise TimeoutError('another native LP GPU session owns the scoped lock')
            time.sleep(2)
    stop_burner()
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=MASK
    proc=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        proc=subprocess.Popen(args.command,env=env,start_new_session=True)
        deadline=time.monotonic()+args.timeout
        while True:
            try:
                code=proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                status=Path(env['NATIVE_LP_OUTPUT'])/'status.json'
                if status.exists() and not json.loads(status.read_text()).get('ok',False):
                    code=1
                    break
                if time.monotonic()>=deadline:
                    raise TimeoutError('scoped GPU session watchdog')
    finally:
        if proc:
            # Workers may outlive their parent on teardown. This session owns
            # precisely this newly created process group.
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            deadline=time.monotonic()+20
            while compute_pids() and time.monotonic()<deadline: time.sleep(1)
            if compute_pids():
                # Exact target GPU PIDs are checked for ownership before kill.
                for pid in compute_pids():
                    try:
                        if owned_experiment(pid): os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                time.sleep(5)
                for pid in compute_pids():
                    if owned_experiment(pid):
                        try: os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                time.sleep(3)
        restore()
    raise SystemExit(code)


if __name__=="__main__": main()
