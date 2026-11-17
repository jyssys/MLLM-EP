"""Stop only an owned serving job after a confirmed idle-progress stall."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .common import ROOT,write
from .analysis import records


def main():
    seen={};events=[];heartbeat=0.
    write(ROOT/'state/stall_guard_owner.json',dict(pid=os.getpid(),uid=os.getuid(),time=time.time(),
        progress_limit_s=180,sampling_s=10,consecutive_idle_samples=8,
        scope='Only this namespace\'s UID-owned coordinator can receive SIGTERM. The dispatcher cleans up its owned workers. No unknown GPU process or burner is terminated.'))
    while not (ROOT/'state/CAMPAIGN_CLOSED.json').exists():
        if time.time()-heartbeat>=60:
            heartbeat=time.time();print('STALL_GUARD_ACTIVE',len(events),flush=True)
        process=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
        for line in process.splitlines():
            if ' -u -m poc_flashvep.dense_ar_layer_token_pd.' not in line or str(ROOT/'configs') not in line:
                continue
            parts=line.split();pid=int(parts[0]);config=Path(parts[-1])
            if config.parent!=ROOT/'configs' or not config.exists():
                continue
            folder=ROOT/'raw'/config.stem
            if not (folder/'measurement_begin.json').exists() or (folder/'dispatcher_result.json').exists():
                continue
            observations=records(folder/'queue_observations.jsonl')
            if not observations:
                continue
            latest=observations[-1];completed=latest['completed'];now=time.time()
            state=seen.setdefault(pid,dict(completed=completed,since=now,idle=0))
            if completed!=state['completed']:
                state.update(completed=completed,since=now,idle=0)
            query=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=10)
            if query.returncode:
                continue
            utilization={int(a):int(b) for a,b in (row.split(',') for row in query.stdout.splitlines())}
            state['idle']=state['idle']+1 if all(utilization[g]<=1 for g in (0,1)) else 0
            if now-state['since']<180 or state['idle']<8 or latest['outstanding']<=0:
                continue
            proc=Path(f'/proc/{pid}')
            if not proc.exists() or proc.stat().st_uid!=os.getuid():
                continue
            command=(proc/'cmdline').read_bytes()
            if str(config).encode() not in command or b'poc_flashvep.dense_ar_layer_token_pd.' not in command:
                continue
            # Check GPU ownership through the process ancestor chain.
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            def descendant(child):
                visited=set()
                while child>1 and child not in visited:
                    if child==pid:
                        return True
                    visited.add(child)
                    try:
                        fields=Path(f'/proc/{child}/stat').read_text().rsplit(')',1)[1].split()
                        child=int(fields[1])
                    except (FileNotFoundError,IndexError,ValueError):
                        return False
                return False
            app_ids=[int(line.strip()) for line in apps.splitlines() if line.strip().isdigit()]
            if not app_ids or not all(descendant(child) for child in app_ids):
                continue
            snapshot=dict(time=now,pid=pid,status='FAILED',utilization=utilization,
                unchanged_completion_s=now-state['since'],consecutive_idle_samples=state['idle'],
                queue_observation=latest,command=command.decode().replace(chr(0),' '),
                reason='Idle-progress stall. Preserve this run. Stop only its coordinator and let the owning dispatcher retire its workers.')
            write(folder/'automatic_stall_guard.json',snapshot)
            events.append(snapshot);write(ROOT/'state/stall_guard_events.json',events)
            os.kill(pid,signal.SIGTERM);seen.pop(pid,None)
            print('STOP_OWNED_STALLED_COORDINATOR',pid,config.stem,flush=True)
        time.sleep(10)
    print('STALL_GUARD_CLOSED',flush=True)


if __name__=='__main__':
    main()
