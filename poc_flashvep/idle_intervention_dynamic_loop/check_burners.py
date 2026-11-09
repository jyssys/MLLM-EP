"""Check resident engine ownership on all eight GPUs, not wrapper PIDs alone."""
import argparse
import re
import subprocess
from .common import *


def main(restore=False):
    configure()
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    from ..looped_pd_autonomous_headroom.common import resources
    owner=resources()
    if restore:owner.restore(list(range(8)))
    old=ROOT/'state/final_burner_verification.json'
    initial=ROOT/'state/burner_startup_verification_initial.json'
    if old.exists() and not initial.exists() and not json.loads(old.read_text()).get('all_generation_started_and_completed_round'):
        write(initial,json.loads(old.read_text()))
    samples=[];engines={};generation={}
    for _ in range(3):
        wrappers=owner.burners();engines={gpu:[] for gpu in range(8)}
        unknown=[]
        for row in owner.processes():
            try:
                pid=row['pid'];proc=Path('/proc')/str(pid)
                args=(proc/'cmdline').read_bytes().decode().replace('\0',' ').strip()
                verified=(proc.stat().st_uid==os.getuid() and os.getsid(pid)==wrappers.get(row['gpu'])
                    and pid!=wrappers.get(row['gpu']))
                value=dict(pid=pid,session=os.getsid(pid),UID=proc.stat().st_uid,
                    command=args,ownership_verified=verified)
                engines[row['gpu']].append(value)
                if not verified:unknown.append(dict(gpu=row['gpu'],**value))
            except (FileNotFoundError,ProcessLookupError):pass
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'],text=True)
        values={}
        for line in output.splitlines():
            gpu,util,memory,power=[float(value.strip()) for value in line.split(',')]
            values[int(gpu)]=dict(utilization_percent=util,memory_MiB=memory,power_W=power)
        logs={};generation={}
        for gpu,pid in wrappers.items():
            path=(Path('/proc')/str(pid)/'fd/1').resolve()
            if path not in logs:logs[path]=path.read_text(errors='replace')
            text=logs[path];start=text.rfind(f'[GPU {gpu}] loading')
            window=text[start:] if start>=0 else ''
            matches=re.findall(r'\[GPU '+str(gpu)+r'\] rounds=(\d+) generated_tokens=(\d+) elapsed=([0-9.]+)s',window)
            generation[gpu]=dict(log=str(path),generation_started=f'[GPU {gpu}] generation started' in window,
                completed_rounds=int(matches[-1][0]) if matches else 0,
                generated_tokens=int(matches[-1][1]) if matches else 0,
                generation_elapsed_seconds=float(matches[-1][2]) if matches else None)
        samples.append(dict(time=time.time(),GPUs=values,generation=generation))
        time.sleep(2)
    generating=all(generation.get(gpu,{}).get('generation_started') and generation[gpu]['completed_rounds']>0 for gpu in range(8))
    verified=(not unknown and all(engines[gpu] and all(e['ownership_verified'] for e in engines[gpu])
        for gpu in range(8)) and generating
        and all(any(s['GPUs'][gpu]['utilization_percent']>0 for s in samples) for gpu in range(8)))
    result=dict(time=time.time(),all_eight_verified=verified,
        all_generation_started_and_completed_round=generating,
        wrappers=wrappers,engines=engines,generation=generation,unknown_processes=unknown,samples=samples,
        scope='Resident NVML compute engines match approved worker sessions and the current UID. Each latest worker logs actual generation and a completed round. Every GPU has nonzero activity in this sample window. This is not a perpetual utilization guarantee.')
    write(ROOT/'state'/('final_burner_verification.json' if verified else 'pending_burner_verification.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true')
    main(parser.parse_args().restore)
