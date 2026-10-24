"""Run registered stages. Stop on failure and preserve each attempt."""
import argparse
import json
import subprocess
import sys
import time
from .common import ROOT,REPO,write

def main():
    p=argparse.ArgumentParser();p.add_argument('plans',nargs='+');p.add_argument('--after',default=None);a=p.parse_args()
    if a.after:
        plan=json.loads((ROOT/'plans'/f'{a.after}.json').read_text())
        jobs=plan.get('jobs',[])
        until=time.monotonic()+5400
        while not all((ROOT/'raw'/j['id']/'completed.json').exists() for j in jobs):
            assert not any((ROOT/'raw'/j['id']/'failure.json').exists() for j in jobs),'Prior attempt failed.'
            assert time.monotonic()<until,'Prior stage did not finish.'
            time.sleep(2)
        # Release follows completion hashing and process teardown.
        from .resources import processes,burners
        until=time.monotonic()+300
        while any(r['gpu'] not in burners() for r in processes()):
            assert time.monotonic()<until
            time.sleep(2)
    for name in a.plans:
        import fcntl
        (ROOT/'state').mkdir(exist_ok=True)
        lease=(ROOT/'state'/'dispatcher.lock').open('a')
        until=time.monotonic()+5400
        while True:
            try:
                fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN);lease.close();break
            except BlockingIOError:
                assert time.monotonic()<until
                time.sleep(2)
        command=[sys.executable,'-m','poc_flashvep.ouro_kv_residency.dispatch',
            '--manifest',str(ROOT/'plans'/f'{name}.json'),'--manage-burners']
        write(ROOT/'state'/'campaign.json',dict(stage=name,command=command,start=time.time()))
        subprocess.run(command,cwd=REPO,check=True)
    write(ROOT/'state'/'campaign.json',dict(status='COMPLETE',plans=a.plans,end=time.time()))

if __name__=='__main__':main()
