"""Bounded sequential completion of the already frozen validation plan."""
import json
import subprocess
import time

from poc_speculative_moe_ep.contract import ROOT,PYTHON


def main():
    deadline=time.monotonic()+1200
    while True:
        states=[ROOT/'raw'/f'phase_b_key_reverse_v1_g{g}'/'status.json' for g in (8,4,2,0)]
        ready=[json.loads(p.read_text()) if p.exists() else None for p in states]
        if any(s is not None and not s['ok'] for s in ready):raise RuntimeError('reverse block failed; preserved, no retry')
        if all(s and s['ok'] for s in ready):break
        if time.monotonic()>deadline:raise TimeoutError('bounded wait for reverse block')
        time.sleep(3)
    for prefix,gammas,lane,repeats in (
        ('phase_b_bypass_v1','0,2,4,8','bypass',3),
        ('phase_b_route_repeat_v1','2,8','trace',1)):
        subprocess.run([PYTHON,'-m','poc_speculative_moe_ep.scripts.campaign','--prefix',prefix,
            '--gammas',gammas,'--lane',lane,'--requests','2','--repeats',str(repeats),'--tokens','64'],
            cwd=ROOT.parent,check=True)


if __name__=='__main__':main()
