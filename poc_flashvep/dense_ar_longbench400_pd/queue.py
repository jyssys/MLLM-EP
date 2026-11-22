"""Execute frozen job groups and checkpoint each completed experiment."""
import argparse
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,dispatch_allowed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest');args=parser.parse_args()
    path=ROOT/'jobs'/args.manifest;plan=json.loads(path.read_text());results=[]
    state=ROOT/'state'/(path.stem+'_queue.json')
    if state.exists():results=json.loads(state.read_text())['results']
    finished={r['name'] for r in results}
    for name in plan['jobs']:
        if name in finished:continue
        budget=json.loads((ROOT/'state/budget.json').read_text())
        while (ROOT/'state'/('PAUSE_QUEUE_'+path.stem+'.json')).exists():
            if not dispatch_allowed(budget):break
            time.sleep(1)
        if not dispatch_allowed(budget):break
        began=time.time();print('START',name,flush=True)
        code=subprocess.run([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_longbench400_pd.run',name],cwd=REPO).returncode
        row=dict(name=name,returncode=code,complete=code==0,elapsed_seconds=time.time()-began)
        results.append(row);write(state,dict(time=time.time(),results=results,expected=len(plan['jobs'])))
        print('END',name,code,flush=True)
        if code and plan.get('stop_on_failure',True):break
    write(state,dict(time=time.time(),results=results,expected=len(plan['jobs']),
        all_complete=len(results)==len(plan['jobs']) and all(r['complete'] for r in results)))


if __name__=='__main__':main()
