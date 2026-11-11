"""Run diagnosis cells sequentially. Restore only approved burners on exit."""
import argparse,subprocess,traceback
from .common import *


def main():
    initialize()
    plan=json.loads((ROOT/'state/diagnostic_plan.json').read_text())
    p=argparse.ArgumentParser();p.add_argument('names',nargs='*');args=p.parse_args()
    results=[]
    try:
        for name in args.names or plan['jobs']:
            print('START',name,flush=True)
            result=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-m',
                'poc_flashvep.huginn_loop28_tail_diagnosis.run',name],cwd=REPO)
            item=dict(name=name,returncode=result.returncode,time=time.time())
            if result.returncode==0 and name=='tail_serial_state8':
                from ..huginn_net_benefit.common import analysis
                item['qualification']=analysis().qualify_frozen_reference(name)
                assert item['qualification']['passed'],'Serialized diagnostic changes the Native state'
            results.append(item);write(ROOT/'state/progress.json',dict(status='RUNNING',results=results))
            print('END',name,result.returncode,flush=True)
            if result.returncode:break
    finally:
        configure()
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources()
        if not owner.processes():owner.restore(list(range(8)))
        write(ROOT/'state/progress.json',dict(status='CLOSED',results=results,time=time.time()))


if __name__=='__main__':main()
