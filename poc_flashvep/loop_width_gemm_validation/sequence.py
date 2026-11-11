import argparse,subprocess
from .common import *


def main():
    initialize();p=argparse.ArgumentParser();p.add_argument('names',nargs='+');args=p.parse_args();results=[]
    try:
        for name in args.names:
            print('START',name,flush=True)
            code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
                'poc_flashvep.loop_width_gemm_validation.run',name],cwd=REPO).returncode
            item=dict(name=name,returncode=code,time=time.time())
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if code==0 and cfg.get('require_realized_quantum_contract'):
                from .quantum_contract import audit
                item['quantum_contract']=audit(name)
                if not item['quantum_contract']['passed']:item['returncode']=2
            if code==0 and cfg.get('state_probe'):
                from ..huginn_net_benefit.common import analysis
                result=analysis().qualify_frozen_reference(name)
                receipts=[json.loads(p.read_text()) for p in (ROOT/'raw'/name).glob('native_worker_exit_*.json')]
                result['native_worker_exit_integrity']=any(x['passed'] and {r['role'] for r in x['records'] if r['exitcode']==0 and not r['forced_cleanup']}=={'prefill','decode'} for x in receipts)
                result['campaign_gate_passed']=result['passed'] and result['native_worker_exit_integrity'] and item['returncode']==0
                write(ROOT/'raw'/name/'qualification_with_frozen_reference.json',result)
                item['qualification']=result
                if not result['campaign_gate_passed']:item['returncode']=2
            results.append(item);write(ROOT/'state/runtime_progress.json',dict(results=results,closed=False))
            print('END',name,item['returncode'],flush=True)
            if item['returncode']:break
    finally:
        configure()
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources()
        if not owner.processes():owner.restore(list(range(8)))
        write(ROOT/'state/runtime_progress.json',dict(results=results,closed=True,time=time.time()))
    return 0 if len(results)==len(args.names) and all(r['returncode']==0 for r in results) else 2


if __name__=='__main__':raise SystemExit(main())
