"""Run bounded immutable cells. Stop this sequence at a failed state gate."""
import argparse,json,fcntl
from .common import *
from .run import execute


def main():
    initialize();p=argparse.ArgumentParser();p.add_argument('names',nargs='+');args=p.parse_args();records=[]
    try:
        for name in args.names:
            print('START',name,flush=True);ok=execute(name)
            entry=dict(name=name,execution_complete=ok,time=__import__('time').time())
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            quantum=None
            if ok and cfg.get('require_realized_quantum_contract'):
                from ..loop_width_gemm_validation import quantum_contract
                quantum_contract.ROOT=ROOT;quantum=quantum_contract.audit(name)
                entry['quantum_contract']=quantum;ok=quantum['passed']
            if ok and cfg.get('state_probe'):
                if cfg.get('qualification_fresh_native') or cfg.get('campaign_state_reference'):
                    from .fresh_qualification import qualify
                    value=qualify(name)
                else:
                    from ..huginn_net_benefit.common import analysis
                    value=analysis().qualify_frozen_reference(name)
                exits=[json.loads(p.read_text()) for p in (ROOT/'raw'/name).glob('native_worker_exit_*.json')]
                normal=any(x['passed'] and {r['role'] for r in x['records'] if r['exitcode']==0 and not r['forced_cleanup']}=={'prefill','decode'} for x in exits)
                value['campaign_gate_passed']=value['passed'] and normal and (quantum is None or quantum['passed'])
                write(ROOT/'raw'/name/'qualification_with_frozen_reference.json',value)
                entry['qualification']=value;ok=value['campaign_gate_passed']
            entry['passed']=ok;records.append(entry)
            write(ROOT/'state/sequence_progress.json',dict(results=records,closed=False))
            print('END',name,ok,flush=True)
            if not ok:break
    finally:
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources()
        with (ROOT/'state/quiet_lane.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not owner.processes():owner.restore(list(range(8)))
        write(ROOT/'state/sequence_progress.json',dict(results=records,closed=True))
    return 0 if len(records)==len(args.names) and all(r['passed'] for r in records) else 2


if __name__=='__main__':raise SystemExit(main())
