"""Run an explicit serial job list and restore approved Burners on exit."""
import argparse
import subprocess
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args()
    initialize();results=[];unsafe=set()
    write(ROOT/'state/controller.json',dict(pid=os.getpid(),names=a.names,status='RUNNING',time=time.time()))
    try:
        for name in a.names:
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            if cfg.get('primary_safety_arm') in unsafe:
                results.append(dict(name=name,status='NOT_RUN_ABSOLUTE_DECODE_SAFETY_STOP',time=time.time()))
                write(ROOT/'state/sequence_latest.json',results)
                continue
            if cfg.get('required_qualification'):
                required=ROOT/'raw'/cfg['required_qualification']/'qualification_with_frozen_reference.json'
                assert required.exists(),'Qualification evidence does not exist'
                gate=json.loads(required.read_text())
                assert gate.get('campaign_gate_passed',gate['passed']),'Qualification gate did not pass'
            result=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-m',
                'poc_flashvep.completion_aware_loop_pd.run',name],cwd=REPO)
            results.append(dict(name=name,returncode=result.returncode,time=time.time()))
            write(ROOT/'state/sequence_latest.json',results)
            print(json.dumps(results[-1]),flush=True)
            if result.returncode:
                if cfg.get('continue_after_failure'):continue
                break
            if cfg.get('primary_safety_arm'):
                from .analyze import metrics
                measured=metrics(name)
                if not measured.get('Decode_SLO_screen'):
                    unsafe.add(cfg['primary_safety_arm'])
            if cfg.get('state_probe'):
                from .analyze import qualify_frozen_reference
                qualification=qualify_frozen_reference(name)
                qualification['campaign_gate_passed']=(qualification['passed'] and
                    (qualification.get('actual_dynamic_continuation_tested',False)
                        if cfg.get('require_actual_offload_qualification') else True))
                write(ROOT/'raw'/name/'qualification_with_frozen_reference.json',qualification)
                print('STATE_QUALIFICATION',name,qualification['passed'],qualification.get('offload'),flush=True)
                if not qualification['campaign_gate_passed']:break
    finally:
        configure()
        from ..looped_pd_autonomous_headroom.common import resources
        owner=resources()
        if not owner.processes():owner.restore(list(range(8)))
        write(ROOT/'state/controller.json',dict(pid=os.getpid(),status='CLOSED',results=results,time=time.time()))


if __name__=='__main__':main()
