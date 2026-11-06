"""Finish Native load controls and quantify A/A tails without changing gates."""
import copy
import time
from .prepare import *
from .run import execute


def main():
    prepare();results=[]
    while time.time()<LONG_START_CUTOFF:
        path=ROOT/'state/forced_mixed_qualification.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='COMPLETE':break
        time.sleep(5)
    for model in ('ouro','huginn'):
        for load in ('low','medium'):
            if time.time()>=LONG_START_CUTOFF:break
            cfg=configuration(model,f'a_{model}_{load}_off_load_control_n128',load=load,observer='off')
            save(cfg);results.append(dict(name=cfg['name'],complete=execute(cfg['name'])))
            write(ROOT/'state/finish_controls.json',dict(status='RUNNING',results=results))
    for start in (2,3):
        for model in ('ouro','huginn'):
            if time.time()>=LONG_START_CUTOFF:break
            base=f'a_{model}_heavy_off_minimal_pilot_n128'
            cfg=copy.deepcopy(json.loads((ROOT/'configs'/(base+'.json')).read_text()))
            cfg.update(name=f'diag_{model}_native_minimal_AA_start{start}_n128',
                phase='NATIVE_AA_MINIMAL_HOST_OBSERVER_CONTROL',independent_start=start)
            save(cfg);complete=execute(cfg['name']);result=dict(name=cfg['name'],complete=complete)
            if complete:
                from .analyze import matched
                from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
                with quiet_analysis('Native_AA_minimal_observer_tail_check'):
                    result['comparison']=matched(cfg['name'],base)
            results.append(result)
            write(ROOT/'state/finish_controls.json',dict(status='RUNNING',results=results))
    write(ROOT/'state/finish_controls.json',dict(status='COMPLETE',results=results,
        scope='Matched Native A/A tails and normal-load controls. A/A variation does not override or relax the frozen instrumentation gate. No adaptive serving is claimed.'))


if __name__=='__main__':main()
