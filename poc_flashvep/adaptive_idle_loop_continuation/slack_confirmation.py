"""Freeze three paired starts for the context-corrected low-record-rate observer."""
import time
from .prepare import *
from .run import execute


def main():
    prepare()
    while time.time()<LONG_START_CUTOFF:
        path=ROOT/'state/primitive_full.json'
        if path.exists() and json.loads(path.read_text())['status']=='COMPLETE':break
        time.sleep(5)
    results=[]
    for start in (1,2,3):
        for model in ('ouro','huginn'):
            for observer in (('off','graph') if start%2 else ('graph','off')):
                name=f'a_{model}_heavy_{observer}_cctx_s{start}_n128'
                cfg=configuration(model,name,seed=2026110620+start,observer=observer)
                cfg.update(idle_activity_device_context=True,idle_observer_roles=['decode'],
                    idle_queue_stride=16,idle_trace_seconds=8.,independent_start=start,
                    observer_selection='Frozen context-corrected graph envelopes plus non-graph kernels. Three paired starts. No result-based seed replacement.')
                save(cfg);write(ROOT/'state/slack_confirmation.json',dict(time=time.time(),status='RUNNING',name=name,results=results))
                results.append(dict(name=name,complete=execute(name)))
    for model in ('ouro','huginn'):
        for load in ('low','medium'):
            cfg=configuration(model,f'a_{model}_{load}_off_load_control_n128',load=load,observer='off')
            save(cfg);results.append(dict(name=cfg['name'],complete=execute(cfg['name'])))
    write(ROOT/'state/slack_confirmation.json',dict(time=time.time(),status='COMPLETE',results=results))


if __name__=='__main__':main()
