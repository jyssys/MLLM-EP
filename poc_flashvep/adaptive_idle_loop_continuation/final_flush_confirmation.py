"""Freeze paired trials after both forced-final-flush smoke controls passed."""
from .prepare import *
from .run import execute


def main():
    prepare();results=[]
    for start in (1,2,3):
        for model in ('ouro','huginn'):
            for mode in (('off','graph') if start%2 else ('graph','off')):
                cfg=configuration(model,f'a_{model}_heavy_{mode}_force_s{start}_n128',
                    seed=2026110660+start,observer=mode)
                cfg.update(idle_activity_device_context=True,idle_close_device_context=True,idle_forced_final_flush=True,
                    idle_observer_roles=['decode'],idle_queue_stride=16,idle_trace_seconds=8.,independent_start=start,
                    observer_selection='Frozen three-start final-flush confirmation. Same local-tail contract. No replacement of failed seeds.')
                save(cfg);results.append(dict(name=cfg['name'],complete=execute(cfg['name'])))
                write(ROOT/'state/final_flush_confirmation.json',dict(status='RUNNING',results=results))
    write(ROOT/'state/final_flush_confirmation.json',dict(status='COMPLETE',results=results))


if __name__=='__main__':main()
