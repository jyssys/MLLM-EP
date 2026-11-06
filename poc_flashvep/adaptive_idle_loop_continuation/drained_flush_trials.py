"""Test all-device drain and active final flush, after a forced-export failure."""
from .prepare import *
from .run import execute


def main():
    prepare();results=[]
    for model in ('ouro','huginn'):
        for mode in ('off','graph'):
            cfg=configuration(model,f'a_{model}_heavy_{mode}_drain_pilot_n128',seed=2026110671,observer=mode)
            cfg.update(idle_activity_device_context=True,idle_close_device_context=True,idle_forced_final_flush=True,
                idle_drain_all_device_before_export=True,idle_observer_roles=['decode'],idle_queue_stride=16,
                phase='ALL_DEVICE_DRAIN_AND_ACTIVE_FINAL_FLUSH_PILOT')
            save(cfg);results.append(dict(name=cfg['name'],complete=execute(cfg['name'])))
            write(ROOT/'state/drained_flush_trials.json',dict(status='RUNNING',results=results))
    write(ROOT/'state/drained_flush_trials.json',dict(status='COMPLETE',results=results))


if __name__=='__main__':main()
