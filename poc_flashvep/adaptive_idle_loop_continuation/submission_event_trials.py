"""Qualify complete model-stream enclosures without per-kernel CUPTI records."""
from .prepare import *
from .run import execute


def main():
    prepare();results=[]
    for start in (1,2,3):
        for model in ('ouro','huginn'):
            for mode in (('off','submit_events') if start%2 else ('submit_events','off')):
                cfg=configuration(model,f'a_{model}_heavy_{mode}_submit_s{start}_n128',
                    seed=2026110640+start,observer=mode)
                cfg.update(idle_observer_roles=['decode'],idle_queue_stride=16,idle_trace_seconds=8.,
                    independent_start=start,observer_selection='Frozen all-submit stream enclosures. Three paired starts. No per-kernel activity collection.')
                save(cfg);results.append(dict(name=cfg['name'],complete=execute(cfg['name'])))
                write(ROOT/'state/submission_event_trials.json',dict(status='RUNNING',results=results))
    write(ROOT/'state/submission_event_trials.json',dict(status='COMPLETE',results=results))


if __name__=='__main__':main()
