"""Separate live HBM-query observation from model and transfer computation."""
from .prepare import *
from .run import execute


def main():
    prepare();results=[]
    for model in ('ouro','huginn'):
        variants=[('full_host',f'diag_{model}_native_full_host_pilot_n128','off'),
            ('light_hbm',f'diag_{model}_native_light_hbm_pilot_n128','off'),
            ('minimal',f'a_{model}_heavy_off_minimal_pilot_n128','off'),
            ('minimal_events',f'a_{model}_heavy_submit_events_minimal_pilot_n128','submit_events')]
        reference=None
        for variant,name,observer in variants:
            cfg=configuration(model,name,seed=2026110691,observer=observer)
            cfg.update(observe_python_gc=True,observe_host_pauses=True,phase='HOST_OBSERVER_COST_DIAGNOSIS',
                observer_variant=variant,idle_observer_roles=['decode'],idle_queue_stride=16)
            if variant!='full_host':cfg.update(memory_sample_seconds=1e12,idle_light_owner=True)
            if variant in ('minimal','minimal_events'):cfg['observe_transport']=False
            save(cfg);complete=execute(name);result=dict(name=name,complete=complete)
            if complete:
                from ..looped_pd_atomic_prefill import evidence
                from ..dynamic_loop_split_serving.lifetime_gate import check
                from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
                with quiet_analysis('minimal_host_observer_native_contract'):
                    result.update(output=evidence.compare(reference,name) if reference else dict(passed=True),
                        work=evidence.full_work(name),lifetime=check(name))
                result['passed']=all(result[k]['passed'] for k in ('output','work','lifetime'))
                if reference is None:reference=name
            results.append(result);write(ROOT/'state/host_observer_controls.json',dict(status='RUNNING',results=results))
    write(ROOT/'state/host_observer_controls.json',dict(status='COMPLETE',results=results,
        scope='Matched Native observer ablations. No model, KV lifetime, scheduling policy, weights, or SLO changes. Correlation with GC or long host calls is not proof of cause.'))


if __name__=='__main__':main()
