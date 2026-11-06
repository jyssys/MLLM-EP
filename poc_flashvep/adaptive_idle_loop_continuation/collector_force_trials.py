"""Test final forced flush after documented incomplete-buffer behavior."""
from .prepare import *
from .run import execute


def main():
    prepare();results=[]
    for device in (0,1):
        cfg=configuration('ouro',f'collector_device1_flush{device}_launches10000_v2',count=8)
        cfg.update(worker='operator_micro',idle_worker='collector_selftest',activity_mode=3,
            selftest_device=1,selftest_flush_device=device,selftest_launches=10000,timeout=180,
            phase='COLLECTOR_COVERAGE_DIAGNOSIS_NOT_SERVING')
        save(cfg);results.append(dict(name=cfg['name'],complete=execute(cfg['name'])))
    for model in ('ouro','huginn'):
        cfg=configuration(model,f'collector_{model}_forced_final_smoke24',count=24,observer='graph')
        cfg.update(idle_activity_device_context=True,idle_observer_roles=['decode'],idle_close_device_context=True,
            idle_forced_final_flush=True,idle_queue_stride=16,phase='COLLECTOR_COVERAGE_DIAGNOSIS_NOT_CLEAN_SERVING')
        save(cfg);results.append(dict(name=cfg['name'],complete=execute(cfg['name'])))
    write(ROOT/'state/collector_force_trials.json',dict(status='COMPLETE',results=results))


if __name__=='__main__':main()
