"""Freeze a matched, marker-free confirmation after recovery diagnostics."""
from .common import *
from .secondary import recovery_config


def main():
    initialize()
    jobs=[]
    seeds=(2026110931,2026110932,2026110933)
    for model in ('ouro','huginn'):
        for index,seed in enumerate(seeds):
            arms=['native','loop','token']
            arms=arms[index:]+arms[:index]
            for arm in arms:
                name=f'recovery_{model}_{arm}_seed{seed}'
                cfg=recovery_config(model,name,arm,128,seed)
                cfg.update(phase='FROZEN_RECOVERY_HELD_OUT_ARRIVAL_CONFIRMATION',
                    clean_confirmation=True,comparison_reference=f'recovery_{model}_native_seed{seed}',
                    primary_safety_arm=f'recovery_{model}_{arm}',
                    native_transport_limits=dict(shutdown_timeout=60))
                if arm!='native':
                    cfg['required_qualification']=f'secondary_{model}_{arm}_state32'
                save(cfg)
                jobs.append(name)
    write(ROOT/'recovery_plan.json',dict(jobs=jobs,seeds=seeds,created=time.time(),
        scope='Marker-free confirmation. The arrival seeds differ from calibration and selection.',
        prompt_scope='The same preserved 128 distinct prompt IDs remain in use. This is not a held-out text test.',
        fairness='All methods use the same pool, graph policy, prompt IDs, arrival trace, and frozen SLOs within each model.',
        policy='The v2 policy is frozen before these arrivals. No change follows an individual confirmation result.',
        safety='Stop later starts of an arm after an absolute Decode SLO failure. Report Native-relative tails separately.'))
    print(' '.join(jobs))


if __name__=='__main__':
    main()
