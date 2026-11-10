"""Freeze three paired arrival starts without tuning on confirmation results."""
from .prepare import prepare,config
from .calibrate import new_config
from .common import *


def main():
    prepare();jobs=[];seeds=(2026110911,2026110912,2026110913)
    for model in ('ouro','huginn'):
        for index,seed in enumerate(seeds):
            arms=['native','old','loop','token'];arms=arms[index:]+arms[:index]
            for arm in arms:
                name=f'primary_{model}_{arm}_seed{seed}'
                cfg=(new_config(model,name,arm,count=128,seed=seed) if arm in ('loop','token')
                    else config(model,name,arm,count=128,seed=seed))
                cfg.update(phase='HELD_OUT_ARRIVAL_PRIMARY_CONFIRMATION',clean_confirmation=True,
                    comparison_reference=f'primary_{model}_native_seed{seed}',
                    primary_safety_arm=f'{model}_{arm}',native_transport_limits=dict(shutdown_timeout=60))
                if arm in ('loop','token'):cfg['required_qualification']=f'q2r2_{model}_{arm}_state32'
                save(cfg);jobs.append(name)
    write(ROOT/'primary_plan.json',dict(seeds=seeds,jobs=jobs,created=time.time(),
        text_scope='The same preserved 128 distinct prompt IDs appear in every start. Only arrival seeds are held out.',
        safety='Stop later starts of an arm after an absolute Decode SLO failure. Report the Native-relative 5% target independently.',
        tuning='No tuning on these arrival seeds. Negative results remain in the comparison.',
        loop_scope='One-way full-prompt recurrent suffix ownership. Execution may use qualified token tiles and loop-boundary yields.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
