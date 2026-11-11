"""Prepare state gates before the first matched profiler controls."""
import json,random,shutil
from .common import *
from .gates import model_contract


def main():
    initialize()
    for source in (WIDTH/'data').glob('*.json'):
        destination=ROOT/'data'/source.name
        if not destination.exists():shutil.copyfile(source,destination)
    for relative in ('tools/lt_build.json','configs/frozen_SLO.json'):
        destination=ROOT/relative
        if not destination.exists():shutil.copyfile(WIDTH/relative,destination)
    state=[];controls=[]
    for model,cap,rate in (('ouro',32,6.356),('huginn',16,1.971)):
        for count in (8,32):
            cfg=json.loads((WIDTH/'configs'/f'kernel_v2_{model}_zero_state{count}.json').read_text())
            name=f'native_{model}_early_profile_state{count}'
            cfg.update(name=name,phase='EARLY_PREALLOCATED_COLLECTOR_STATE_QUALIFICATION',
                required_qualification=f'native_{model}_early_profile_state8' if count==32 else None,
                discovery_activity=count==8,activity_memcpy=False,activity_window_seconds=5.,
                observe_P_submission_readiness=True,timeout=600,startup_timeout=240)
            assert model_contract(model,cfg['length'],cap)
            save(cfg);state.append(name)
        origin=json.loads((WIDTH/'configs'/f'serve_v2_{model}_zero_O{cap}_N64_a2026111110.json').read_text())
        seed=2026111210;rng=random.Random(seed);offsets=[0.]
        for _ in range(63):offsets.append(offsets[-1]+rng.expovariate(rate))
        for mode in ('clean','capture','repeat'):
            cfg=dict(origin);name=f'anchor_{model}_L{cfg["length"]}_O{cap}_{mode}_N64_a{seed}'
            cfg.update(name=name,phase='MATCHED_EARLY_COLLECTOR_NATIVE_A_B_A',arrival_seed=seed,arrival_offsets=offsets,
                required_qualification=f'native_{model}_early_profile_state32',
                discovery_activity=mode=='capture',activity_memcpy=False,activity_window_seconds=5.,
                activity_window_fraction=.35,observe_P_submission_readiness=True,
                timeout=600,startup_timeout=240)
            save(cfg);controls.append(name)
    write(ROOT/'state/initial_qualification_plan.json',dict(jobs=state,
        scope='N8 includes early activity collection and complete state probes. N32 uses the same qualified numerical path without intrusive profiling. Neither is serving performance evidence.'))
    write(ROOT/'state/initial_anchor_controls.json',dict(jobs=controls,
        scope='One marker-free A/B/A on a fixed trace per model. These are profiler pilots, not three independent confirmation starts.'))
    print(json.dumps(dict(state=state,controls=controls)))


if __name__=='__main__':main()
