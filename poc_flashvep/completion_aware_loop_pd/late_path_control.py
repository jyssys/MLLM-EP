"""Confirm the combined Native-path effect with an interleaved A/B/A control."""
import copy
from .common import *


def main():
    initialize();jobs=[]
    for suffix,arm in (('a','native'),('b','zero'),('a2','native')):
        source=f'recovery_huginn_native_seed2026110931' if arm=='native' else 'control_huginn_zero_seed2026110931'
        cfg=copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
        cfg.update(name=f'late_huginn_{arm}_{suffix}',phase='PREDECLARED_LATE_ABA_RUNTIME_PATH_CONTROL',
            primary_safety_arm=f'late_huginn_{arm}',comparison_reference='late_huginn_native_a',
            late_control_arm=arm)
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/late_ABA_plan.json',dict(time=time.time(),jobs=jobs,
        order=['Original Native A','Native-path zero-offload B','Original Native A2'],
        arrival_seed=2026110931,
        scope='Confirm a combined Runtime-path difference. Do not identify a unique component from this bundled intervention. All outcomes remain reported.'))
    closed=ROOT/'state/CAMPAIGN_CLOSED.json'
    if closed.exists():
        archive=ROOT/'state/scientific_closure_before_ABA.json'
        assert not archive.exists();closed.rename(archive)
    print(' '.join(jobs))


if __name__=='__main__':
    main()
