"""Calibrate the confirmation estimator with matched N128 Native controls."""
import random
from .common import *
from .predictive32 import prototype


def main():
    assert not (ROOT/'heldout_plan.json').exists()
    jobs=[]
    for model in ('ouro','huginn'):
        for arm in ('native','zero'):
            cfg=prototype(model,arm,count=128)
            rng=random.Random(2026110910);offsets=[0.]
            for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(cfg['arrival_rate']))
            cfg.update(name=f'final_calibration_{model}_{arm}128',arrival_seed=2026110910,
                arrival_offsets=offsets,phase='MATCHED_POOL_N128_NATIVE_ESTIMATOR_CALIBRATION',
                timeout=300,startup_timeout=180,empty_admission_uses_current_cap=True,
                calibration_scope='Independent calibration arrival seed. Same 128 text IDs, HBM pool, and graph policy as confirmation. No compute migration is enabled.')
            if arm=='zero':cfg['comparison_reference']=f'final_calibration_{model}_native128'
            save(cfg);jobs.append(cfg['name'])
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text())
    assert not any(name in queue for name in jobs)
    write(path,queue+jobs)
    print(json.dumps(jobs))


if __name__=='__main__':main()
