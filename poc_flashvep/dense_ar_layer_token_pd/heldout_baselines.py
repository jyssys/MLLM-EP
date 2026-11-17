"""Freeze method-independent baselines before candidate confirmation."""
import fcntl
import json
import time
from .common import ROOT, SLO_PRIOR, QUIET_LOCK, resources, save, write, digest
from .run import execute
from .axis_configuration import base
from .analysis import metrics
from .fresh_reference import REFERENCE

SEEDS = (2026111740, 2026111741, 2026111742)
RATES = (2., 4.8, 6.4, 10.)


def configuration(method, cap, qps, seed):
    name = f'heldout_{method}_N128_O{cap}_q{qps:g}_s{seed}_v1'
    changes = dict(dataset='heldout128_8000.json', reference_name=REFERENCE,
        phase='FROZEN_HELDOUT_BASELINE', confirmation_method=method)
    if method == 'Native':
        return base(name, method='Native', count=128, cap=cap, rate=qps, seed=seed,
            D_admission_batch=16, D_combine_ready_macros=True, D_wait_for_source_macro=False,
            **changes)
    from ..duetserve_agg_decode import configuration as inherited
    inherited.ROOT, inherited.PRIOR = ROOT, SLO_PRIOR
    return inherited.base(name, budget=8192, slots=1024, count=128, cap=cap,
        rate=qps, seed=seed, entry_module='poc_flashvep.dense_ar_layer_token_pd.serving',
        explicit_cudagraph_capture_sizes=inherited.standard_capture_sizes(1024), **changes)


def main():
    assert json.loads((ROOT / 'analysis/unseen_reference_gate.json').read_text())['passed']
    path = ROOT / 'state/heldout_baseline_freeze.json'
    contract = dict(dataset_SHA256=digest(ROOT / 'data/heldout128_8000.json'),
        SLO_SHA256=digest(ROOT / 'configs/frozen_SLO_regimes.json'), seeds=list(SEEDS),
        QPS=list(RATES), caps=[16,64], Native=dict(P_slots=4,D_slots=16,D_max_cohort=16),
        Colocated=dict(B=8192,S=1024,graph='FULL',routing='ROUND_ROBIN'),
        scope='User-selected primary continuous DP2, not a global decode-priority optimum. Baseline results cannot calibrate or tune the new candidate.')
    if path.exists():
        assert json.loads(path.read_text())['contract'] == contract
    else:
        write(path, dict(time=time.time(), contract=contract))
    values = []
    try:
        # The overload and low-load comparisons are explicit priorities. No
        # missing endpoint is inferred from an intermediate finite cohort.
        for cap in (16,64):
            for qps in (6.4,2.,10.,4.8):
                for seed in SEEDS:
                    methods = ('Native','Colocated') if seed%2==0 else ('Colocated','Native')
                    for method in methods:
                        cfg = configuration(method, cap, qps, seed)
                        if not execute(save(cfg)):
                            continue
                        result = metrics(cfg['name'])
                        assert result['output_gate']['passed']
                        values.append(result)
                        write(ROOT / 'analysis/heldout_baselines.json', values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
