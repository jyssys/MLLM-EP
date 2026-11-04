"""Measure a no-probe control with the recovered execution settings."""
import copy
import random
from .common import *
from .run import execute
from .recovery_qualification import save


def run(base):
    from .static_screen import metrics
    from ..looped_pd_atomic_prefill import evidence
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    cfg = copy.deepcopy(base)
    cfg.update(name='recovery_huginn_final_matched_no_probe512',
        kernel_activity_seconds=0., arrival_seed=2026111601,
        phase='MATCHED_CLEAN_NO_PROBE_CONTROL')
    for key in ('D_probe_tokens', 'D_probe_count', 'D_probe_interval_s', 'matched_clean'):
        cfg.pop(key, None)
    rng = random.Random(cfg['arrival_seed'])
    cfg['arrival_offsets'] = [0.]
    for _ in range(cfg['count'] - 1):
        cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
    save(cfg)
    complete = execute(cfg['name'])
    result = dict(name=cfg['name'], complete=complete)
    if complete:
        with quiet_analysis(cfg['name'] + '_analysis'):
            result.update(metrics=metrics(cfg['name']),
                output=evidence.compare('recovery_huginn_qualified_admission_no_probe512', cfg['name']),
                work=evidence.full_work(cfg['name']))
    result['correct'] = bool(complete and result['output']['passed'] and result['work']['passed'])
    write(ROOT / 'state/final_matched_no_probe.json', result)
    return result['correct']
