"""Retain an eight-second capture in a longer paired finite cohort."""
import copy
import random
import time
from .common import *
from .run import execute
from .recovery_qualification import save
from .recovery_analysis import analyze
from .confirm_recovery import operational


def main():
    initialize()
    original = json.loads((ROOT / 'configs/recovery_huginn_original_clean384.json').read_text())
    source = json.loads((ROOT / 'data' / original['dataset']).read_text())
    dataset = 'recovery_huginn_L2048_512_cyclic.json'
    data = [copy.deepcopy(source[index % 128]) for index in range(512)]
    target = ROOT / 'data' / dataset
    if target.exists():
        assert json.loads(target.read_text()) == data
    else:
        write(target, data)
    ledger = []
    for start in (1, 2, 3):
        if time.time() >= LONG_START_CUTOFF:
            break
        cfg = copy.deepcopy(original)
        cfg.update(name=f'recovery_huginn_extended{start}_clean512',
                   count=512, dataset=dataset, request_caps=[16] * 512,
                   prompt_ids=[row['prompt_id'] for row in data],
                   independent_start=start, phase='LONGER_PAIRED_CAPTURE_CONTROL',
                   arrival_seed=2026111500 + start,
                   data_scope=dict(original_pool_records=128, offered_count=512,
                                   explicit_cyclic_reuse=True))
        rng = random.Random(cfg['arrival_seed'])
        cfg['arrival_offsets'] = [0.]
        for _ in range(1, 512):
            cfg['arrival_offsets'].append(cfg['arrival_offsets'][-1] + rng.expovariate(cfg['arrival_rate']))
        save(cfg)
        if not execute(cfg['name']):
            ledger.append(dict(start=start, status='CLEAN_FAILED'))
            break
        clean = cfg['name']
        profile = copy.deepcopy(cfg)
        profile.update(name=f'recovery_huginn_extended{start}_profile512',
                       matched_clean=clean, nsys=True, P_only_profiler=True,
                       autonomous_capture_seconds=8., capture_arrival_fraction=.35,
                       capture_role='prefill', capture_seconds=0.,
                       capture_prewarmed=False, capture_until_close=False,
                       profile_stop_at_exit=False, nsys_cuda_flush_ms=300000,
                       profile_finalization_timeout=300)
        save(profile)
        if not execute(profile['name']):
            ledger.append(dict(start=start, status='PROFILE_FAILED'))
            break
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from ..looped_pd_headroom_recovery.profile_analysis import analyze as parse
        with quiet_analysis(profile['name'] + '_parse'):
            parsed = parse(profile['name'], clean)
        screen = analyze(profile['name'])
        row = dict(start=start, clean=clean, profile=profile['name'],
                   trace=parsed, screen=screen, operational_pass=operational(screen),
                   correctness='PENDING_FULL_PROBE_AND_STATE_GATE')
        ledger.append(row)
        write(ROOT / 'state/longer_capture_controls.json', dict(status='IN_PROGRESS', results=ledger))
        if not row['operational_pass']:
            break
    write(ROOT / 'state/longer_capture_controls.json', dict(status='REVIEW_REQUIRED', results=ledger,
        scope='Longer cohorts amortize collector boundary calls. This does not remove local capture perturbation or diagnose the old stall.'))


if __name__ == '__main__':
    main()
