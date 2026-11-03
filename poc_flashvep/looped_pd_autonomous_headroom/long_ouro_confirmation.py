"""Check persistence over a larger cohort after three independent joint passes."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .resume_joint_review import analyze_capture
from .verify_long_probes import verify
from .independent_joint_repeats import captures


def main():
    activate()
    from ..looped_pd_atomic_prefill import evidence
    rows = json.loads((ROOT/'state/independent_joint_repeats_results.json').read_text())
    assert len(rows) == 3 and all(r.get('accepted_joint_screen') for r in rows)
    base = json.loads((ROOT/'configs/qualified_ouro_s1_baseline128.json').read_text())
    original = json.loads((ROOT/'data'/base['dataset']).read_text())
    write(ROOT/'data/persistence_ouro_L4096.json', [copy.deepcopy(original[i % 128]) for i in range(512)])
    base.update(name='persistence_ouro_baseline512', arrival_seed=2026111601, gpus=[2, 3], timeout=1200,
        data_scope=dict(original_pool_records=128, offered_count=512, explicit_cyclic_reuse=True,
            scope='Longer finite stress observation. Not 512 independent prompts or sustainable capacity.'))
    baseline = put(base, 512, base['arrival_rate'], 'persistence')
    write(ROOT/'state/long_ouro_confirmation.json', dict(stage='BASELINE', time=time.time()))
    assert execute(baseline) and evidence.full_work(baseline)['passed']
    probe = copy.deepcopy(base)
    probe.update(name='persistence_ouro_probe512', D_probe_tokens=16, D_probe_count=24, D_probe_interval_s=2.)
    clean = put(probe, 512, probe['arrival_rate'], 'persistence')
    assert execute(clean) and evidence.compare(baseline, clean)['passed']
    correctness = verify(clean)
    assert correctness.get('exact') and correctness.get('full_work')
    prof = copy.deepcopy(probe)
    prof.update(name='persistence_ouro_nsys512', matched_clean=clean, P_only_profiler=True,
        nsys=True, cuda_only_profile=False, bounded_native_capture_seconds=0., capture_role='prefill',
        capture_seconds=3., capture_prewarmed=True, capture_until_close=True, profile_stop_at_exit=True,
        nsys_cuda_flush_ms=300000, profile_finalization_timeout=300)
    name = put(prof, 512, prof['arrival_rate'], 'persistence')
    row = dict(model='ouro', baseline=baseline, clean=clean)
    if execute(name):
        row.update(analyze_capture(prof, clean))
        row['measured_captures'] = {n: captures(n) for n in (baseline, clean, name)}
        row['accepted_joint_screen'] = row['joint_screen_pass'] and not any(row['measured_captures'].values())
    else:
        row['status'] = 'PROFILE_FAILED'
    write(ROOT/'state/long_ouro_confirmation_results.json', [row])
    write(ROOT/'state/long_ouro_confirmation.json', dict(stage='COMPLETE_REVIEW_REQUIRED', time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/long_ouro_confirmation.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
