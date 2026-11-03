"""Test whether mid-serving profiler finalization causes measured perturbation."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .analysis_lane import quiet_analysis
from .aligned_gates import one
from .ready_queue_audit import audit
from .verify_long_probes import verify


def main():
    activate()
    from ..looped_pd_headroom_recovery.cuda_only_profile import analyze
    source = 'bulk_joint_huginn_start1_probe16_128'
    cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
    cfg.update(name='huginn_native_tail_capture_probe16_128',
        phase='collector_finalization_control', matched_clean=source,
        bounded_native_capture_seconds=6., native_capture_stop_at_close=True,
        capture_arrival_fraction=.25, cuda_only_profile=False, P_only_profiler=False,
        nsys=False, capture_seconds=0., autonomous_capture_seconds=0.,
        postwarm_capture=False, capture_prewarmed=False, capture_until_close=False,
        capture_role='prefill', profile_finalization_timeout=300)
    name = put(cfg, 128, cfg['arrival_rate'], split='confirmation')
    row = dict(name=name, hypothesis='Finalize the native CUDA collector after measured responses. Longer collection can still perturb serving.',
        scope='Diagnostic collector control. Unchanged parity, perturbation, queue, service, and SLO acceptance criteria.')
    write(ROOT/'state/native_tail_control.json', dict(stage='QUEUED', **row))
    if time.time() >= LONG_START_CUTOFF:
        return
    row['complete'] = execute(name)
    if row['complete']:
        with quiet_analysis(name+'_profile'):
            row['profile'] = analyze(name, source)
            if row['profile'].get('status') == 'PARSED':
                row['gates'] = one(cfg)
                row['queue_audit'] = audit(cfg, row['gates'])
        row['correctness'] = verify(name)
    write(ROOT/'state/native_tail_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', result=row, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/native_tail_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
