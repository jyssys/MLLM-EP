"""Test whether collection during graph warm-up causes the Huginn failure."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .aligned_gates import one


def main():
    activate()
    from ..looped_pd_headroom_recovery.profile_analysis import analyze
    source = 'graph_pressure_huginn_factor110'
    cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
    cfg.update(name=source+'_nsys_postwarm_stats_handshake', phase='profiling_cause_control',
               P_only_profiler=True, nsys=True, capture_role='prefill', capture_seconds=0.,
               capture_prewarmed=False, capture_until_close=True, profile_stop_at_exit=True,
               postwarm_capture=True, nsys_cuda_flush_ms=300000, matched_clean=source,
               profile_finalization_timeout=300)
    cfg['native_transport_limits'] = dict(cfg.get('native_transport_limits', {}), shutdown_timeout=300)
    name = put(cfg, 24, cfg['arrival_rate'], split='confirmation')
    write(ROOT/'state/postwarm_control.json', dict(stage='RUNNING', name=name, time=time.time()))
    result = dict(name=name, complete=execute(name),
        hypothesis='Collection during native graph warm-up contributes to stalls or measured overhead.',
        alternatives=['NIXL progress overhead during collection', 'Per-node graph tracing cost during steady replay'])
    if result['complete']:
        result['profile'] = analyze(name, source)
        result['gates'] = one(cfg)
    write(ROOT/'state/postwarm_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', result=result, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/postwarm_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
