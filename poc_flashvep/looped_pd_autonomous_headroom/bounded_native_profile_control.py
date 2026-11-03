"""Test the existing native CUDA-only collector with a bounded window."""
import copy
from .common import *
from .prepare import put
from .run import execute
from .aligned_gates import one


def main():
    activate()
    from ..looped_pd_headroom_recovery.cuda_only_profile import analyze
    source = 'graph_pressure_huginn_factor110'
    cfg = copy.deepcopy(json.loads((ROOT/'configs'/(source+'.json')).read_text()))
    cfg.update(name='huginn_native_cuda_bounded_graph_control24',
               phase='profiling_cause_control', cuda_only_profile=True,
               capture_roles=['prefill'], capture_role='prefill',
               native_capture_seconds=3., P_only_profiler=False,
               capture_seconds=0., nsys=False, capture_prewarmed=False,
               matched_clean=source, profile_finalization_timeout=300)
    name = put(cfg, 24, cfg['arrival_rate'], split='confirmation')
    write(ROOT/'state/bounded_native_profile_control.json', dict(stage='RUNNING', name=name, time=time.time()))
    row = dict(name=name, complete=execute(name),
        hypothesis='Disable native CUDA activity collection during warm-up and limit collection to three seconds.',
        scope='The native collector and original model execution are reused. Earlier unbounded collector failures remain separate.')
    if row['complete']:
        row['profile'] = analyze(name, source)
        if row['profile'].get('status') == 'PARSED':
            row['gates'] = one(cfg)
    write(ROOT/'state/bounded_native_profile_control.json', dict(stage='COMPLETE_REVIEW_REQUIRED', result=row, time=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'state/bounded_native_profile_control.json', dict(stage='FAILED_REVIEW_REQUIRED', error=repr(error), time=time.time()))
        raise
