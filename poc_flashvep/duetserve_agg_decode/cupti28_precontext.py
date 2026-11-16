"""Test the pre-context activation that passed the collector witness."""
import fcntl
import json
import time

from .campaign import run_job
from .common import QUIET_LOCK, ROOT, preserve, resources, write


def main():
    assert json.loads((ROOT / 'analysis/cupti28_v3_witness_gate.json').read_text())['passed']
    prefix = 'api28_Native_N128_O200_q6_s2026111641_'
    control = json.loads((ROOT / 'configs' / (prefix + 'A1_v1.json')).read_text())
    write(ROOT / 'state/api28_precontext_capture_predeclaration.json', dict(time=time.time(),
        source_failure=prefix + 'CUPTI_v1', control=prefix + 'A1_v1',
        hypothesis='The successful unit witness activates before the CUDA context. The failed serving trial first enabled activity on a background thread during warmup. Test pre-context activation without a concurrent enable/disable transition.',
        scope='One Native trial only. Full recording with a predeclared short analysis window. Keep the unchanged5% interference gate. No additional retry follows a failure.'))
    capture_failed = None
    try:
        cfg = dict(control, name=prefix + 'CUPTI_v2', clean_timing=False,
            api28_bounded_activity=True, api28_capture_mode='PRECONTEXT_FULL',
            activity_roles=['prefill', 'decode'], activity_host_pool_bytes=1024**3,
            activity_memcpy=False)
        try:
            run_job(cfg)
        except Exception as error:
            capture_failed = repr(error)
        run_job(dict(control, name=prefix + 'A2_v1'))
        if capture_failed is None:
            from .cupti28_analysis import qualify
            qualify('Native')
        else:
            write(ROOT / 'analysis/api28_Native_serving_gate.json', dict(method='Native', qualified=False,
                status='FAILED_CAPTURE_NOT_MEASURED', role_windows={}, error=capture_failed, evidence='FAILED'))
        write(ROOT / 'state/api28_serving_complete.json', dict(time=time.time(), status='COMPLETE',
            DP2='NOT_RUN_AFTER_NATIVE_INSTRUMENTATION_FAILURE',
            scope='No further collector retry. Native pre-context trial and the second marker-free control are accounted for.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))
            preserve()


if __name__ == '__main__':
    main()
