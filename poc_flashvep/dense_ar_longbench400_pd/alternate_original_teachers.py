"""Capture the second observed original compiled numerical path as a teacher."""
import json
import time
from .common import ROOT, write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .remote_setup_qualification import native_state
from .qualification_gate import compare


def main():
    capsule, sha = freeze('alternate_original_compiled_teachers_v1_20261121')
    common = dict(pair=(2, 3), correctness=True, serial=True, output=16, fixed=True,
        qps=2.8, seed=2026112201, execution_family='INSTALLED_DEFAULT_COMPILE',
        state_capture=True, axis_state_capture=False, gpu_state_digest=True,
        numerical_probe=True, numerical_probe_sync=True,
        kernel_activity=True, collector_namespace='kernel_only_collector_v3',
        activity_copies=False, operator_correlation=False, runtime_api_activity=False,
        kernel_activity_seconds=7200, source_capsule=capsule, source_capsule_SHA256=sha,
        wall_timeout=4500, serving_timeout=4000)
    jobs = [create('Q31_OriginalCompiledNative_AlternatePath_N8_v1', 'Native', 'qualification8.json',
        controlled=True, continuous_source_credit_limit=0, d_compute_throttle=True,
        defer_D_remote_setup=False, kernel_activity_roles=['prefill'], **common),
        create('Q31_OriginalFull36_TokenBoundary4_AlternatePath_N8_v1', 'Native', 'qualification8.json',
        reference_single=True, token_teacher_cut=4, kv_layout='HND',
        kernel_activity_roles=['single'], **common)]
    write(ROOT / 'configs/alternate_original_compiled_teachers_v1.json', dict(jobs=jobs,
        evidence='FROZEN_NUMERICAL_PATH_DIAGNOSTIC_DESIGN',
        scope='Q30 original Native without the READ intervention produced exactly the same P and D state as '
            'the previously mismatching clean Native path. Capture all8 original-model requests in that context. '
            'The token teacher uses the unmodified original full36 model with only a legal prefix chunk boundary. '
            'These profiled teachers are correctness diagnostics,not latency references. '
            'Keep every historical strict failure. Do not replace model tensors or generated tokens. '
            'A matching original path does not guarantee cross-start or cross-batch invariance.'))
    results = []
    for i, name in enumerate(jobs):
        assert execute(name), name
        if i == 0:
            try:
                result = native_state('Q28_RemoteSetup_Native_N8_v1', name)
            except AssertionError as error:
                result = dict(passed=False, error=repr(error))
        else:
            result = compare(name, 'Q28_RemoteSetup_StockToken_N8_v1')
        results.append(dict(teacher=name, actual=('Q28_RemoteSetup_Native_N8_v1' if i == 0
            else 'Q28_RemoteSetup_StockToken_N8_v1'), comparison=result))
        write(ROOT / 'analysis/alternate_original_compiled_path_teacher_results_v1.json',
              dict(time=time.time(), results=results, evidence='CORRECTNESS_ONLY',
                  scope='Strict state/output comparisons retain their result. This file does not automatically authorize performance or erase the earlier failed comparison.'))
        print('ALTERNATE_ORIGINAL_TEACHER', name, result, flush=True)


if __name__ == '__main__':
    main()
