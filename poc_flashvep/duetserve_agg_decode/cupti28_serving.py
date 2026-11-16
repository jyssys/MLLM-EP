"""Run one bounded A/activity/A trial with the qualified API28 collector."""
import fcntl
import json
import time

from .campaign import run_job
from .common import QUIET_LOCK, ROOT, preserve, resources, write
from .configuration import base, standard_capture_sizes


def main():
    gate = json.loads((ROOT / 'analysis/cupti28_v3_witness_gate.json').read_text())
    assert gate['passed'] and gate['kernels'] == 200 and not gate['counters']['invalid']
    build = json.loads((ROOT / 'tools/cupti28_v3_build.json').read_text())
    path = ROOT / 'tools/activity_build.json'
    if path.exists():
        assert json.loads(path.read_text()) == build
    else:
        write(path, build)
    write(ROOT / 'state/api28_capture_predeclaration.json', dict(time=time.time(),
        methods=['Native', 'DP2_B8192_S1024'], rate=6., count=128, cap=200,
        arrival_seed=2026111641, capture_offset_seconds=10., capture_seconds=20.,
        edge_trim_seconds=.1, maximum_relative_interference=.05,
        whole_run_metrics=['mean_TBT_ms', 'TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'output_tokens_s'],
        ROI_metrics=['parent_mean', 'parent_p99', 'worker_mean', 'worker_p99'],
        scope='New collector qualification only. The fixed Primary configuration, SLO, and confirmation plan remain unchanged. A collector witness does not establish clean serving measurements.'))
    try:
        for method in ('Native', 'DP2_B8192_S1024'):
            capture_failed = None
            for label in ('A1', 'CUPTI', 'A2'):
                name = f'api28_{method}_N128_O200_q6_s2026111641_{label}_v1'
                cfg = base(name, arm='PD_G1' if method == 'Native' else 'DP2',
                    budget=8192, slots=16 if method == 'Native' else 1024,
                    rate=6., count=128, seed=2026111641,
                    dataset='fresh_confirmation.json',
                    reference_name='fresh_reference_Native_N128_O200_repeat1_v1',
                    entry_module='poc_flashvep.duetserve_agg_decode.cupti28_entry',
                    phase='API28_BOUNDED_KERNEL_ONLY_INTERFERENCE_QUALIFICATION')
                if method != 'Native':
                    cfg['explicit_cudagraph_capture_sizes'] = standard_capture_sizes(1024)
                if label == 'CUPTI':
                    cfg.update(api28_bounded_activity=True, clean_timing=False,
                        activity_roles=['prefill', 'decode'] if method == 'Native' else ['replica0', 'replica1'],
                        activity_host_pool_bytes=512 * 1024**2, activity_memcpy=False,
                        activity_background_offset_s=10., activity_background_seconds=20.)
                try:
                    run_job(cfg)
                except Exception as error:
                    if label != 'CUPTI':
                        raise
                    capture_failed = repr(error)
                    write(ROOT / 'analysis' / ('api28_' + method + '_capture_failure.json'),
                        dict(name=name, error=capture_failed, evidence='FAILED',
                             scope='Finish the second marker-free control. Do not use partial latency or kernel records as clean measurements.'))
            if capture_failed is None:
                from .cupti28_analysis import qualify
                qualify(method)
            else:
                write(ROOT / 'analysis' / ('api28_' + method + '_serving_gate.json'),
                    dict(method=method, qualified=False, status='FAILED_CAPTURE_NOT_MEASURED',
                         role_windows={}, error=capture_failed, evidence='FAILED'))
        write(ROOT / 'state/api28_serving_complete.json', dict(time=time.time(), status='COMPLETE'))
    except BaseException as error:
        write(ROOT / 'state/api28_serving_failed.json', dict(time=time.time(), error=repr(error),
            scope='Preserve all failed activity and serving evidence. No clean physical-capacity claim follows.'))
        raise
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))
            preserve()


if __name__ == '__main__':
    main()
