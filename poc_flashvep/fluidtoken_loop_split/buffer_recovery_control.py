"""Test the profiler-buffer allocation hypothesis, not model optimization."""
import copy
import time
from .common import *
from .run import execute
from .recovery_qualification import save


def main():
    initialize()
    write(ROOT / 'state/buffer_recovery_plan.json', dict(
        hypothesis='Dynamic CUPTI buffer allocation interacts with CUDA IPC progress under dense graph activity.',
        prediction='Preallocating 16 buffers permits a 32-second capture that failed with the default three buffers.',
        alternative='The fault is unrelated to buffer exhaustion. The same launch error or stall remains.',
        evidence_limit='A successful control does not identify the driver lock owner or prove a vendor bug.',
        reference='https://docs.nvidia.com/cupti/12.9/main/main.html',
        storage='16 times the queried device-buffer size. HBM effects are separate from model work.'))
    waiting = ROOT / 'state/adaptive_activity_recovery.json'
    while time.time() < LONG_START_CUTOFF:
        if waiting.exists() and json.loads(waiting.read_text())['status'] != 'SCREENING':
            break
        time.sleep(5)
    else:
        return
    test = json.loads((ROOT / 'configs/activity_only_collector_selftest.json').read_text())
    test.update(name='activity_buffer_stress400k_v2', collector_test_iterations=200000,
                cupti_preallocated_buffers=16, timeout=240, estimated_seconds=60)
    save(test)
    result = dict(selftest_name=test['name'], selftest_complete=execute(test['name']))
    write(ROOT / 'state/buffer_recovery_result_v2.json', result)
    if not result['selftest_complete']:
        return
    measured = json.loads((ROOT / 'raw' / test['name'] / 'activity_selftest.json').read_text())
    result['selftest'] = measured
    if not measured['passed']:
        return
    cfg = copy.deepcopy(json.loads((ROOT / 'configs/recovery_huginn_activity_window32_profile512.json').read_text()))
    cfg.update(name='recovery_huginn_buffer16_window32_diagnostic512',
               cupti_preallocated_buffers=16, allow_long_activity_diagnostic=True,
               native_stack_diagnostic=True, phase='BUFFER_ALLOCATION_DIAGNOSTIC')
    save(cfg)
    result['name'] = cfg['name']
    result['complete'] = execute(cfg['name'])
    if result['complete']:
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        from .activity_analysis import analyze
        from .recovery_analysis import analyze as screen
        from .confirm_recovery import operational
        with quiet_analysis(cfg['name'] + '_parse'):
            result['profile'] = analyze(cfg['name'], cfg['matched_clean'])
        result['screen'] = screen(cfg['name'])
        result['operational_pass'] = operational(result['screen'])
    write(ROOT / 'state/buffer_recovery_result_v2.json', result)


if __name__ == '__main__':
    main()
