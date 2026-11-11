"""Prepare isolated transport ablations and the matched optimized-runtime arms."""
import random
from .common import *
from .prepare import clone


FAST = dict(decode_loop_grouping=32, fixed_depth_submit=True)
# Selection data only. This is a cycle-enclosure allocation, not kernel WCET.
CALIBRATION = dict(loop_p99_ms=(93.10616207122803-8.068305926322935)/32,
    boundary_guard_ms=8.068305926322935,
    scope='Allocated from the maximum p99 of the two improved-Decode selection controls. Not measured exclusive per-loop GPU time.')
TARGET = 1.05*93.10616207122803


def paired(cfg, name, qualification):
    rng = random.Random(2026110940)
    offsets = [0.]
    for _ in range(127):
        offsets.append(offsets[-1]+rng.expovariate(1.971))
    cfg.update(name=name, phase='OPTIMIZATION_SELECTION_NOT_CONFIRMATION',
        state_probe=False, verify_transfer=False, count=128, request_caps=[16]*128,
        arrival_seed=2026110940, arrival_offsets=offsets,
        required_qualification=qualification, timeout=600, clean_confirmation=True)
    return cfg


def main():
    initialize()
    jobs = []
    for arm, options, qualification in (
            ('blockwire', dict(whole_owned_block_wire=True,
                native_transport_limits=dict(shutdown_timeout=300,
                    transfer_chunk_bytes=512*1024**2, max_inflight_bytes=2*1024**3,
                    max_transfer_descriptors=4096)), 'candidate_blockwire_state32'),
            ('earlytap', dict(native_loop_event_transfer=True, native_event_tap_depth=16), 'candidate_native_earlytap_state32'),
            ('integers', dict(exact_integer_validation=True), 'candidate_integer_validation_state32')):
        cfg = clone('control_huginn_zero_seed2026110931', f'selection_{arm}_seed2026110940', **options)
        if arm == 'earlytap':
            qualified = json.loads((ROOT/'configs/candidate_native_earlytap_state32.json').read_text())
            cfg.update({k:v for k,v in qualified.items() if k.startswith('native_loop_') or k.startswith('native_event_')})
        save(paired(cfg, cfg['name'], qualification))
        jobs.append(cfg['name'])
    for count in (8, 32):
        for arm, source, options in (
                ('sourceevents', 'loop', dict(track_source_device_events=True,
                    net_benefit_extension=True, selective_checkpoints=True)),
                ('fastloop', 'loop', dict(**FAST, track_source_device_events=True,
                    net_benefit_extension=True, selective_checkpoints=True,
                    device_decode_costs=CALIBRATION, native_ITL_target_ms=TARGET,
                    feedback_ITL_limit_ms=TARGET, checkpoint_selection_deadline_ms=350.)),
                ('fasttoken', 'token', dict(**FAST, track_source_device_events=True,
                    device_decode_costs=CALIBRATION, native_ITL_target_ms=TARGET,
                    feedback_ITL_limit_ms=TARGET))):
            name = f'candidate_{arm}_state{count}'
            cfg = clone(f'secondary_huginn_{source}_state{count}', name,
                phase='OPTIMIZED_RUNTIME_EXACT_STATE_QUALIFICATION', **options)
            cfg['require_actual_offload_qualification'] = True
            if count == 32:
                cfg['required_qualification'] = f'candidate_{arm}_state8'
            save(cfg)
            jobs.append(name)
    for arm, source, options, qualification in (
            ('fastzero', 'control_huginn_zero_seed2026110931', FAST, 'candidate_fixedsubmit_group32_state32'),
            ('fastloop', 'recovery_huginn_loop_seed2026110931', dict(**FAST,
                track_source_device_events=True, net_benefit_extension=True,
                selective_checkpoints=True, device_decode_costs=CALIBRATION,
                native_ITL_target_ms=TARGET, feedback_ITL_limit_ms=TARGET,
                checkpoint_selection_deadline_ms=350., completion_deadline_ms=350.), 'candidate_fastloop_state32'),
            ('fasttoken', 'recovery_huginn_token_seed2026110931', dict(**FAST,
                track_source_device_events=True, device_decode_costs=CALIBRATION,
                native_ITL_target_ms=TARGET, feedback_ITL_limit_ms=TARGET,
                completion_deadline_ms=350.), 'candidate_fasttoken_state32')):
        name = f'selection_{arm}_seed2026110940'
        cfg = clone(source, name, **options)
        save(paired(cfg, name, qualification))
        jobs.append(name)
    write(ROOT/'state/optimization_cells.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
