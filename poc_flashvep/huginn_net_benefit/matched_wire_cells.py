"""Remove the discovered Native wire-credit and draining-policy confound."""
from .common import *
from .prepare import clone
from .optimization_cells import FAST, CALIBRATION, TARGET, paired

NATIVE_WIRE = dict(max_inflight_bytes=8*1024**3, max_draining_requests=0, shutdown_timeout=300)


def fields(arm):
    value = dict(**FAST, native_transport_limits=NATIVE_WIRE,
        track_source_device_events=True, device_decode_costs=CALIBRATION,
        native_ITL_target_ms=TARGET, feedback_ITL_limit_ms=TARGET)
    if arm != 'token':
        value.update(net_benefit_extension=True, selective_checkpoints=arm != 'checkpoint',
            boundary_checkpoints=[16, 21, 24, 28, 30, 31],
            checkpoint_selection_deadline_ms=0. if arm == 'nomigration' else 350.)
    if arm in ('nomigration', 'checkpoint'):
        value['force_destination_reject'] = True
    return value


def main():
    initialize()
    jobs = []
    for arm in ('nomigration', 'checkpoint', 'loop', 'token'):
        for count in (8, 32):
            source = 'token' if arm == 'token' else 'loop'
            name = f'matchedwire_{arm}_state{count}'
            cfg = clone(f'secondary_huginn_{source}_state{count}', name,
                phase='MATCHED_8GIB_ZERO_DRAINING_EXACTNESS', **fields(arm))
            cfg['require_actual_offload_qualification'] = arm in ('loop', 'token')
            if arm in ('loop', 'token'):
                cfg.update(arrival_offsets=[10.*(i//2) for i in range(count)],
                    arrival_pattern='CONTROLLED_TWO_REQUEST_BURSTS_10S',
                    qualification_scope='Online budget decisions in a controlled cohort. Cuts are not assigned in advance. This is not a simultaneous-submission N32 qualification.')
            if count == 32:
                cfg['required_qualification'] = f'matchedwire_{arm}_state8'
            save(cfg)
            jobs.append(name)
        source = f'recovery_huginn_{"token" if arm == "token" else "loop"}_seed2026110931'
        name = f'selection_matchedwire_{arm}_seed2026110940'
        cfg = paired(clone(source, name, **fields(arm)), name, f'matchedwire_{arm}_state32')
        cfg['phase'] = 'WIRE_AND_DRAINING_MATCHED_POLICY_SELECTION'
        if arm in ('loop', 'token'):
            cfg['completion_deadline_ms'] = 350.
        save(cfg)
        jobs.append(name)
    write(ROOT/'state/matched_wire_policy_jobs.json', jobs)
    write(ROOT/'state/wire_configuration_confound.json', dict(
        Native_path_Zero=dict(max_inflight_bytes=8*1024**3, max_draining_requests=0),
        inherited_recovery_Loop_Token=dict(max_inflight_bytes=256*1024**2, max_draining_requests=8),
        finding='The recovered policy configuration inherited PD defaults. The stronger Zero explicitly overrode them. Earlier bundled comparisons cannot isolate checkpoint or policy overhead.',
        action='Retain all observations. Match resolved Native transport limits before causal attribution or final confirmation.'))
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
