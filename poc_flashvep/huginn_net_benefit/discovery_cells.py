"""Keep the primary anchor frozen; qualify an explicit short-Decode discovery."""
from .common import *
from .prepare import clone
from .optimization_cells import FAST, CALIBRATION, TARGET, paired


def main():
    initialize()
    names = []
    for count in (8, 32):
        for arm, source, options in (
                ('zero', 'control_huginn_zero', FAST),
                ('loop', 'secondary_huginn_loop', dict(**FAST,
                    track_source_device_events=True, net_benefit_extension=True,
                    selective_checkpoints=True, device_decode_costs=CALIBRATION,
                    native_ITL_target_ms=TARGET, feedback_ITL_limit_ms=TARGET,
                    checkpoint_selection_deadline_ms=350.))):
            name = f'discovery_output4_{arm}_state{count}'
            cfg = clone(f'{source}_state{count}', name,
                phase='OUTPUT4_DISCOVERY_NOT_PRIMARY_QUALIFICATION',
                cap=4, request_caps=[4]*count, workload='L2048_O4_DISCOVERY',
                qualification_output_prefix=True, **options)
            if arm == 'loop':
                cfg['require_actual_offload_qualification'] = True
            if count == 32:
                cfg['required_qualification'] = f'discovery_output4_{arm}_state8'
            save(cfg)
            names.append(name)
    for arm, source, options in (
            ('zero', 'control_huginn_zero_seed2026110931', FAST),
            ('loop', 'recovery_huginn_loop_seed2026110931', dict(**FAST,
                track_source_device_events=True, net_benefit_extension=True,
                selective_checkpoints=True, device_decode_costs=CALIBRATION,
                native_ITL_target_ms=TARGET, feedback_ITL_limit_ms=TARGET,
                checkpoint_selection_deadline_ms=350., completion_deadline_ms=350.))):
        name = f'discovery_output4_{arm}_seed2026110940'
        cfg = paired(clone(source, name, **options), name, f'discovery_output4_{arm}_state32')
        cfg.update(cap=4, request_caps=[4]*128, workload='L2048_O4_DISCOVERY',
            phase='OUTPUT4_DISCOVERY_NOT_PRIMARY_SELECTION')
        save(cfg)
        names.append(name)
    write(ROOT/'state/output4_discovery_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
