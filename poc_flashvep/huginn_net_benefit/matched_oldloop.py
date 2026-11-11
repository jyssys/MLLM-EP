"""Compare the previous checkpoint controller on the common stronger Runtime."""
from .common import *
from .prepare import clone
from .matched_wire_cells import fields
from .optimization_cells import paired


def main():
    initialize()
    options = fields('loop')
    options.update(net_benefit_extension=False, selective_checkpoints=False,
        source_monitor_boundaries=True)
    jobs = []
    for count in (8, 32):
        name = f'matched_oldloop_state{count}'
        cfg = clone(f'secondary_huginn_loop_state{count}', name,
            phase='PREVIOUS_CHECKPOINT_CONTROLLER_COMMON_RUNTIME_QUALIFICATION', **options)
        cfg['require_actual_offload_qualification'] = True
        if count == 32:
            cfg['required_qualification'] = 'matched_oldloop_state8'
        save(cfg)
        jobs.append(name)
    name = 'selection_matched_oldloop_seed2026110940'
    cfg = paired(clone('recovery_huginn_loop_seed2026110931', name, **options),
        name, 'matched_oldloop_state32')
    cfg.update(phase='PREVIOUS_CHECKPOINT_CONTROLLER_COMMON_RUNTIME_SELECTION',
        completion_deadline_ms=350.)
    save(cfg)
    jobs.append(name)
    write(ROOT/'state/matched_old_controller_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
