"""Measure control-plane cost without exposing a source checkpoint."""
from .common import *
from .optimization_cells import paired


def main():
    initialize()
    names = []
    for count in (8, 32):
        cfg = json.loads((ROOT/'configs'/f'candidate_fastloop_v2_state{count}.json').read_text())
        cfg.update(name=f'selective_nomigration_state{count}',
            require_actual_offload_qualification=False, force_destination_reject=True,
            checkpoint_selection_deadline_ms=0.,
            phase='SELECTIVE_NO_CHECKPOINT_NO_MIGRATION_EXACTNESS')
        if count == 32:
            cfg['required_qualification'] = 'selective_nomigration_state8'
        else:
            cfg.pop('required_qualification', None)
        save(cfg)
        names.append(cfg['name'])
    cfg = json.loads((ROOT/'configs/selection_fastloop_v2_seed2026110940.json').read_text())
    cfg = paired(cfg, 'selection_selective_nomigration_seed2026110940', 'selective_nomigration_state32')
    cfg.update(force_destination_reject=True, checkpoint_selection_deadline_ms=0.,
        phase='CONTROL_PLANE_WITHOUT_EXPOSED_CHECKPOINTS_OR_MIGRATION')
    save(cfg)
    names.append(cfg['name'])
    write(ROOT/'state/selective_nomigration_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
