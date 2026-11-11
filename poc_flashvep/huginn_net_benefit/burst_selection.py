"""Select clean handoff candidates only after both exactness gates pass."""
from .common import *
from .prepare import clone
from .optimization_cells import paired


def main():
    initialize()
    names = []
    for arm, whole in (('packedburst', False), ('blockburst', True)):
        name = f'selection_{arm}_seed2026110940'
        cfg = clone('control_huginn_zero_seed2026110931', name,
            whole_owned_block_wire=whole, wire_progress_burst=True,
            wire_progress_steps=64, wire_progress_budget_ms=5.,
            native_transport_limits=dict(shutdown_timeout=300,
                transfer_chunk_bytes=512*1024**2, max_inflight_bytes=16*1024**3,
                max_transfer_descriptors=4096))
        save(paired(cfg, name, f'candidate_{arm}_state32'))
        names.append(name)
    write(ROOT/'state/wire_burst_selection_jobs.json', names)
    print(' '.join(names))


if __name__ == '__main__':
    main()
