"""Separate progress cadence from the physical-byte/descriptor tradeoff."""
from .common import *
from .prepare import clone


def main():
    initialize()
    jobs = []
    for count in (8, 32):
        for arm, whole in (('packedburst', False), ('blockburst', True)):
            name = f'candidate_{arm}_state{count}'
            cfg = clone(f'control_huginn_zero_state{count}', name,
                phase='BOUNDED_NATIVE_WIRE_PROGRESS_QUALIFICATION',
                whole_owned_block_wire=whole, wire_progress_burst=True,
                wire_progress_steps=64, wire_progress_budget_ms=5.,
                native_transport_limits=dict(shutdown_timeout=300,
                    transfer_chunk_bytes=512*1024**2, max_inflight_bytes=16*1024**3,
                    max_transfer_descriptors=4096))
            if count == 32:
                cfg['required_qualification'] = f'candidate_{arm}_state8'
            save(cfg)
            jobs.append(name)
    write(ROOT/'state/wire_burst_qualification_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
