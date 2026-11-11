"""Qualify descriptor geometry before changing the strongest baseline."""
from .common import *
from .prepare import clone


def main():
    initialize()
    jobs = []
    for count in (8, 32):
        name = f'candidate_blockwire_state{count}'
        cfg = clone(f'control_huginn_zero_state{count}', name,
            phase='EXACT_OWNED_BLOCK_WIRE_QUALIFICATION', whole_owned_block_wire=True,
            native_transport_limits=dict(shutdown_timeout=300,
                transfer_chunk_bytes=512*1024**2, max_inflight_bytes=2*1024**3,
                max_transfer_descriptors=4096))
        if count == 32:
            cfg['required_qualification'] = 'candidate_blockwire_state8'
        save(cfg)
        jobs.append(name)
    write(ROOT / 'state/block_wire_jobs.json', jobs)
    print(' '.join(jobs))


if __name__ == '__main__':
    main()
