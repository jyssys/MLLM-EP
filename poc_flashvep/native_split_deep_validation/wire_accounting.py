"""Count actual accepted native transfer regions without reading their contents."""
import json
import os
import time


def classify(local, bounds, size):
    result = dict(KV_bytes=0, hidden_bytes=0, descriptors=len(local))
    for pointer, length, device in local:
        matches = [kind for left, right, kind in bounds if left <= pointer and pointer + length <= right]
        if len(matches) != 1:
            raise RuntimeError('A submitted region does not belong to exactly one registered KV/hidden storage')
        result[matches[0]] += length
    if result['KV_bytes'] + result['hidden_bytes'] != size:
        raise RuntimeError('Submitted region lengths differ from the native ticket size')
    return result


def install(cfg):
    from vllm_rlt.pd.transport import NixlConnector
    from vllm_rlt.engine.stage_engine import StageEngine
    from .common import ROOT, write
    folder = ROOT / 'raw' / cfg['name']
    marker = folder / 'measurement_begin.json'
    initialize, submit, close = NixlConnector.__init__, NixlConnector.submit, StageEngine.close

    def initialized(owner, *args, **kwargs):
        result = initialize(owner, *args, **kwargs)
        owner._deep_wire_bounds = [(tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                                    'hidden_bytes' if index == 2 else 'KV_bytes')
                                   for index, tensor in enumerate(owner.storage)]
        owner._deep_wire_measured = False
        owner._deep_wire_counts = {phase: dict(KV_bytes=0, hidden_bytes=0, descriptors=0,
            accepted_submissions=0, notification_payload_bytes=0, observer_host_seconds=0.)
            for phase in ('startup', 'measured')}
        return result

    def submitted(owner, transfer_id, sequence, peer, local, remote, size):
        accepted = submit(owner, transfer_id, sequence, peer, local, remote, size)
        if not accepted:
            return accepted
        start = time.perf_counter()
        if not owner._deep_wire_measured:
            owner._deep_wire_measured = marker.exists()
        phase = 'measured' if owner._deep_wire_measured else 'startup'
        counter = owner._deep_wire_counts[phase]
        regions = classify(local, owner._deep_wire_bounds, size)
        for key, amount in regions.items():
            counter[key] += amount
        counter['accepted_submissions'] += 1
        counter['notification_payload_bytes'] += len(json.dumps([transfer_id, sequence]).encode())
        counter['observer_host_seconds'] += time.perf_counter() - start
        return accepted

    def closed(engine):
        transport = getattr(getattr(engine, 'connector', None), 'transport', None)
        if transport is not None and hasattr(transport, '_deep_wire_counts'):
            counts = transport._deep_wire_counts
            submitted_bytes = sum(row['KV_bytes'] + row['hidden_bytes'] for row in counts.values())
            complete = not transport.pending and submitted_bytes == transport.bytes_sent
            write(folder / f'wire_payload_{os.getpid()}.json', dict(role=engine.stage_config.role,
                counts=counts, actual_completed_payload_bytes=transport.bytes_sent,
                submitted_equals_completed_at_close=complete,
                scope='Actual successful native NIXL WRITE regions. KV and hidden are classified by registered storage. Notification payload bytes exclude PDEngine control/ACK messages and wire headers. Host observer time excludes the original submission.'))
        return close(engine)
    NixlConnector.__init__, NixlConnector.submit, StageEngine.close = initialized, submitted, closed
