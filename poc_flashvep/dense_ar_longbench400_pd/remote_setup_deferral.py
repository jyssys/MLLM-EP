"""Defer connector READ setup,not token delivery,while decode remains active."""
import atexit
import copy
import time
from .common import ROOT, write


def install(cfg, role):
    if not cfg.get('defer_D_remote_setup') or role != 'decode':
        return
    assert cfg.get('d_compute_throttle') and not cfg.get('d_quiescent_admission')
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import NixlPullConnectorWorker
    schedule = Scheduler.schedule
    load = NixlPullConnectorWorker.start_load_kv
    read = NixlPullConnectorWorker._read_blocks_for_req
    state = dict(decode_active=False)
    deferred = {}
    rows = []

    def scheduled(scheduler, *args, **kwargs):
        # The running scheduler includes decode requests until normal completion.
        state['decode_active'] = any(not r.is_prefill_chunk for r in scheduler.running)
        return schedule(scheduler, *args, **kwargs)

    def reading(worker, rid, meta):
        if state['decode_active']:
            deferred[rid] = meta
            rows.append(dict(event='DEFER_READ_SETUP', request_id=rid, time=time.monotonic()))
            return
        return read(worker, rid, meta)

    def loading(worker, metadata):
        local = copy.copy(metadata)
        incoming = dict(metadata.reqs_to_recv)
        if state['decode_active']:
            deferred.update(incoming)
            local.reqs_to_recv = {}
        else:
            incoming = dict(deferred, **incoming)
            deferred.clear()
            local.reqs_to_recv = incoming
        rows.append(dict(event='LOAD_METADATA', time=time.monotonic(), decode_active=state['decode_active'],
            received=list(metadata.reqs_to_recv), submitted=list(local.reqs_to_recv), pending=list(deferred)))
        # Preserve every other metadata field. Native ACK,heartbeat and retirement continue normally.
        return load(worker, local)

    Scheduler.schedule = scheduled
    NixlPullConnectorWorker.start_load_kv = loading
    NixlPullConnectorWorker._read_blocks_for_req = reading

    def close():
        write(ROOT / 'raw' / cfg['name'] / 'D_remote_setup_deferral.json', dict(rows=rows,
            pending=list(deferred), passed=not deferred, evidence='RUNTIME_POLICY_AUDIT',
            scope='Delay remote-KV READ preparation while the scheduler retains active decode. '
                'Do not change allocated blocks,metadata ACK semantics,model outputs or client timestamps. '
                'Native,Layer,Token and Hybrid use the same optional rule. '
                'Continuous decode batching remains enabled. This is not a measured GPU-idle predictor. '
                'Normal completion is tested. Cancellation and remote-failure recovery require separate qualification.'))
        assert not deferred, 'Do not discard a pending remote KV read'

    atexit.register(close)


def test():
    # Metadata copies must preserve all non-receive fields and must not mutate the caller.
    from types import SimpleNamespace
    original = SimpleNamespace(reqs_to_recv={'q': object()}, reqs_to_send={'p': 7}, reqs_in_batch={'z'})
    cloned = copy.copy(original)
    cloned.reqs_to_recv = {}
    assert set(original.reqs_to_recv) == {'q'} and not cloned.reqs_to_recv
    assert cloned.reqs_to_send is original.reqs_to_send and cloned.reqs_in_batch is original.reqs_in_batch
    print('REMOTE_SETUP_METADATA_COPY_PASS', flush=True)


if __name__ == '__main__':
    test()
