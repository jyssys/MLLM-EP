"""Diagnostic-only event observations. Never enabled in clean serving jobs."""
import os
from pathlib import Path
import time


def install():
    if os.environ.get('B_EVENT_TRACE') != '1':
        return
    from vllm_rlt.pd.connector import PDStageConnector
    from vllm_rlt.pd.transport import NixlConnector
    from ..looped_pd_abc.telemetry import append
    if getattr(NixlConnector, '_B_event_trace', False):
        return
    NixlConnector._B_event_trace = True

    def record(kind, **value):
        path = Path(os.environ['ABC_RUN_DIR']) / f'transfer_events_{os.getpid()}.jsonl'
        append(path, dict(kind=kind, time=time.monotonic(), pid=os.getpid(), **value,
                          evidence='MEASURED_DIAGNOSTIC_HOST_OBSERVATION_NOT_GPU_DMA_TIME'))

    progress = PDStageConnector.progress
    def traced_progress(self):
        if not hasattr(self, '_B_seen_ready'):
            self._B_seen_ready = set()
        for transfer in list(self.transfers.values()):
            for event, sequence, local, remote, size in list(transfer.chunks):
                key = transfer.tid, sequence
                if key not in self._B_seen_ready and (event is None or event.query()):
                    self._B_seen_ready.add(key)
                    record('ready_first_observed', tid=transfer.tid, sequence=sequence, bytes=size,
                           scope='First diagnostic poll that observes the producer CUDA event complete. Not exact kernel completion time.')
        return progress(self)
    PDStageConnector.progress = traced_progress

    def observe_prefill(original):
        def wrapped(self, request, **kwargs):
            transfer = self.transfers.get(request.request_id)
            sequence_start = transfer.next_sequence if transfer else None
            result = original(self, request, **kwargs)
            transfer = self.transfers.get(request.request_id)
            if transfer:
                progress = kwargs['progress']
                record('prefill_quantum_callback', tid=transfer.tid, request_id=request.request_id,
                       phase=str(kwargs['phase']), chunk_start=kwargs['start'],
                       chunk_complete=bool(progress.chunk_complete), prompt_complete=bool(progress.prompt_complete),
                       sequence_start=sequence_start, sequence_end=transfer.next_sequence,
                       prefilled_tokens=request.num_prefilled_tokens,
                       depth_ranges=request.prefill_depth_ranges,
                       scope='Host callback and queued sequence range. Not an exact GPU completion timestamp.')
            return result
        return wrapped
    PDStageConnector.prefill_submitted = observe_prefill(PDStageConnector.prefill_submitted)
    from ..looped_pd_abc.frontier import FrontierConnector
    FrontierConnector.prefill_submitted = observe_prefill(FrontierConnector.prefill_submitted)

    submit = NixlConnector.submit
    def traced_submit(self, tid, sequence, peer, local, remote, size):
        begin = time.monotonic()
        result = submit(self, tid, sequence, peer, local, remote, size)
        elapsed_us = 1e6 * (time.monotonic() - begin)
        if not hasattr(self, '_B_submit_cost'):
            self._B_submit_cost = dict(attempts=0, accepted=0, retries=0, accepted_us=0., retry_us=0.)
        cost = self._B_submit_cost
        cost['attempts'] += 1
        cost['accepted' if result else 'retries'] += 1
        cost['accepted_us' if result else 'retry_us'] += elapsed_us
        if result:
            record('submitted', tid=tid, sequence=sequence, bytes=size,
                   descriptor_count=len(local), metadata_and_issue_us=elapsed_us)
        return result
    NixlConnector.submit = traced_submit

    close = NixlConnector.close
    def traced_close(self):
        if hasattr(self, '_B_submit_cost'):
            record('transport_submission_summary', **self._B_submit_cost,
                   scope='Sum of host API call durations, including rejected attempts. Can overlap GPU work. Not additive E2E cost.')
        return close(self)
    NixlConnector.close = traced_close

    poll = NixlConnector.poll
    def traced_poll(self):
        finished, notifications = poll(self)
        for transfer in finished:
            record('completion_observed', tid=transfer.transfer_id, sequence=transfer.sequence,
                   bytes=transfer.size, actual_submit=transfer.submitted)
        return finished, notifications
    NixlConnector.poll = traced_poll
