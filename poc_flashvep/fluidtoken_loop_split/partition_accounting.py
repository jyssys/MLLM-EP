"""Record actual accepted boundaries separately from initial policy choices."""
import os
import time


def install(cfg):
    if not cfg.get('split_evaluation') or cfg.get('arm') == 'PD_G1':
        return
    from .common import ROOT, prior
    from ..adaptive_loop_token_pd_2gpu.connector import PartitionConnector
    accept = PartitionConnector.accept_cut

    def accepted(connector, request, hidden, *, cached_tokens=0):
        result = accept(connector, request, hidden, cached_tokens=cached_tokens)
        if result:
            transfer = connector.transfers[request.request_id]
            length = len(request.prompt_token_ids)
            depth = connector.cache.max_loops
            cut, prefix = transfer.cut, transfer.prefix
            rows = length * (depth - cut) if cfg['axis'] == 'loop' else (length - prefix) * depth
            prior.append(ROOT / 'raw' / cfg['name'] / f'accepted_partitions_{os.getpid()}.jsonl', dict(
                time=time.monotonic(), request_id=request.request_id,
                seed=request.sampling_params.seed, axis=cfg['axis'],
                prompt_tokens=length, full_depth=depth, accepted_cut=cut,
                accepted_prefix_tokens=prefix, partial=rows > 0,
                assigned_D_recurrent_token_rows=rows,
                evidence='MEASURED_ACCEPTED_BOUNDARY_WITH_STRUCTURAL_ROW_ACCOUNTING',
                scope='Acceptance is not completion. Output and full-work gates must separately pass. Rows are not FLOPs or kernel counts.'))
        return result

    PartitionConnector.accept_cut = accepted
