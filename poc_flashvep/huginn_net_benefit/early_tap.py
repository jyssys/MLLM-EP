"""Reuse Native per-loop graph events without a source scheduling checkpoint."""
import os
from types import MethodType
from .common import ROOT, write
from .early_kv import publish
from .progressive import delta_layout


def install(cfg):
    assert cfg.get('intervention_method', 'native') == 'native' or cfg.get('monotone_early_KV_protocol'), (
        'Early Native graph publication is qualified only for Zero. Online cut renegotiation after receipt requires a separate monotone-frontier protocol.')
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from ..native_loop_2d_oracle import native_path
    from ..native_loop_2d_oracle.packing import packed_segments
    connector_class = native_path.ACTIVE_CONNECTOR
    records = []
    if not cfg.get('progressive_kv'):
        native_segments = connector_class.segments
        def segments(connector, transfer, info, tables):
            published = getattr(transfer, 'net_published_planes', {})
            if not published:
                yield from native_segments(connector, transfer, info, tables)
                return
            yield from packed_segments(info, tables, delta_layout(list(connector.layout(transfer)), published))
        connector_class.segments = segments
    initialize, close = StageEngine.__init__, StageEngine.close

    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        if engine.stage_config.role != 'prefill':
            return result
        graphs = engine._autonomous_prefill_graphs
        original = graphs.execute
        selected = set(cfg.get('early_transfer_depths', [16]))
        measured = [False]

        def executed(owner, call, hidden):
            result = original(call, hidden)
            metadata = call.metadata
            if not metadata.rows or metadata.cu_seqlens_q is None or engine.connector is None:
                return result
            depth = metadata.rows[0][1]+1
            if depth not in selected:
                return result
            if not measured[0]:
                measured[0] = (ROOT / 'raw' / cfg['name'] / 'measurement_begin.json').exists()
            if not measured[0]:
                return result
            assert all(row[1]+1 == depth for row in metadata.rows), 'Early graph publication requires one depth'
            for request_id, allocation in metadata.allocations:
                transfer = engine.connector.transfers.get(request_id)
                if (transfer is None or not transfer.outgoing or transfer.cancelled or transfer.compute_done
                        or not 0 <= transfer.request.sampling_params.seed-cfg['seed'] < cfg['count']):
                    continue
                rows = [row for row in metadata.rows if row[0] is allocation]
                positions = [row[2] for row in rows]
                if positions != list(range(len(transfer.request.prompt_token_ids))):
                    continue
                assert depth <= transfer.cut and depth < cfg['depth']
                # Native graph replay already records this event after its
                # output clones. No new GPU event is needed on a cache hit.
                event = owner.last_event if call.entry is not None else torch.cuda.Event()
                if call.entry is None:
                    event.record(torch.cuda.current_stream(owner.cache.device))
                publish(engine.connector, transfer, depth, event, records,
                    'NATIVE_GRAPH_EVENT_NO_SCHEDULER_CHECKPOINT')
            return result
        graphs.execute = MethodType(executed, graphs)
        return result

    def closed(engine):
        try:
            return close(engine)
        finally:
            write(ROOT / 'raw' / cfg['name'] / f'native_early_KV_{os.getpid()}.json', dict(
                role=engine.stage_config.role, records=records,
                scope='Exact progressive Native wire publication from existing graph completion events. Source remains a Native atomic submission. This is not an Online Loop ownership decision.'))
    StageEngine.__init__, StageEngine.close = initialized, closed
