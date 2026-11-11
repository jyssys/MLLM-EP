"""Optional, request-scoped checkpoint exposure on the qualified protocol."""
import os
import time
from .common import ROOT, write
from .planner import choose_checkpoint


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import RequestPhase as P
    from ..native_loop_2d_oracle import native_path
    Connector = native_path.ACTIVE_CONNECTOR
    records = []
    folder = ROOT / 'raw' / cfg['name']
    previous_init, previous_close = StageEngine.__init__, StageEngine.close

    if cfg.get('progressive_kv'):
        from types import SimpleNamespace
        from ..native_loop_2d_oracle.packing import packed_segments
        from ..looped_pd_abc.segments import coalesce
        from vllm_rlt.pd.transport import partition_segments
        from .progressive import delta_layout, disjoint_publication
        native_submitted = Connector.prefill_submitted
        native_segments = Connector.segments
        native_can_schedule = Connector.can_schedule
        native_ready_ids = Connector.ready_P_ids

        def segments(connector, transfer, info, tables):
            published = getattr(transfer, 'net_published_planes', {})
            if not published:
                yield from native_segments(connector, transfer, info, tables)
                return
            layout = delta_layout(list(connector.layout(transfer)), published)
            yield from packed_segments(info, tables, layout)

        def submitted(connector, request, *, phase, start, progress, event, stream):
            transfer = connector.transfers[request.request_id]
            depth = sum(left == len(request.prompt_token_ids) for left, right in request.prefill_depth_ranges)
            if (transfer.outgoing and not transfer.cancelled and not transfer.compute_done
                    and 0 < depth < cfg['depth'] and depth in cfg['boundary_checkpoints']
                    and getattr(request, 'idle_armed', False)):
                assert transfer.cut == cfg['depth'], 'Only source-owned boundaries can publish early KV'
                assert event is not None, 'Early KV requires the producing CUDA event'
                view = SimpleNamespace(cut=depth, request=request)
                published = getattr(transfer, 'net_published_planes', {})
                layout = delta_layout(list(connector.layout(view)), published)
                if layout:
                    if cfg.get('monotone_early_KV_protocol'):
                        assert connector.cache.storage_depths == cfg['depth'], (
                            'Early transfer requires disjoint recurrent KV storage')
                        connector.notify('idle_immutable_KV_frontier', tid=transfer.tid,
                            depth=depth, KV_only=True,
                            full_prompt=len(request.prompt_token_ids))
                    peer = connector.transport.peers[transfer.peer]
                    local = list(packed_segments(connector.transport.info,
                        connector.cache.plane_block_tables(transfer.tid), layout))
                    remote = list(packed_segments(peer, transfer.target_tables, layout))
                    local, remote = coalesce(local, remote)
                    before = transfer.next_sequence
                    for a, b, amount in partition_segments(local, remote,
                            connector.config.transfer_chunk_bytes, connector.config.max_transfer_descriptors):
                        transfer.chunks.append((event, transfer.next_sequence, a, b, amount))
                        transfer.next_sequence += 1
                    disjoint_publication(layout, published)
                    transfer.net_published_planes = published
                    records.append(dict(time=time.monotonic(), event='EARLY_KV_QUEUED',
                        tid=transfer.tid, seed=request.sampling_params.seed, depth=depth,
                        KV_bytes=sum(s[1] for s in local), fragments=transfer.next_sequence-before,
                        scope='Exact immutable KV planes. Native notifications cannot activate D before the final commit and hidden handoff.'))
            return native_submitted(connector, request, phase=phase, start=start,
                progress=progress, event=event, stream=stream)

        def can_schedule(connector, request):
            transfer = connector.transfers.get(request.request_id)
            if (transfer is not None and transfer.outgoing and transfer.request is request
                    and getattr(transfer, 'net_published_planes', None) and not transfer.compute_done):
                boundary = connector.boundaries.get(transfer.tid)
                # Future depths use disjoint KV planes. Publication must not
                # block their computation, but producer dependencies remain.
                return bool(not transfer.cancelled and transfer.tid not in connector.pending_callbacks
                    and (boundary is None or boundary.status == 'running')
                    and (transfer.last_event is None or transfer.last_event.query()))
            return native_can_schedule(connector, request)

        def ready_ids(connector, executable=False):
            result = set(native_ready_ids(connector, executable))
            pending = connector.pending_P_device_ids() if executable else set()
            for transfer in connector.transfers.values():
                request = transfer.request
                if (not transfer.outgoing or transfer.cancelled or transfer.compute_done
                        or not getattr(transfer, 'net_published_planes', None)
                        or request.phase not in (P.WAITING, P.PREFILL, P.PREFILLING_PRELUDE, P.PREFILLING_RECURRENT)):
                    continue
                if transfer.last_event is not None and not transfer.last_event.query():
                    continue
                boundary = connector.boundaries.get(transfer.tid)
                if executable and (request.request_id in pending or transfer.tid in connector.pending_callbacks
                        or (boundary is not None and boundary.status != 'running')):
                    continue
                # Immutable early writes do not erase the remaining source
                # computation. Actual executable pressure still excludes waits.
                result.add(request.request_id)
            return result

        Connector.segments = segments
        Connector.prefill_submitted = submitted
        Connector.can_schedule = can_schedule
        Connector.ready_P_ids = ready_ids

    def initialize(engine, *args, **kwargs):
        result = previous_init(engine, *args, **kwargs)
        if engine.stage_config.role != 'prefill' or not cfg.get('selective_checkpoints'):
            return result
        scheduler = engine.scheduler
        original = scheduler._planned_loop_count
        total = cfg['depth']
        checkpoints = cfg.get('boundary_checkpoints',
            [1, 2, 3] if total == 4 else [24, 28, 30, 31])

        def selected(items):
            if not all(i.phase in (P.PREFILL, P.PREFILLING_RECURRENT) for i in items):
                return original(items)
            # The worker creates its connector after StageEngine construction.
            connector = engine.connector
            pressure = len(connector.ready_P_ids() | connector.pending_P_device_ids())
            ad = connector.advertisement
            pressure += ad.get('P_credit_clear_waiters', 0) if ad else 0
            now = time.monotonic()
            for item in items:
                r = item.request
                index = r.sampling_params.seed - cfg['seed']
                if not 0 <= index < cfg['count'] or hasattr(r, 'idle_warm_width'):
                    continue
                depth = 0 if item.phase == P.PREFILL else item.depth
                choice = choose_checkpoint(depth, total, checkpoints, ad, now, cfg, pressure)
                r.idle_armed = choice.target < total
                r.idle_next_checkpoint = choice.target
                records.append(dict(time=now, event='SELECTIVE_CHECKPOINT', tid=r.request_id,
                    seed=r.sampling_params.seed, depth=depth, target=choice.target,
                    reason=choice.reason, predicted_D_ms=choice.predicted_D_ms, P_pressure=pressure))
            # The qualified planner still owns grouped-loop geometry.
            # Its reconsideration must not overwrite a deliberate full Native choice.
            monitor = cfg.get('source_monitor_boundaries')
            cfg['source_monitor_boundaries'] = False
            try:
                return original(items)
            finally:
                cfg['source_monitor_boundaries'] = monitor

        scheduler._planned_loop_count = selected
        return result

    def close(engine):
        if records:
            write(folder / f'net_benefit_decisions_{os.getpid()}.json', dict(records=records,
                scope='Observed request-scoped checkpoint choices. Actual admission and commit use the unchanged boundary protocol.'))
        return previous_close(engine)

    StageEngine.__init__, StageEngine.close = initialize, close
