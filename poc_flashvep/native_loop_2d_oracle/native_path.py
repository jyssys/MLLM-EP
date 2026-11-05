"""Keep atomic full requests. Materialize state only for an actual partial cut.

The candidate uses native non-wavefront progress, not the wavefront callback.
It keeps the original wavefront guard unchanged. It is not GPU-qualified yet.
"""
import os
import time
from dataclasses import replace

ACTIVE_CONNECTOR = None


class NativeFactory:
    """Pickle a stable global factory, not a local candidate class."""
    @classmethod
    def create(cls, engine, *, config, notify):
        if ACTIVE_CONNECTOR is None:
            raise RuntimeError('Native connector was not installed in this owner')
        result = ACTIVE_CONNECTOR.create(engine, config=config, notify=notify)
        import torch
        from .common import ROOT, write
        write(ROOT / 'raw' / result.cfg['name'] / f'native_factory_memory_{os.getpid()}.json', dict(
            role=engine.stage_config.role, hidden_staging_bytes=result.hidden.numel() * result.hidden.element_size(),
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            device_total_bytes=torch.cuda.get_device_properties(engine.model_runner.device).total_memory,
            scope='After native connector construction and registration. Allocator totals include component tensors. Not peak HBM or complete opaque graph attribution.'))
        return result


def install(cfg):
    global ACTIVE_CONNECTOR
    if not cfg.get('native_path'):
        return
    import torch
    from vllm_rlt.worker import buffers, model_runner
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.core.scheduling_policy import RefillPolicy
    from vllm_rlt.request import RequestPhase
    from vllm_rlt.worker.executor import Operation
    from vllm_rlt.pd.connector import PDStageConnector
    from vllm_rlt.pd.transport import NixlConnector
    from ..adaptive_recurrence_cut.connector import CutConnector
    from ..looped_pd_abc.boundary import BoundaryConnector
    from .common import ROOT, write
    assert not cfg['wavefront'] and cfg['scheduler_mode'] == 'refill'
    if getattr(Scheduler, '_native_continuation_candidate', False):
        return
    Scheduler._native_continuation_candidate = True

    # Keep zero-offload's full-Prompt submission path. Provision continuation
    # storage in both cases so the feature-enabled memory cost is visible.
    maximum = cfg['state_capacity_tokens']
    def state_rows(config, scheduler, *, staged_prefill=None):
        return maximum
    buffers.prefill_stage_rows = model_runner.prefill_stage_rows = state_rows

    # Imported partial requests use existing non-wavefront recurrent phases.
    # Full requests remain PREFILL and use the original native runner path.
    phases = (RequestPhase.PREFILL, RequestPhase.PREFILLING_PRELUDE,
              RequestPhase.PREFILLING_RECURRENT)
    RefillPolicy.PREFILL_PHASES = phases
    from vllm_rlt.core.scheduling_policy import PromptOnlyPolicy
    policy_init = PromptOnlyPolicy.__init__
    def prompt_policy(self, config, *, staged_prefill):
        policy_init(self, config, staged_prefill=staged_prefill)
        self.PREFILL_PHASES = phases
    PromptOnlyPolicy.__init__ = prompt_policy

    planned = Scheduler._planned_loop_count
    def grouped(self, items):
        selected = {item.phase for item in items}
        if selected == {RequestPhase.PREFILL}:
            return min(getattr(item.request, 'arc_export_cut', self.total_ut_steps) for item in items)
        if selected == {RequestPhase.PREFILLING_RECURRENT}:
            counts = []
            for item in items:
                r = item.request
                end = item.token_start + item.token_count
                target = getattr(r, 'arc_export_cut', self.total_ut_steps)
                if (item.token_start == 0 and end == len(r.prompt_token_ids)
                        and r.prefill_depth_ranges[item.depth][1] >= end
                        and all(left == 0 for left, right in r.prefill_depth_ranges[item.depth:target])):
                    counts.append(target - item.depth)
                else:
                    counts.append(self._max_contiguous_prefill_loops(item))
            return min(counts)
        return planned(self, items)
    Scheduler._planned_loop_count = grouped

    progress = Scheduler._apply_prefill_progress
    def advance(self, item, phase, *, loop_count=None, event=None):
        if phase == RequestPhase.PREFILL and loop_count < self.total_ut_steps:
            if item.token_start != 0 or item.token_count != len(item.request.prompt_token_ids):
                raise ValueError('Native partial loop continuation requires the full prompt span')
            return self._apply_split_prefill_progress(item.request, RequestPhase.PREFILLING_PRELUDE,
                item, loop_count=loop_count, event=event)
        result = progress(self, item, phase, loop_count=loop_count, event=event)
        if phase == RequestPhase.PREFILL and cfg.get('feature_native_wire_v5'):
            r = item.request
            end = item.token_start + item.token_count
            r.num_preluded_tokens = end
            for depth in range(self.total_ut_steps):
                r.prefill_depth_ranges[depth][0] = end
                r.prefill_depth_ranges[depth][1] = end
        return result
    Scheduler._apply_prefill_progress = advance

    # The original fused Prelude path also appends Huginn Coda-KV work.
    # That work is valid only at full depth. D performs it after continuation.
    prepare = model_runner.ModelRunner._prepare_loop_inputs
    def inputs(self, batch, packed_metadata=None):
        result = prepare(self, batch, packed_metadata)
        partial = any(span.depth + batch.loop_count < cfg['depth'] for span in batch.spans)
        if partial:
            result = tuple(entry for entry in result if entry.forward.operation != Operation.CODA)
        return result
    model_runner.ModelRunner._prepare_loop_inputs = inputs

    execute = model_runner.ModelRunner._execute_batch
    def run_batch(self, batch, prepared):
        partial = (batch.operation == Operation.PRELUDE and batch.loop_count
            and any(span.depth + batch.loop_count < cfg['depth'] for span in batch.spans))
        if not partial:
            return execute(self, batch, prepared)
        original = self.runtime.forward
        def forward(entry):
            result = original(entry)
            if entry.operation == Operation.PRELUDE:
                offset = 0
                for span in batch.spans:
                    slot = self._ensure_request_slot(span.request)
                    self.prefill_states[slot, span.token_start:span.token_start + span.token_count].copy_(
                        result.hidden[offset:offset + span.token_count])
                    offset += span.token_count
            return result
        self.runtime.forward = forward
        try:
            return execute(self, batch, prepared)
        finally:
            self.runtime.forward = original
    model_runner.ModelRunner._execute_batch = run_batch

    class NativeCutConnector(CutConnector):
        def segments(self, transfer, info, tables):
            if cfg.get('feature_native_wire_v5'):
                from .packing import packed_segments
                yield from packed_segments(info, tables, self.layout(transfer))
            else:
                yield from super().segments(transfer, info, tables)

        @classmethod
        def create(cls, engine, *, config, notify):
            # Same validated partial-state format and native transport/ACK ABI.
            # No claim of an incremental native KV wire protocol is made here.
            parameter = next(engine.model.parameters())
            limit = engine.scheduler.config.max_num_seqs + (config.max_receiving_requests
                if engine.stage_config.produces_tokens else config.max_draining_requests)
            hidden = torch.zeros((limit, maximum * engine.model.config.hidden_size),
                                 device=parameter.device, dtype=parameter.dtype)
            self = cls(engine, hidden, NixlConnector(engine.stage_config.stage_id,
                engine.cache_manager, hidden, config), config, notify=notify)
            self.cfg = cfg
            self.original_accept = engine.accept_import
            engine.accept_import = self.accept_cut
            original_notify = self.notify
            def notification(kind, **fields):
                if kind == 'commit':
                    t = self.transfers[fields['tid']]
                    fields.update(rng_state=getattr(t, 'rng_state', None), arc_cut=t.cut)
                    if cfg.get('verify_transfer'):
                        fields['fingerprint'] = self.fingerprint(t)
                return original_notify(kind, **fields)
            self.notify = notification
            return self

    # Zero-offload uses the native transport, with the same continuation buffer
    # provision. Partial runs use the existing exact snapshot format once.
    import vllm_rlt.pd.engine as pd
    if cfg.get('native_partial_cut') is not None or cfg.get('feature_native_wire_v5'):
        cut = cfg.get('native_partial_cut') or cfg['depth']
        assert 0 < cut < cfg['depth'] or (cut == cfg['depth'] and (
            cfg.get('feature_native_wire_v5') and cfg.get('native_partial_cut') is None
            or cfg.get('native_frontier') is not None and cfg['native_frontier'][0] == 0))
        if cfg.get('feature_native_wire_v5') and cfg.get('native_partial_cut') is None:
            assert cut == cfg['depth']
        pd.PDStageConnector = NativeCutConnector
        ACTIVE_CONNECTOR = NativeCutConnector
        from vllm_rlt.pd.engine import PDEngine
        initialize_pd = PDEngine.__init__
        def init(engine, *args, **kwargs):
            # The reused worker installs its ordinary connector again in main().
            # Select the candidate at the actual engine construction boundary.
            pd.PDStageConnector = NativeFactory
            result = initialize_pd(engine, *args, **kwargs)
            send = engine._send
            def sending(destination, kind, **fields):
                if kind in ('reserve', 'prefill', 'commit'):
                    selected = cut
                    pending = engine.transfers.get(fields.get('tid'))
                    if pending is not None and cfg.get('native_request_cuts'):
                        index = pending.request.sampling_params.seed - cfg['seed']
                        if 0 <= index < cfg['count']:
                            selected = cfg['native_request_cuts'][index]
                    fields['arc_cut'] = selected
                    if pending is not None and cfg.get('native_frontier') is not None:
                        import json
                        from .staircase import resolve_prefix
                        marker = ROOT / 'raw' / cfg['name'] / 'native_graph_warmup_phase.json'
                        phase = json.loads(marker.read_text()) if marker.exists() else {}
                        request = pending.request
                        fields['prefix_k'] = resolve_prefix(cfg, len(request.prompt_token_ids),
                            request.request_id, request.sampling_params.seed, phase)
                return send(destination, kind, **fields)
            engine._send = sending
            return result
        PDEngine.__init__ = init
    else:
        pd.PDStageConnector = BoundaryConnector

    from vllm_rlt.engine.stage_engine import StageEngine
    initialize_stage = StageEngine.__init__
    def stage(self, *args, **kwargs):
        result = initialize_stage(self, *args, **kwargs)
        runner = self.model_runner
        graphs = getattr(self, '_autonomous_prefill_graphs', None)
        if cfg.get('feature_native_wire_v5') and graphs is not None:
            write(ROOT / 'raw' / cfg['name'] / f'native_prefill_shapes_{os.getpid()}.json', dict(
                role=self.stage_config.role, graph_class=type(graphs).__name__,
                capture_sizes=getattr(graphs, 'capture_sizes', None),
                maximum_cached_graphs=graphs.execution.cuda_graph_max_graphs,
                scope='No graph selection change. Native packed RecurrentGraphs keys exact query count and context geometry. Do not confuse this with bucketed SteadyRecurrentGraphs. Token and 2D exact large shapes already use the same native implementation and graph budget.'))
        write(ROOT / 'raw' / cfg['name'] / f'native_path_memory_{os.getpid()}.json', dict(
            role=self.stage_config.role, time=time.monotonic(),
            prefill_state_bytes=runner.prefill_states.numel() * runner.prefill_states.element_size(),
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            partial_cut=cfg.get('native_partial_cut'), feature_enabled=True,
            scope='Feature-enabled initial allocator totals overlap component tensors. Not peak or exclusive graph-pool memory.'))
        return result
    StageEngine.__init__ = stage
    if cfg.get('native_frontier') is not None:
        from .staircase import install as install_frontier
        install_frontier(cfg, NativeCutConnector)
