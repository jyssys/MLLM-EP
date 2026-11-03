"""Diagnostic use of native recurrent graphs for packed prefill.

The native implementation owns metadata copies, KV validity, and stream events.
The optional D row limit applies only to packed D-local prefill.
Decode execution and projection arithmetic remain unchanged.
"""
from dataclasses import replace
from types import MethodType
import time
from .common import ROOT, append


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.cuda_graph import RecurrentGraphs
    initialize = StageEngine.__init__
    close = StageEngine.close

    def initialized(self, *args, **kwargs):
        result = initialize(self, *args, **kwargs)
        local_capacity = bool(cfg.get('capacity_mode') and cfg.get('P_graph_local_capacity'))
        role = self.stage_config.role
        row_limit = (cfg.get('P_prefill_graph_rows', 0) if role == 'prefill' or local_capacity
                     else cfg.get('D_prefill_graph_rows', 0) if role == 'decode'
                     else cfg.get('colocated_prefill_graph_rows', 0) if role == 'prefill_decode' else 0)
        if not row_limit:
            return result
        runner = self.model_runner
        execution = replace(runner.execution_config,
                            cuda_graph_max_batch_size=row_limit,
                            cuda_graph_max_graphs=cfg.get('P_prefill_graph_shapes', 4))
        graph_type = RecurrentGraphs
        if cfg.get('P_graph_bulk_written'):
            from .bulk_written import graph_class
            graph_type = graph_class()
        graphs = graph_type(self.model, self.cache_manager, execution, False)
        self._autonomous_prefill_graphs = graphs
        self._autonomous_prefill_graph_row_limit = row_limit
        self._autonomous_prefill_graph_host = dict(select_s=0., prepare_s=0., execute_submit_and_bookkeeping_s=0., calls=0)
        costs = self._autonomous_prefill_graph_host
        measured_costs = dict(costs)
        self._autonomous_prefill_graph_host_measured = measured_costs
        measured = [False]
        marker = ROOT/'raw'/cfg['name']/'measurement_begin.json'
        original = self.model.recurrent_prepared
        cache_owner = self.cache_manager

        def recurrent(model, hidden, metadata, cache, *, compute_gate=True):
            # Capture uses a native device-cache proxy. It must not re-enter
            # graph selection. Decode and non-packed calls remain unchanged.
            eligible = (cache is cache_owner and not compute_gate
                        and metadata.cu_seqlens_q is not None
                        and metadata.max_seqlen_q > 1
                        and len(hidden) <= row_limit)
            if not eligible:
                return original(hidden, metadata, cache, compute_gate=compute_gate)
            if not measured[0]:
                measured[0] = marker.exists()
            begin = time.perf_counter()
            call = graphs.select(hidden, metadata)
            selected = time.perf_counter()
            costs['select_s'] += selected-begin
            if call.entry is None:
                return original(hidden, metadata, cache, compute_gate=compute_gate)
            graphs.prepare_inputs(call, hidden)
            prepared = time.perf_counter()
            result = graphs.execute(call, hidden)
            finished = time.perf_counter()
            costs['prepare_s'] += prepared-selected
            costs['execute_submit_and_bookkeeping_s'] += finished-prepared
            costs['calls'] += 1
            if measured[0]:
                measured_costs['select_s'] += selected-begin
                measured_costs['prepare_s'] += prepared-selected
                measured_costs['execute_submit_and_bookkeeping_s'] += finished-prepared
                measured_costs['calls'] += 1
            return result

        self.model.recurrent_prepared = MethodType(recurrent, self.model)
        return result

    def closed(self):
        graphs = getattr(self, '_autonomous_prefill_graphs', None)
        if graphs is not None:
            append(ROOT/'raw'/cfg['name']/'native_prefill_graph_counts.jsonl',
                   dict(role=self.stage_config.role, captures=graphs.captures,
                        replays=graphs.replays, fallbacks=graphs.fallbacks,
                        row_limit=self._autonomous_prefill_graph_row_limit,
                        host_costs=getattr(self, '_autonomous_prefill_graph_host', {}),
                        host_costs_after_measurement_marker=getattr(self, '_autonomous_prefill_graph_host_measured', {}),
                        host_cost_scope='Host durations include capture and warmup. They can overlap GPU execution and are not exclusive E2E components.'))
        return close(self)

    StageEngine.__init__ = initialized
    StageEngine.close = closed
