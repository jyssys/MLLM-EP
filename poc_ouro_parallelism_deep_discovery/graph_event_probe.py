"""Process-local event-node diagnostic. Select this worker only with explicit opt-in."""
from pathlib import Path
import hashlib
import os

if os.environ.get('OURO_CUDA_GRAPH_EVENT_DIAGNOSTIC') != '1':
    raise RuntimeError('Set OURO_CUDA_GRAPH_EVENT_DIAGNOSTIC=1 for this diagnostic worker')

import torch
from poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes import (
    Cudart, EventGraph, chain_order, validate_native_dot)
from poc_ouro_parallelism_deep_discovery.worker_probe import CampaignProbe

if os.environ.get('OURO_CUDA_GRAPH_TYPED_EVENT_DIAGNOSTIC') == '1':
    from poc_ouro_parallelism_deep_discovery.cuda_graph_typed_event_nodes import (
        TypedCudart as Cudart, TypedEventGraph as EventGraph)

_graphs, _timers, _observed = {}, {}, {}
_armed, _runtime, _settings = False, None, None
_original_new = torch.cuda.CUDAGraph.__new__
_original_init = torch.cuda.CUDAGraph.__init__
_original_replay = torch.cuda.CUDAGraph.replay


def _new(cls, keep_graph=False):
    return _original_new(cls, keep_graph=True)


def _init(graph, *args, **kwargs):
    if len(args) > 1 or set(kwargs) - {'keep_graph'}:
        raise TypeError('Unsupported CUDAGraph constructor arguments')
    _original_init(graph, True)
    _graphs[id(graph)] = graph


def _replay(graph, *args, **kwargs):
    if not _armed:
        return _original_replay(graph, *args, **kwargs)
    if args or kwargs or torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Observe only ordinary post-capture graph replay calls')
    identifier = id(graph)
    _observed[identifier] = _observed.get(identifier, 0) + 1
    if identifier not in _timers:
        raw = int(graph.raw_cuda_graph())
        nodes, edges, kinds = _runtime.topology(raw)
        if len(nodes) != _settings['expected_nodes']:
            return _original_replay(graph)
        order = chain_order(nodes, edges, kinds)
        output = Path(os.environ['LOOPED_KV_OUTPUT'])
        path = output / f"rank{_settings['rank']}_event_source_graph{len(_timers)}.dot"
        _runtime.dot(raw, path)
        payload = path.read_bytes()
        summary, rows = validate_native_dot(payload.decode(), _settings['layers'], order,
                                            _settings['expected_nodes'])
        timer = EventGraph(_runtime, graph, summary['logical_pass_kernel_cuts'],
                           replay_callback=lambda: _original_replay(graph))
        timer.source_metadata = dict(source_dot=str(path), source_sha256=hashlib.sha256(payload).hexdigest(),
                                     topology=summary, cut_scope=summary['cut_scope'],
                                     cut_kernels=[rows[cut]['kernel'] for cut in timer.cuts[:-1]])
        _timers[identifier] = timer
    return _timers[identifier].launch(torch.cuda.current_stream().cuda_stream)


# These wrappers exist only in this explicitly selected worker process.
torch.cuda.CUDAGraph.__new__ = staticmethod(_new)
torch.cuda.CUDAGraph.__init__ = _init
torch.cuda.CUDAGraph.replay = _replay


class GraphEventProbe(CampaignProbe):
    def graph_events_arm(self, expected_nodes=1154, layers=24):
        global _armed, _runtime, _settings
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        if _armed or _timers or _settings is not None:
            raise RuntimeError('Use a fresh worker for each armed diagnostic request')
        if get_tensor_model_parallel_world_size() != 1 or layers not in (24, 48) or expected_nodes < 2:
            raise ValueError('This diagnostic supports one P1 request and a declared graph size')
        torch.cuda.synchronize()
        _runtime = Cudart()
        _settings = dict(expected_nodes=expected_nodes, layers=layers,
                         rank=get_tensor_model_parallel_rank())
        _observed.clear()
        _armed = True
        return dict(**_settings, runtime_library=_runtime.path, runtime_version=_runtime.version,
                    measurement='LAST_REPLAY_COMPONENT_DIAGNOSTIC', request_extrapolation='FORBIDDEN')

    def graph_events_finish(self):
        global _armed
        _armed = False
        torch.cuda.synchronize()
        records = []
        try:
            if not _timers:
                raise RuntimeError('No graph matched the declared diagnostic topology')
            for identifier, timer in _timers.items():
                records.append(dict(graph_object_id=identifier, **timer.last_replay(), **timer.source_metadata))
        finally:
            for timer in _timers.values():
                timer.close()
            _timers.clear()
        return dict(rank=_settings['rank'], records=records,
                    total_observed_replays=sum(_observed.values()),
                    observed_replays_by_graph={str(k): v for k, v in _observed.items()},
                    topology_restored=True, original_torch_replay_used=True,
                    evidence_class='GPU_MICROBENCH',
                    timing='Each record contains only that graph last replay. Never multiply it by replay count.',
                    scope='One controlled P1 request. Fused kernel-group cuts. No semantic pass or request-total timing.')
