"""Opt-in event cuts that preserve every existing CUDA edge-data byte."""
from collections import Counter

from poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes import (
    Cudart, EventGraph, chain_order, cut_plan)
from poc_ouro_parallelism_deep_discovery.cuda_graph_typed_edges import ReadOnlyGraphRuntime


def edge_records(snapshot):
    return Counter((edge['source'], edge['target'], edge['raw_hex'])
                   for edge in snapshot['edges'])


def require_default_cuts(snapshot, cuts):
    """Reject a cut across any nondefault dependency. Do not replace such edges."""
    nodes = [int(node['handle'], 16) for node in snapshot['nodes']]
    edges = [(int(edge['source'], 16), int(edge['target'], 16))
             for edge in snapshot['edges']]
    kinds = {int(node['handle'], 16): node['kind'] for node in snapshot['nodes']}
    order = chain_order(nodes, edges, kinds)
    plan = cut_plan(order, cuts)
    positions = {hex(node): index for index, node in enumerate(order)}
    if snapshot['reserved_nonzero_count']:
        raise ValueError('Reserved edge data is unsupported')
    for edge in snapshot['edges']:
        if edge['raw_hex'] not in ('0000000000000000', '0100010000000000'):
            raise ValueError('The graph has unsupported edge semantics')
    for cut, _, _ in plan:
        for edge in snapshot['edges']:
            if positions[edge['source']] < cut <= positions[edge['target']] and not edge['is_default']:
                raise ValueError('An event cut would strengthen a nondefault dependency')
    return order


def validate_typed_insertion(before, after, event_nodes):
    """Require unchanged original edges and default-only new event edges."""
    original, changed = edge_records(before), edge_records(after)
    if original - changed:
        raise ValueError('Instrumentation changed original dependency metadata')
    events = {hex(node) for node in event_nodes}
    for (source, target, raw), count in (changed - original).items():
        if count != 1 or raw != '0000000000000000' or not ({source, target} & events):
            raise ValueError('Instrumentation added an unsupported dependency')
    original_nodes = {node['handle']: node['kind'] for node in before['nodes']}
    changed_nodes = {node['handle']: node['kind'] for node in after['nodes']}
    if any(changed_nodes.get(node) != kind for node, kind in original_nodes.items()):
        raise ValueError('Instrumentation changed an original node')
    if set(changed_nodes) - set(original_nodes) != events:
        raise ValueError('Instrumentation added an unexpected node')
    if any(changed_nodes[node] != 7 for node in events):
        raise ValueError('An added node is not an event-record node')


class TypedCudart(Cudart):
    def __init__(self):
        super().__init__()
        self.reader = ReadOnlyGraphRuntime()
        if self.reader.path != self.path or self.reader.version != self.version:
            raise RuntimeError('Typed getters and graph mutation must use the same runtime')

    def snapshot(self, graph):
        return self.reader.snapshot(graph)

    def topology(self, graph):
        snapshot = self.snapshot(graph)
        return ([int(node['handle'], 16) for node in snapshot['nodes']],
                [(int(edge['source'], 16), int(edge['target'], 16)) for edge in snapshot['edges']],
                {int(node['handle'], 16): node['kind'] for node in snapshot['nodes']})


class TypedEventGraph(EventGraph):
    """Reuse the existing reversible cuts and additionally check all edge data."""
    def __init__(self, runtime, retained_graph, cuts, replay_callback=None):
        runtime.call('cudaDeviceSynchronize')
        self.typed_original = runtime.snapshot(int(retained_graph.raw_cuda_graph()))
        require_default_cuts(self.typed_original, cuts)
        super().__init__(runtime, retained_graph, cuts, replay_callback)
        try:
            self.typed_instrumented = runtime.snapshot(self.raw_graph)
            validate_typed_insertion(self.typed_original, self.typed_instrumented, self.event_nodes)
        except BaseException:
            self.close()
            raise

    def last_replay(self):
        return dict(super().last_replay(),
                    original_typed_fingerprint=self.typed_original['fingerprint'],
                    original_metadata_counts=self.typed_original['metadata_counts'],
                    original_nondefault_edges=self.typed_original['nondefault_edge_count'],
                    typed_dependencies_preserved=True,
                    boundary_scope='Default-edge kernel-group cuts. Not semantic pass completion or a full-serialization claim.')

    def close(self):
        super().close()
        restored = self.runtime.snapshot(self.raw_graph)
        if restored['fingerprint'] != self.typed_original['fingerprint']:
            raise RuntimeError('The original typed graph did not restore exactly')
