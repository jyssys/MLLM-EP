"""Read-only CUDA 12.9 edge metadata. This module does not load CUDA on import."""
from collections import Counter
import ctypes as C
import hashlib
import json
import os

from poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes import loaded_cudart_path


class CudaGraphEdgeData(C.Structure):
    _fields_ = [('from_port', C.c_ubyte), ('to_port', C.c_ubyte),
                ('type', C.c_ubyte), ('reserved', C.c_ubyte * 5)]


def edge_abi():
    return dict(size=C.sizeof(CudaGraphEdgeData), alignment=C.alignment(CudaGraphEdgeData),
                offsets={name: getattr(CudaGraphEdgeData, name).offset
                         for name in ('from_port', 'to_port', 'type', 'reserved')},
                reserved_bytes=5)


def decode_edge(data):
    raw = bytes(data)
    if len(raw) != 8:
        raise RuntimeError('Unexpected cudaGraphEdgeData size')
    return dict(from_port=int(data.from_port), to_port=int(data.to_port), type=int(data.type),
                reserved=list(data.reserved), raw_hex=raw.hex(), raw_bytes=list(raw),
                is_default=not any(raw), reserved_nonzero=any(data.reserved),
                dependency_type={0: 'DEFAULT', 1: 'PROGRAMMATIC'}.get(data.type, 'UNKNOWN'),
                source_port={0: 'FULL_COMPLETION', 1: 'PROGRAMMATIC', 2: 'LAUNCH_COMPLETION'}
                            .get(data.from_port, 'UNKNOWN'))


def snapshot_fingerprint(snapshot):
    canonical = dict(graph_handle=snapshot['graph_handle'],
                     nodes=sorted((n['handle'], n['kind']) for n in snapshot['nodes']),
                     edges=sorted((e['source'], e['target'], e['raw_hex']) for e in snapshot['edges']))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


class ReadOnlyGraphRuntime:
    """Bind only graph getters and runtime identification. Never bind graph mutation."""
    def __init__(self, library=None):
        if edge_abi() != dict(size=8, alignment=1,
                             offsets=dict(from_port=0, to_port=1, type=2, reserved=3), reserved_bytes=5):
            raise RuntimeError('The host edge-data ABI differs from CUDA 12.9')
        self.path = loaded_cudart_path() if library is None else 'CPU_TEST_DOUBLE'
        self.lib = C.CDLL(self.path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD) if library is None else library
        pointer, pointers, sizes = C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)
        self.bound_functions = {
            'cudaRuntimeGetVersion': [C.POINTER(C.c_int)],
            'cudaGetErrorString': [C.c_int],
            'cudaGraphGetNodes': [pointer, pointers, sizes],
            'cudaGraphGetEdges_v2': [pointer, pointers, pointers, C.POINTER(CudaGraphEdgeData), sizes],
            'cudaGraphNodeGetType': [pointer, C.POINTER(C.c_int)],
        }
        for name, arguments in self.bound_functions.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, C.c_int
        self.lib.cudaGetErrorString.restype = C.c_char_p
        version = C.c_int()
        self.call('cudaRuntimeGetVersion', C.byref(version))
        self.version = version.value
        if self.version // 10 != 1209:
            raise RuntimeError(f'This typed-edge export requires CUDA 12.9, not {self.version}')

    def call(self, name, *arguments):
        if name not in self.bound_functions:
            raise RuntimeError('The read-only runtime refuses an unbound API')
        code = getattr(self.lib, name)(*arguments)
        if code:
            raise RuntimeError(f'{name} failed with CUDA error {code}: {self.lib.cudaGetErrorString(code)!r}')

    def snapshot(self, graph):
        count = C.c_size_t()
        self.call('cudaGraphGetNodes', graph, None, C.byref(count))
        capacity = count.value
        if capacity > 100000:
            raise RuntimeError('The node count exceeds the bounded diagnostic scope')
        nodes = (C.c_void_p * capacity)()
        self.call('cudaGraphGetNodes', graph, nodes, C.byref(count))
        if count.value > capacity:
            raise RuntimeError('The graph changed during the node query')
        nodes = list(nodes)[:count.value]
        if None in nodes or len(set(nodes)) != len(nodes):
            raise RuntimeError('The node query returned null or duplicate handles')
        node_rows = []
        for node in nodes:
            kind = C.c_int()
            self.call('cudaGraphNodeGetType', node, C.byref(kind))
            node_rows.append(dict(handle=hex(node), kind=kind.value))
        count = C.c_size_t()
        # The all-NULL count-only query is documented. The data query is fully typed.
        self.call('cudaGraphGetEdges_v2', graph, None, None, None, C.byref(count))
        capacity = count.value
        if capacity > 1000000:
            raise RuntimeError('The edge count exceeds the bounded diagnostic scope')
        sources, targets = (C.c_void_p * capacity)(), (C.c_void_p * capacity)()
        metadata = (CudaGraphEdgeData * capacity)()
        if capacity:
            self.call('cudaGraphGetEdges_v2', graph, sources, targets, metadata, C.byref(count))
        if count.value > capacity:
            raise RuntimeError('The graph changed during the edge query')
        edge_rows, node_set = [], set(nodes)
        for index in range(count.value):
            source, target = sources[index], targets[index]
            if source not in node_set or target not in node_set:
                raise RuntimeError('An edge query returned an unknown node')
            edge_rows.append(dict(api_index=index, source=hex(source), target=hex(target),
                                  **decode_edge(metadata[index])))
        result = dict(graph_handle=hex(int(graph)), abi=edge_abi(), nodes=node_rows, edges=edge_rows,
                      node_count=len(node_rows), edge_count=len(edge_rows),
                      nondefault_edge_count=sum(not e['is_default'] for e in edge_rows),
                      reserved_nonzero_count=sum(e['reserved_nonzero'] for e in edge_rows),
                      metadata_counts=dict(Counter(e['raw_hex'] for e in edge_rows)),
                      scope='Top-level raw-graph metadata only. No recursive child-graph query.',
                      timing='NOT_MEASURED')
        result['fingerprint'] = snapshot_fingerprint(result)
        return result


def inspect_nominal_cuts(snapshot, dot_text, layers):
    """Annotate kernel-group cuts. Never convert typed edges to full fences."""
    from poc_ouro_parallelism_deep_discovery.scripts.parse_native_graph_dot import analyze, parse_graph
    try:
        summary, rows = analyze(dot_text, layers)
        dot_nodes, dot_edges, *_ = parse_graph(dot_text)
    except (ValueError, KeyError) as error:
        return dict(status='CUT_ANALYSIS_UNAVAILABLE', error=str(error), timing='NOT_MEASURED')
    dot_handles = {key: hex(int(node['handle'], 16)) for key, node in dot_nodes.items()}
    dot_pairs = Counter((dot_handles[a], dot_handles[b]) for a, b in dot_edges)
    runtime_pairs = Counter((edge['source'], edge['target']) for edge in snapshot['edges'])
    runtime_nodes = {node['handle'] for node in snapshot['nodes']}
    if set(dot_handles.values()) != runtime_nodes or dot_pairs != runtime_pairs:
        return dict(status='DOT_RUNTIME_MISMATCH', timing='NOT_MEASURED')
    if not summary.get('verified_repeated_pattern'):
        return dict(status='NO_VERIFIED_REPEATED_PATTERN', topology=summary, timing='NOT_MEASURED')
    positions = {hex(int(row['handle'], 16)): i for i, row in enumerate(rows)}
    cuts = []
    for cut in summary['logical_pass_kernel_cuts']:
        crossing = [edge for edge in snapshot['edges']
                    if positions[edge['source']] < cut <= positions[edge['target']]]
        cuts.append(dict(position=cut, crossing_edge_indices=[e['api_index'] for e in crossing],
                         nondefault_edge_indices=[e['api_index'] for e in crossing if not e['is_default']],
                         on_nondefault_edge=any(not e['is_default'] for e in crossing) if crossing else None,
                         location='INTERIOR_EDGE' if crossing else 'GRAPH_ENDPOINT',
                         following_kernel=rows[cut]['kernel'] if cut < len(rows) else None))
    return dict(status='VERIFIED_KERNEL_GROUP_CUTS', cuts=cuts, topology=summary,
                cut_scope=summary['cut_scope'], timing='NOT_MEASURED', overlap_duration='UNKNOWN')
