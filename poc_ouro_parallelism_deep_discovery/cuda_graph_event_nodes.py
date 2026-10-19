"""Opt-in CUDA graph component timing. Importing this module does not load CUDA."""
import ctypes as C
from pathlib import Path
import os


def chain_order(nodes, edges, kinds=None):
    """Return the unique chain order. Reject unsupported graph shapes."""
    nodes, edges = list(nodes), list(edges)
    if not nodes or len(set(nodes)) != len(nodes) or 0 in nodes:
        raise ValueError('Graph nodes must be nonempty, unique, and nonzero')
    if len(set(edges)) != len(edges) or len(edges) != len(nodes) - 1:
        raise ValueError('The graph must have exactly one chain of edges')
    incoming = {node: [] for node in nodes}
    outgoing = {node: [] for node in nodes}
    for source, target in edges:
        if source not in incoming or target not in incoming:
            raise ValueError('An edge references an unknown node')
        incoming[target].append(source)
        outgoing[source].append(target)
    if any(len(incoming[n]) > 1 or len(outgoing[n]) > 1 for n in nodes):
        raise ValueError('The graph branches or merges')
    roots = [node for node in nodes if not incoming[node]]
    if len(roots) != 1:
        raise ValueError('The graph must have exactly one root')
    order, seen = [], set()
    node = roots[0]
    while node not in seen:
        order.append(node)
        seen.add(node)
        if not outgoing[node]:
            break
        node = outgoing[node][0]
    if len(order) != len(nodes):
        raise ValueError('The graph has a cycle or disconnected nodes')
    if kinds is not None and (set(kinds) != set(nodes) or any(kinds[n] not in (0, 1) for n in nodes)):
        raise ValueError('Only kernel and memcpy nodes are supported')
    return order


def cut_plan(order, cuts):
    """Locate event cuts without removing any original dependency."""
    cuts = list(cuts)
    if (any(type(cut) is not int for cut in cuts) or len(cuts) < 2
            or cuts[0] != 0 or cuts[-1] != len(order)
            or cuts != sorted(set(cuts))):
        raise ValueError('Cuts must increase strictly from zero to the node count')
    return [(cut, order[cut - 1] if cut else None,
             order[cut] if cut < len(order) else None) for cut in cuts]


def validate_native_dot(text, layers, live_order, expected_nodes):
    """Require the live repeated kernel pattern. Node count alone is insufficient."""
    from poc_ouro_parallelism_deep_discovery.scripts.parse_native_graph_dot import analyze
    summary, rows = analyze(text, layers)
    if (layers not in (24, 48) or len(rows) != expected_nodes or len(rows) != len(live_order)
            or not summary['single_chain'] or not summary.get('verified_repeated_pattern')
            or summary['node_kinds'] != {'KERNEL': len(rows) - 1, 'MEMCPY': 1}
            or [int(row['handle'], 16) for row in rows] != list(live_order)
            or len(summary['logical_pass_kernel_cuts']) != 5):
        raise RuntimeError('The live graph does not match the supported kernel-group pattern')
    cut_plan(live_order, summary['logical_pass_kernel_cuts'])
    return summary, rows


def validate_event_insertion(order, original_edges, plan, event_nodes, nodes, edges, kinds):
    """Check the exact added edges and the unchanged model-node order."""
    expected_nodes = set(order) | set(event_nodes)
    expected_edges = set(original_edges)
    if len(event_nodes) != len(plan) or len(expected_nodes) != len(order) + len(plan):
        raise ValueError('Event nodes must be distinct from original nodes')
    for (_, previous, following), event in zip(plan, event_nodes):
        if previous is not None:
            expected_edges.add((previous, event))
        if following is not None:
            expected_edges.add((event, following))
    if set(nodes) != expected_nodes or len(nodes) != len(expected_nodes):
        raise ValueError('Instrumentation changed the original node set')
    if set(edges) != expected_edges or len(edges) != len(expected_edges):
        raise ValueError('Instrumentation changed an unexpected edge')
    if any(kinds[event] != 7 for event in event_nodes):
        raise ValueError('An inserted node is not an event-record node')
    incoming = {n: 0 for n in nodes}
    outgoing = {n: [] for n in nodes}
    for source, target in edges:
        incoming[target] += 1
        outgoing[source].append(target)
    ready = [n for n in nodes if not incoming[n]]
    result = []
    while ready:
        if len(ready) != 1:
            raise ValueError('Instrumentation no longer has one total execution order')
        node = ready.pop()
        result.append(node)
        for target in outgoing[node]:
            incoming[target] -= 1
            if not incoming[target]:
                ready.append(target)
    by_cut = {cut: event for (cut, _, _), event in zip(plan, event_nodes)}
    expected_order = []
    for index in range(len(order) + 1):
        if index in by_cut:
            expected_order.append(by_cut[index])
        if index < len(order):
            expected_order.append(order[index])
    if result != expected_order:
        raise ValueError('Instrumentation changed the model-node order')


def torch_linked_runtime_path(maps_text):
    """Resolve a runtime symbol through the already-loaded Torch dependency tree."""
    torch_paths = {line.split(maxsplit=5)[5].strip() for line in maps_text.splitlines()
                   if '/libtorch_cuda.so' in line and len(line.split(maxsplit=5)) == 6}
    if len(torch_paths) != 1 or any(path.endswith(' (deleted)') for path in torch_paths):
        raise RuntimeError('One live libtorch_cuda is required for runtime ownership resolution')
    torch_library = C.CDLL(torch_paths.pop(), mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    function = torch_library.cudaRuntimeGetVersion
    address = C.cast(function, C.c_void_p)
    class DlInfo(C.Structure):
        _fields_ = [('dli_fname', C.c_char_p), ('dli_fbase', C.c_void_p),
                    ('dli_sname', C.c_char_p), ('dli_saddr', C.c_void_p)]
    linker = C.CDLL(None)
    linker.dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    linker.dladdr.restype = C.c_int
    information = DlInfo()
    if not linker.dladdr(address, C.byref(information)) or not information.dli_fname:
        raise RuntimeError('The Torch-linked runtime symbol has no resolved library owner')
    selected = os.fsdecode(information.dli_fname)
    if '/libcudart.so' not in selected or selected.endswith(' (deleted)'):
        raise RuntimeError('The resolved Torch runtime symbol does not belong to a live libcudart')
    return selected


def loaded_cudart_path(maps_text=None):
    """Use Torch's runtime. Multiple loaded runtimes need explicit symbol ownership."""
    live_process = maps_text is None
    if live_process:
        maps_text = Path('/proc/self/maps').read_text()
    paths = {line.split(maxsplit=5)[5].strip() for line in maps_text.splitlines()
             if '/libcudart.so' in line and len(line.split(maxsplit=5)) == 6}
    if live_process and len(paths) > 1 and not any(path.endswith(' (deleted)') for path in paths):
        selected = torch_linked_runtime_path(maps_text)
        by_resolved_path = {str(Path(path).resolve()): path for path in paths}
        normalized = str(Path(selected).resolve())
        if normalized not in by_resolved_path:
            raise RuntimeError('The Torch-linked runtime is absent from the loaded runtime mappings')
        return by_resolved_path[normalized]
    if len(paths) != 1 or any(path.endswith(' (deleted)') for path in paths):
        raise RuntimeError('Exactly one already-loaded libcudart is required. '
                           f'Observed runtime libraries: {sorted(paths)!r}')
    return paths.pop()


class Cudart:
    """Small CUDA 12.9 ABI wrapper. Construct this only in an authorized GPU probe."""
    def __init__(self):
        self.path = loaded_cudart_path()
        self.lib = C.CDLL(self.path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
        pointer, pointer_array, size_pointer = C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)
        signatures = {
            'cudaRuntimeGetVersion': [C.POINTER(C.c_int)],
            'cudaGetErrorString': [C.c_int],
            'cudaGraphGetNodes': [pointer, pointer_array, size_pointer],
            'cudaGraphGetEdges_v2': [pointer, pointer_array, pointer_array, pointer, size_pointer],
            'cudaGraphNodeGetType': [pointer, C.POINTER(C.c_int)],
            'cudaGraphAddEventRecordNode': [pointer_array, pointer, pointer_array, C.c_size_t, pointer],
            'cudaGraphAddDependencies': [pointer, pointer_array, pointer_array, C.c_size_t],
            'cudaGraphDestroyNode': [pointer],
            'cudaGraphDebugDotPrint': [pointer, C.c_char_p, C.c_uint],
            'cudaEventCreateWithFlags': [pointer_array, C.c_uint],
            'cudaEventDestroy': [pointer],
            'cudaEventElapsedTime': [C.POINTER(C.c_float), pointer, pointer],
            'cudaStreamSynchronize': [pointer],
            'cudaDeviceSynchronize': [],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, C.c_int
        self.lib.cudaGetErrorString.restype = C.c_char_p
        version = C.c_int()
        self.call('cudaRuntimeGetVersion', C.byref(version))
        self.version = version.value
        if self.version // 10 != 1209:
            raise RuntimeError(f'This diagnostic requires CUDA 12.9, not {self.version}')

    def call(self, name, *arguments):
        code = getattr(self.lib, name)(*arguments)
        if code:
            description = self.lib.cudaGetErrorString(code)
            raise RuntimeError(f'{name} failed with CUDA error {code}: {description!r}')

    def topology(self, graph):
        count = C.c_size_t()
        self.call('cudaGraphGetNodes', graph, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        self.call('cudaGraphGetNodes', graph, nodes, C.byref(count))
        node_list = list(nodes)[:count.value]
        count = C.c_size_t()
        # A NULL edgeData argument rejects non-default dependency data.
        self.call('cudaGraphGetEdges_v2', graph, None, None, None, C.byref(count))
        sources, targets = (C.c_void_p * count.value)(), (C.c_void_p * count.value)()
        self.call('cudaGraphGetEdges_v2', graph, sources, targets, None, C.byref(count))
        edges = list(zip(sources, targets))[:count.value]
        kinds = {}
        for node in node_list:
            kind = C.c_int()
            self.call('cudaGraphNodeGetType', node, C.byref(kind))
            kinds[node] = kind.value
        return node_list, edges, kinds

    def dot(self, graph, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        self.call('cudaGraphDebugDotPrint', graph, os.fsencode(path), 1)


class EventGraph:
    """Add reversible event cuts. Keep Torch's original replay bookkeeping."""
    def __init__(self, runtime, retained_graph, cuts, replay_callback=None):
        self.runtime, self.retained_graph = runtime, retained_graph
        runtime.call('cudaDeviceSynchronize')
        self.raw_graph = int(retained_graph.raw_cuda_graph())
        self.graph = self.raw_graph
        self.replay_callback = replay_callback or retained_graph.replay
        self.events, self.event_nodes = [], []
        self.launch_count, self.stream = 0, None
        self.closed = False
        self.cuts = list(cuts)
        original_nodes, original_edges, original_kinds = runtime.topology(self.raw_graph)
        original_order = chain_order(original_nodes, original_edges, original_kinds)
        self.original_topology = (set(original_nodes), set(original_edges), original_kinds)
        cut_plan(original_order, self.cuts)
        try:
            nodes, edges, order = original_nodes, original_edges, original_order
            plan = cut_plan(order, self.cuts)
            for _, previous, following in plan:
                event, node = C.c_void_p(), C.c_void_p()
                # Flags zero keep timing enabled. Torch Event objects are not passed here.
                runtime.call('cudaEventCreateWithFlags', C.byref(event), 0)
                self.events.append(event)
                dependencies = (C.c_void_p * 1)(previous) if previous is not None else None
                runtime.call('cudaGraphAddEventRecordNode', C.byref(node), self.graph,
                             dependencies, int(previous is not None), event)
                self.event_nodes.append(node.value)
                if following is not None:
                    runtime.call('cudaGraphAddDependencies', self.graph,
                                 (C.c_void_p * 1)(node.value), (C.c_void_p * 1)(following), 1)
            changed_nodes, changed_edges, changed_kinds = runtime.topology(self.graph)
            if any(changed_kinds[n] != original_kinds[n] for n in original_nodes):
                raise ValueError('Instrumentation changed an original node kind')
            validate_event_insertion(order, edges, plan, self.event_nodes,
                                     changed_nodes, changed_edges, changed_kinds)
            retained_graph.instantiate()
            self.original_nodes, self.original_edges = len(nodes), len(edges)
            self.instrumented_nodes, self.instrumented_edges = len(changed_nodes), len(changed_edges)
        except BaseException:
            self.close()
            raise

    def launch(self, stream):
        if self.closed:
            raise RuntimeError('The event diagnostic is closed')
        stream = int(stream)
        if self.stream is not None and self.stream != stream:
            raise RuntimeError('One observation cannot use multiple replay streams')
        if int(self.retained_graph.raw_cuda_graph()) != self.raw_graph:
            raise RuntimeError('The retained source graph changed')
        self.replay_callback()
        self.stream = stream
        self.launch_count += 1

    def last_replay(self):
        if self.closed:
            raise RuntimeError('The event diagnostic is closed')
        if not self.launch_count:
            raise RuntimeError('No instrumented replay occurred')
        self.runtime.call('cudaStreamSynchronize', self.stream)
        def elapsed(start, end):
            value = C.c_float()
            self.runtime.call('cudaEventElapsedTime', C.byref(value), start, end)
            if not 0 <= value.value < float('inf'):
                raise RuntimeError('The event interval is invalid')
            return value.value
        intervals = [elapsed(a, b) for a, b in zip(self.events, self.events[1:])]
        return dict(measurement='LAST_REPLAY_COMPONENT_DIAGNOSTIC',
                    observed_replays=self.launch_count, timed_replay_ordinal=self.launch_count,
                    graph_interval_ms=elapsed(self.events[0], self.events[-1]),
                    kernel_group_intervals_ms=intervals, kernel_group_cuts=self.cuts,
                    original_nodes=self.original_nodes, original_edges=self.original_edges,
                    instrumented_nodes=self.instrumented_nodes, instrumented_edges=self.instrumented_edges,
                    request_total_ms=None, request_extrapolation='FORBIDDEN',
                    boundary_scope='Fused kernel-group cuts, not exact semantic pass boundaries.',
                    observer_scope='Event nodes add overhead. No clean E2E timing.')

    def close(self):
        # The caller must complete outstanding launches before destruction.
        if self.closed:
            return
        if int(self.retained_graph.raw_cuda_graph()) != self.raw_graph:
            raise RuntimeError('Cannot restore a replaced source graph')
        if self.launch_count and self.stream is not None:
            self.runtime.call('cudaStreamSynchronize', self.stream)
            self.launch_count = 0
        if self.event_nodes:
            for node in reversed(self.event_nodes):
                self.runtime.call('cudaGraphDestroyNode', node)
            self.event_nodes.clear()
            nodes, edges, kinds = self.runtime.topology(self.graph)
            if (set(nodes), set(edges), kinds) != self.original_topology:
                raise RuntimeError('The source graph topology did not restore exactly')
            self.retained_graph.instantiate()
        for event in self.events:
            self.runtime.call('cudaEventDestroy', event)
        self.events.clear()
        self.closed = True
