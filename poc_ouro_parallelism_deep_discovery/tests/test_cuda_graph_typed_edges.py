"""CPU-only ABI and typed-topology tests. No CUDA library is loaded."""
import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from poc_ouro_parallelism_deep_discovery.cuda_graph_typed_edges import (
    CudaGraphEdgeData, ReadOnlyGraphRuntime, decode_edge, edge_abi,
    inspect_nominal_cuts, snapshot_fingerprint)
from poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes import Cudart
from poc_ouro_parallelism_deep_discovery.scripts.parse_native_graph_dot import analyze


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class FakeLibrary:
    def __init__(self):
        self.nodes = [30, 10, 20]
        self.edge_rows = [(10, 20, bytes([1, 0, 1, 0, 0, 0, 0, 0])),
                          (20, 30, bytes([2, 0, 1, 17, 34, 51, 68, 85]))]
        self.data_fetches = 0
        self.cudaRuntimeGetVersion = Function(lambda p: self.set_scalar(p, C.c_int, 12090))
        self.cudaGetErrorString = Function(lambda code: b'CPU fixture: lossy query')
        self.cudaGraphGetNodes = Function(self.get_nodes)
        self.cudaGraphGetEdges_v2 = Function(self.get_edges)
        self.cudaGraphNodeGetType = Function(lambda node, p: self.set_scalar(p, C.c_int, int(node == 30)))

    @staticmethod
    def set_scalar(pointer, kind, value):
        C.cast(pointer, C.POINTER(kind))[0] = value
        return 0

    def get_nodes(self, graph, nodes, count):
        if nodes is not None:
            for i, node in enumerate(self.nodes):
                nodes[i] = node
        return self.set_scalar(count, C.c_size_t, len(self.nodes))

    def get_edges(self, graph, sources, targets, metadata, count):
        if sources is None and targets is None:
            assert metadata is None
            return self.set_scalar(count, C.c_size_t, len(self.edge_rows))
        if metadata is None:
            return 402
        assert metadata._type_ is CudaGraphEdgeData
        self.data_fetches += 1
        for i, (source, target, raw) in enumerate(self.edge_rows):
            sources[i], targets[i] = source, target
            C.memmove(C.addressof(metadata[i]), raw, 8)
        return self.set_scalar(count, C.c_size_t, len(self.edge_rows))


class TypedEdgeTests(unittest.TestCase):
    def test_ctypes_abi(self):
        self.assertEqual(edge_abi(), dict(size=8, alignment=1,
            offsets=dict(from_port=0, to_port=1, type=2, reserved=3), reserved_bytes=5))

    def test_compiled_installed_header_abi(self):
        compiler = shutil.which('cc')
        self.assertIsNotNone(compiler, 'A CPU C compiler is required for this ABI regression')
        include = Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/nvidia/cuda_runtime/include')
        compiler_include = include.parents[2] / 'triton/backends/nvidia/include'
        source = Path(__file__).with_name('cuda_graph_edge_abi.c')
        with tempfile.TemporaryDirectory(prefix='ouro-edge-abi-') as directory:
            binary = Path(directory) / 'edge_abi'
            built = subprocess.run([compiler, '-std=c11', '-I', str(include), '-I', str(compiler_include),
                                    str(source), '-o', str(binary)],
                                   capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
            self.assertIn('size=8 alignment=1 offsets=0,1,2,3 reserved=5', result.stdout)

    def test_all_field_bytes_round_trip(self):
        for position in range(8):
            for value in range(256):
                raw = bytearray(8)
                raw[position] = value
                decoded = decode_edge(CudaGraphEdgeData.from_buffer_copy(raw))
                self.assertEqual(bytes.fromhex(decoded['raw_hex']), bytes(raw))
                self.assertEqual(decoded['raw_bytes'], list(raw))
                self.assertEqual(decoded['reserved'], list(raw[3:]))
                self.assertEqual(decoded['is_default'], not any(raw))

    def test_unknown_values_and_reserved_bytes_survive(self):
        raw = bytes([255, 254, 253, 1, 2, 3, 4, 5])
        value = decode_edge(CudaGraphEdgeData.from_buffer_copy(raw))
        self.assertEqual(value['dependency_type'], 'UNKNOWN')
        self.assertEqual(value['source_port'], 'UNKNOWN')
        self.assertTrue(value['reserved_nonzero'])
        self.assertFalse(value['is_default'])
        self.assertEqual(value['raw_bytes'], list(raw))

    def test_strict_query_stays_red_and_typed_query_is_green(self):
        library = FakeLibrary()
        strict = object.__new__(Cudart)
        strict.lib = library
        with self.assertRaisesRegex(RuntimeError, '402'):
            strict.topology(1)
        runtime = ReadOnlyGraphRuntime(library=library)
        result = runtime.snapshot(1)
        self.assertEqual(result['node_count'], 3)
        self.assertEqual(result['edge_count'], 2)
        self.assertEqual(result['nondefault_edge_count'], 2)
        self.assertEqual(result['reserved_nonzero_count'], 1)
        self.assertEqual([bytes(e['raw_bytes']) for e in result['edges']], [e[2] for e in library.edge_rows])
        self.assertEqual(library.data_fetches, 1)

    def test_read_only_bindings(self):
        runtime = ReadOnlyGraphRuntime(library=FakeLibrary())
        self.assertEqual(set(runtime.bound_functions), {'cudaRuntimeGetVersion', 'cudaGetErrorString',
            'cudaGraphGetNodes', 'cudaGraphGetEdges_v2', 'cudaGraphNodeGetType'})
        with self.assertRaisesRegex(RuntimeError, 'read-only'):
            runtime.call('cudaGraphAddDependencies')

    def test_fingerprint_ignores_api_enumeration_order(self):
        library = FakeLibrary()
        runtime = ReadOnlyGraphRuntime(library=library)
        first = runtime.snapshot(1)
        library.nodes.reverse()
        library.edge_rows.reverse()
        second = runtime.snapshot(1)
        self.assertEqual(first['fingerprint'], second['fingerprint'])
        self.assertEqual(first['fingerprint'], snapshot_fingerprint(first))
        source, target, raw = library.edge_rows[0]
        library.edge_rows[0] = (source, target, raw[:-1] + bytes([raw[-1] ^ 1]))
        self.assertNotEqual(first['fingerprint'], runtime.snapshot(1)['fingerprint'])

    def test_nominal_cut_classification_from_typed_records(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / 'outputs/native_graph_dag_base14_P1_v4/rank0_replayed_graph0.dot').read_text()
        _, rows = analyze(text, 24)
        order = [hex(int(row['handle'], 16)) for row in rows]
        edge_rows = []
        for index, (source, target) in enumerate(zip(order, order[1:])):
            raw = bytes([1, 0, 1, 0, 0, 0, 0, 0]) if index == 287 else bytes(8)
            edge_rows.append(dict(api_index=index, source=source, target=target,
                                  **decode_edge(CudaGraphEdgeData.from_buffer_copy(raw))))
        snapshot = dict(nodes=[dict(handle=h, kind=0) for h in order], edges=edge_rows)
        result = inspect_nominal_cuts(snapshot, text, 24)
        self.assertEqual(result['status'], 'VERIFIED_KERNEL_GROUP_CUTS')
        self.assertEqual([c['position'] for c in result['cuts']], [0, 288, 576, 864, 1154])
        self.assertEqual([c['on_nondefault_edge'] for c in result['cuts']], [None, True, False, False, None])
        self.assertEqual(result['overlap_duration'], 'UNKNOWN')
        snapshot['edges'][0]['source'] = '0xdeadbeef'
        self.assertEqual(inspect_nominal_cuts(snapshot, text, 24)['status'], 'DOT_RUNTIME_MISMATCH')

    def test_missing_pattern_does_not_discard_exported_edges(self):
        library = FakeLibrary()
        snapshot = ReadOnlyGraphRuntime(library=library).snapshot(1)
        self.assertEqual(inspect_nominal_cuts(snapshot, 'not a supported DOT graph', 24)['status'],
                         'CUT_ANALYSIS_UNAVAILABLE')
        self.assertEqual(snapshot['edge_count'], 2)


if __name__ == '__main__':
    unittest.main()
