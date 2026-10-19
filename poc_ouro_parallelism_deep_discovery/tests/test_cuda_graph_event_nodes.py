"""CPU graph and fake-ABI tests. These are not CUDA timing results."""
import ctypes as C
from pathlib import Path
import unittest
from unittest.mock import patch

from poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes import (
    EventGraph, chain_order, cut_plan, loaded_cudart_path, validate_event_insertion, validate_native_dot)
from poc_ouro_parallelism_deep_discovery.scripts.parse_native_graph_dot import analyze


class FakeRuntime:
    def __init__(self):
        self.nodes, self.edges = {10: 0, 20: 0, 30: 1}, {(10, 20), (20, 30)}
        self.events, self.next_handle = set(), 100

    def topology(self, graph):
        return list(self.nodes), list(self.edges), dict(self.nodes)

    def allocate(self, output):
        self.next_handle += 1
        C.cast(output, C.POINTER(C.c_void_p))[0] = self.next_handle
        return self.next_handle

    def call(self, name, *args):
        if name == 'cudaEventCreateWithFlags':
            assert args[1] == 0
            self.events.add(self.allocate(args[0]))
        elif name == 'cudaGraphAddEventRecordNode':
            node = self.allocate(args[0])
            self.nodes[node] = 7
            if args[3]:
                self.edges.add((args[2][0], node))
        elif name == 'cudaGraphAddDependencies':
            self.edges.add((args[1][0], args[2][0]))
        elif name == 'cudaGraphDestroyNode':
            node = args[0]
            del self.nodes[node]
            self.edges = {(a, b) for a, b in self.edges if a != node and b != node}
        elif name == 'cudaEventDestroy':
            self.events.remove(args[0].value)
        elif name == 'cudaEventElapsedTime':
            C.cast(args[0], C.POINTER(C.c_float))[0] = 1.0
        elif name not in ('cudaDeviceSynchronize', 'cudaStreamSynchronize'):
            raise AssertionError(name)


class FakeGraph:
    def __init__(self, fail_first_instantiate=False):
        self.replays, self.instantiations = 0, 0
        self.fail_first_instantiate = fail_first_instantiate

    def raw_cuda_graph(self):
        return 1

    def instantiate(self):
        self.instantiations += 1
        if self.instantiations == 1 and self.fail_first_instantiate:
            raise RuntimeError('Synthetic instantiate failure')

    def replay(self):
        self.replays += 1


class EventNodeTests(unittest.TestCase):
    def test_unordered_enumeration(self):
        self.assertEqual(chain_order([30, 10, 20], [(20, 30), (10, 20)]), [10, 20, 30])

    def test_reject_branch(self):
        with self.assertRaises(ValueError):
            chain_order([1, 2, 3], [(1, 2), (1, 3)])

    def test_reject_cycle_component(self):
        with self.assertRaises(ValueError):
            chain_order([1, 2, 3], [(2, 3), (3, 2)])

    def test_reject_unknown_node(self):
        with self.assertRaises(ValueError):
            chain_order([1, 2], [(1, 9)])

    def test_reject_unsupported_kind(self):
        with self.assertRaises(ValueError):
            chain_order([1, 2], [(1, 2)], {1: 0, 2: 7})

    def test_cut_validation(self):
        for cuts in ([1, 3], [0, 1, 1, 3], [0, 4, 3], [0, 1.0, 3]):
            with self.subTest(cuts=cuts), self.assertRaises(ValueError):
                cut_plan([10, 20, 30], cuts)
        self.assertEqual(cut_plan([10, 20, 30], [0, 1, 3]),
                         [(0, None, 10), (1, 10, 20), (3, 30, None)])

    def test_fake_api_restores_topology_and_replays_through_torch_callback(self):
        runtime, graph = FakeRuntime(), FakeGraph()
        baseline = runtime.topology(1)
        timer = EventGraph(runtime, graph, [0, 1, 3])
        self.assertTrue(set(baseline[1]) <= runtime.edges)
        for _ in range(3):
            timer.launch(77)
        record = timer.last_replay()
        self.assertEqual(graph.replays, 3)
        self.assertEqual(record['observed_replays'], 3)
        self.assertEqual(record['timed_replay_ordinal'], 3)
        self.assertIsNone(record['request_total_ms'])
        self.assertEqual(record['request_extrapolation'], 'FORBIDDEN')
        timer.close()
        self.assertEqual(runtime.topology(1), baseline)
        self.assertEqual(runtime.events, set())
        self.assertEqual(graph.instantiations, 2)
        timer.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            timer.launch(77)

    def test_rollback_after_instantiation_failure(self):
        runtime, graph = FakeRuntime(), FakeGraph(fail_first_instantiate=True)
        baseline = runtime.topology(1)
        with self.assertRaisesRegex(RuntimeError, 'Synthetic'):
            EventGraph(runtime, graph, [0, 1, 3])
        self.assertEqual(runtime.topology(1), baseline)
        self.assertEqual(runtime.events, set())

    def test_source_graph_change_rejected(self):
        runtime, graph = FakeRuntime(), FakeGraph()
        timer = EventGraph(runtime, graph, [0, 1, 3])
        graph.raw_cuda_graph = lambda: 2
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            timer.launch(77)
        graph.raw_cuda_graph = lambda: 1
        timer.close()

    def test_runtime_path_requires_one_loaded_library(self):
        line = '1000-2000 r-xp 0 00:00 1 /test/libcudart.so.12'
        self.assertEqual(loaded_cudart_path(line + '\n' + line), '/test/libcudart.so.12')
        for text in ('', line + '\n' + line.replace('/test/', '/other/')):
            with self.assertRaises(RuntimeError):
                loaded_cudart_path(text)

    def test_live_multiple_runtimes_need_torch_symbol_owner(self):
        maps = ('1000-2000 r-xp 0 00:00 1 /test/libcudart.so.12\n'
                '2000-3000 r-xp 0 00:00 2 /other/libcudart.so.12.8.90\n')
        module = 'poc_ouro_parallelism_deep_discovery.cuda_graph_event_nodes'
        with patch('pathlib.Path.read_text', return_value=maps):
            with patch(module+'.torch_linked_runtime_path', return_value='/test/libcudart.so.12'):
                self.assertEqual(loaded_cudart_path(), '/test/libcudart.so.12')
            with patch(module+'.torch_linked_runtime_path', return_value='/unmapped/libcudart.so.12'):
                with self.assertRaisesRegex(RuntimeError, 'absent'):
                    loaded_cudart_path()

    def test_deleted_runtime_is_not_selected(self):
        maps = ('1000-2000 r-xp 0 00:00 1 /test/libcudart.so.12\n'
                '2000-3000 r-xp 0 00:00 2 /other/libcudart.so.12 (deleted)\n')
        with patch('pathlib.Path.read_text', return_value=maps):
            with self.assertRaisesRegex(RuntimeError, 'already-loaded'):
                loaded_cudart_path()

    def test_actual_p1_dot_and_added_event_order(self):
        root = Path(__file__).resolve().parents[1]
        dot = root / 'outputs/native_graph_dag_base14_P1_v4/rank0_replayed_graph0.dot'
        summary, rows = analyze(dot.read_text(), 24)
        self.assertEqual(summary['node_count'], 1154)
        self.assertEqual(summary['node_kinds'], {'KERNEL': 1153, 'MEMCPY': 1})
        self.assertEqual(summary['logical_pass_kernel_cuts'], [0, 288, 576, 864, 1154])
        self.assertTrue(summary['single_chain'])
        for cut in summary['logical_pass_kernel_cuts'][:-1]:
            self.assertIn('rms_norm', rows[cut]['kernel'])
        order = [int(row['handle'], 16) for row in rows]
        checked, _ = validate_native_dot(dot.read_text(), 24, order, 1154)
        self.assertEqual(checked['logical_pass_kernel_cuts'], [0, 288, 576, 864, 1154])
        edges = list(zip(order, order[1:]))
        plan = cut_plan(order, summary['logical_pass_kernel_cuts'])
        events = [max(order) + i + 1 for i in range(len(plan))]
        changed_edges = list(edges)
        for (_, previous, following), event in zip(plan, events):
            if previous is not None:
                changed_edges.append((previous, event))
            if following is not None:
                changed_edges.append((event, following))
        kinds = {int(row['handle'], 16): 0 if row['kind'] == 'KERNEL' else 1 for row in rows}
        kinds.update({event: 7 for event in events})
        validate_event_insertion(order, edges, plan, events, order + events, changed_edges, kinds)
        self.assertEqual(len(order + events), 1159)
        self.assertEqual(len(changed_edges), 1161)
        with self.assertRaises(ValueError):
            validate_event_insertion(order, edges, plan, events, order + events, changed_edges[1:], kinds)

    def test_native_dot_rejects_stale_handle_order(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / 'outputs/native_graph_dag_base14_P1_v4/rank0_replayed_graph0.dot').read_text()
        _, rows = analyze(text, 24)
        order = [int(row['handle'], 16) for row in rows]
        with self.assertRaisesRegex(RuntimeError, 'live graph'):
            validate_native_dot(text, 24, list(reversed(order)), 1154)

    def test_native_dot_rejects_missing_repeated_pattern(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / 'outputs/native_graph_dag_base14_P1_v4/rank0_replayed_graph0.dot').read_text()
        _, rows = analyze(text, 24)
        order = [int(row['handle'], 16) for row in rows]
        with self.assertRaisesRegex(RuntimeError, 'live graph'):
            validate_native_dot(text.replace('reshape_and_cache', 'unrecognized_cache'), 24, order, 1154)


if __name__ == '__main__':
    unittest.main()
