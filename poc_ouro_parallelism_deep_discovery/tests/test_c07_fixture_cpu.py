"""CPU-only fixture contracts. These tests do not establish CUDA parity."""
import ast
from contextlib import contextmanager, nullcontext, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from io import StringIO
import sys
import types
import unittest
from unittest.mock import patch

from poc_ouro_parallelism_deep_discovery.contract import ROOT, MODELS, read
from poc_ouro_parallelism_deep_discovery.c07_fixture import (
    clone_metadata, lease_pages, payload_bytes, require_bitwise, selectors, token_addresses)
from poc_ouro_parallelism_deep_discovery.scripts.c07_one_pass_fixture import (
    engine_config, freeze_request, parse_args)
from poc_ouro_parallelism_deep_discovery import c07_fixture


def worker_method(name):
    tree = ast.parse((ROOT / 'c07_fixture_probe.py').read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(vars(c07_fixture))
    namespace['time'] = types.SimpleNamespace(perf_counter_ns=lambda: 123)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / 'c07_fixture_probe.py'), 'exec'), namespace)
    return namespace[name]


class Words:
    comparisons = 0

    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count

    def __ne__(self, other):
        Words.comparisons += 1
        return False


class ParityTensor:
    def __init__(self, shape=(1, 4), dtype='bfloat16', word_count=None):
        self.shape, self.dtype = shape, dtype
        self.word_count = word_count

    def numel(self):
        import math
        return math.prod(self.shape)

    def contiguous(self):
        return self

    def view(self, dtype):
        return Words(self.numel() if self.word_count is None else self.word_count)

    def detach(self):
        return self

    def clone(self):
        return self

    def copy_(self, other):
        return self


def fake_torch():
    module = types.ModuleType('torch')
    module.bfloat16, module.int16 = 'bfloat16', 'int16'
    module.count_nonzero = lambda value: types.SimpleNamespace(item=lambda: 0)
    module.inference_mode = nullcontext
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 1.0
    module.cuda = types.SimpleNamespace(synchronize=lambda: None, Event=Event)
    return module


def service_fixture(validator):
    events = []
    tensor = ParityTensor()
    instance = types.SimpleNamespace(_c07=dict(phase='PREPARED', fixture_input=tensor, fixture_working=tensor))
    instance._c07_one_pass = lambda hidden: (events.append('pass'), tensor)[1]
    instance._c07_compare = validator
    instance._c07_all_ranks_pass = lambda parity: (events.append(('reduction', parity['status'])),
                                                  parity['status'] == 'BITWISE_PASS')[1]
    distributed = types.ModuleType('vllm.distributed')
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    distributed.get_tp_group = lambda: None
    return instance, events, {'torch': fake_torch(), 'vllm.distributed': distributed}


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.values)


@dataclass
class Metadata:
    table: Tensor
    nested: dict
    opaque: object


class FixtureTests(unittest.TestCase):
    def test_compare_rejects_broadcastable_shape_before_word_equality(self):
        instance = types.SimpleNamespace(_c07=dict(cache_descriptions=[], selected=[], position=1,
            fixture_reference=ParityTensor((1, 4))), get_model=lambda: types.SimpleNamespace(model=types.SimpleNamespace(layers=[])))
        Words.comparisons = 0
        with patch.dict(sys.modules, {'torch': fake_torch()}), self.assertRaisesRegex(ValueError, 'shape'):
            worker_method('_c07_compare')(instance, ParityTensor((4, 1)))
        self.assertEqual(Words.comparisons, 0)

    def test_compare_rejects_non_bf16_before_word_equality(self):
        for actual_dtype, reference_dtype in [('float32', 'float32'), ('bfloat16', 'float32')]:
            instance = types.SimpleNamespace(_c07=dict(cache_descriptions=[], selected=[], position=1,
                fixture_reference=ParityTensor(dtype=reference_dtype)),
                get_model=lambda: types.SimpleNamespace(model=types.SimpleNamespace(layers=[])))
            Words.comparisons = 0
            with patch.dict(sys.modules, {'torch': fake_torch()}), self.assertRaisesRegex(ValueError, 'BF16'):
                worker_method('_c07_compare')(instance, ParityTensor(dtype=actual_dtype))
            self.assertEqual(Words.comparisons, 0)

    def test_compare_rejects_word_count_before_word_equality(self):
        instance = types.SimpleNamespace(_c07=dict(cache_descriptions=[], selected=[], position=1,
            fixture_reference=ParityTensor(word_count=8)),
            get_model=lambda: types.SimpleNamespace(model=types.SimpleNamespace(layers=[])))
        Words.comparisons = 0
        with patch.dict(sys.modules, {'torch': fake_torch()}), self.assertRaisesRegex(ValueError, 'word count'):
            worker_method('_c07_compare')(instance, ParityTensor())
        self.assertEqual(Words.comparisons, 0)

    def test_local_validator_error_still_enters_initial_reduction(self):
        def fail(output):
            raise ValueError('CPU_LOCAL_VALIDATOR_FAILURE')
        instance, events, modules = service_fixture(fail)
        with patch.dict(sys.modules, modules):
            result = worker_method('c07_service')(instance, 1)
        self.assertEqual(result['phase'], 'UNRESOLVED')
        self.assertEqual(events, ['pass', ('reduction', 'UNRESOLVED')])
        self.assertEqual(result['parity']['local_validation_error']['type'], 'ValueError')
        self.assertIn('CPU_LOCAL_VALIDATOR_FAILURE', result['parity']['local_validation_error']['message'])

    def test_trial_error_and_healthy_peer_keep_the_same_reduction_order(self):
        def run(local_error):
            calls = []
            def validate(output):
                calls.append(len(calls))
                if local_error and len(calls) == 2:
                    raise c07_fixture.C07ValidationError('CPU_TRIAL_VALIDATOR_FAILURE')
                return dict(status='BITWISE_PASS', checks=[dict(equal=True)])
            instance, events, modules = service_fixture(validate)
            decisions = iter([True, False])
            def reduction(parity):
                events.append(('reduction', parity['status']))
                return next(decisions)
            instance._c07_all_ranks_pass = reduction
            with patch.dict(sys.modules, modules):
                result = worker_method('c07_service')(instance, 3)
            return result, events
        failed, failure_events = run(True)
        healthy, healthy_events = run(False)
        self.assertEqual(failed['phase'], 'UNRESOLVED')
        self.assertEqual(healthy['phase'], 'UNRESOLVED')
        self.assertEqual([event[0] if isinstance(event, tuple) else event for event in failure_events],
                         ['pass', 'reduction', 'pass', 'reduction'])
        self.assertEqual([event[0] if isinstance(event, tuple) else event for event in healthy_events],
                         ['pass', 'reduction', 'pass', 'reduction'])
        self.assertEqual(failed['service'][0]['ordinal'], 0)
        self.assertEqual(len(failed['service']), 1)
        self.assertEqual(failed['final_parity']['validation_stage'], 'TRIAL.0')
        self.assertEqual(failed['final_parity']['local_validation_error']['type'], 'C07ValidationError')

    def test_final_validator_error_enters_final_reduction(self):
        calls = []
        def validate(output):
            calls.append(len(calls))
            if len(calls) == 3:
                raise AssertionError('CPU_FINAL_VALIDATOR_FAILURE')
            return dict(status='BITWISE_PASS', checks=[dict(equal=True)])
        instance, events, modules = service_fixture(validate)
        with patch.dict(sys.modules, modules):
            result = worker_method('c07_service')(instance, 1)
        self.assertEqual(result['phase'], 'UNRESOLVED')
        self.assertEqual(result['final_parity']['validation_stage'], 'FINAL')
        self.assertEqual(events, ['pass', ('reduction', 'BITWISE_PASS'), 'pass',
                                 ('reduction', 'BITWISE_PASS'), ('reduction', 'UNRESOLVED')])

    def test_device_runtime_error_is_not_local_validation_recovery(self):
        def validate(output):
            raise RuntimeError('CUDA error: CPU_FATAL_DEVICE_FAILURE_STUB')
        instance, events, modules = service_fixture(validate)
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, 'CPU_FATAL_DEVICE_FAILURE_STUB'):
            worker_method('c07_service')(instance, 1)
        self.assertEqual(events, ['pass'])

    def test_matching_bf16_metadata_reaches_word_equality(self):
        Words.comparisons = 0
        report = c07_fixture.compare_bf16_words('payload', ParityTensor(), ParityTensor(), fake_torch())
        self.assertTrue(report['equal'])
        self.assertEqual(report['total_words'], report['reference_words'])
        self.assertEqual(Words.comparisons, 1)

    def test_distinct_ready_rows(self):
        self.assertEqual(selectors(4), [(0, 0), (1, 1), (2, 2), (3, 3)])
        for row in range(4):
            for depth in range(4):
                self.assertEqual(selectors(1, row, depth), [(row, depth)])

    def test_bad_selectors(self):
        for args in [(2, 0, 0), (1, -1, 0), (1, 0, 4), (8, 0, 0)]:
            with self.assertRaises(ValueError):
                selectors(*args)

    def test_page_lease_is_disjoint_and_protects_sources(self):
        for position in (1, 15, 16, 17, 1024, 4096):
            for count in (1, 4):
                protected = set(range(1, 100))
                pages = lease_pages(2000, protected, count, position)
                flat = [page for row in pages for page in row]
                self.assertEqual(len(flat), len(set(flat)))
                self.assertFalse(set(flat) & (protected | {0}))
                self.assertTrue(all(len(row) == (position + 16) // 16 for row in pages))
                addresses = [pair for row in pages for pair in token_addresses(row, position + 1)]
                self.assertEqual(len(addresses), len(set(addresses)))

    def test_insufficient_or_invalid_lease(self):
        for arguments in [(10, {1, 2, 3, 4}, 4, 32), (10, {10}, 1, 1),
                          (10, {-1}, 1, 1), (10, set(), 0, 1)]:
            with self.assertRaises((ValueError, RuntimeError)):
                lease_pages(*arguments)
        with self.assertRaises(ValueError):
            token_addresses([1], 17)

    def test_append_address_crosses_prompt_page_boundary(self):
        pages = lease_pages(100, {1, 2}, 1, 1024)[0]
        addresses = token_addresses(pages, 1025)
        self.assertEqual(addresses[1023], (pages[63], 15))
        self.assertEqual(addresses[1024], (pages[64], 0))
        self.assertNotEqual(addresses[1023][0], addresses[1024][0])

    def test_metadata_clones_tensors_and_retains_native_objects(self):
        tensor, opaque = Tensor([1, 2]), object()
        source = Metadata(tensor, dict(alias=tensor, sequence=[tensor]), opaque)
        clone = clone_metadata(source, lambda value: isinstance(value, Tensor))
        self.assertIsNot(clone, source)
        self.assertIsNot(clone.table, tensor)
        self.assertIs(clone.opaque, opaque)
        self.assertIs(clone.table, clone.nested['alias'])
        self.assertIs(clone.table, clone.nested['sequence'][0])
        tensor.values[0] = 99
        self.assertEqual(clone.table.values, [1, 2])

    def test_payload_is_local_not_cluster_or_full_r4(self):
        for layers in (24, 48):
            tp1 = payload_bytes(layers, 16, 1024, 1)
            tp4 = payload_bytes(layers, 4, 1024, 4)
            self.assertEqual(tp1, tp4)
            self.assertEqual(tp1['history_snapshot'], layers * 16 * 256 * 2 * 1024)
            self.assertEqual(tp1['current_reference'] * 1024, tp1['history_snapshot'])
        with self.assertRaises(ValueError):
            payload_bytes(0, 16, 1, 1)

    def test_raw32gib_pool_has_source_and_destination_page_capacity(self):
        for layers in (24, 48):
            for tp in (1, 4):
                block_bytes = 4 * layers * (16 // tp) * 16 * 256 * 2
                num_blocks = (32 * 1024 ** 3) // block_bytes
                for position in (1024, 4096):
                    source_count = 4 * ((position + 16) // 16)
                    pages = lease_pages(num_blocks, range(1, source_count + 1),
                                        1 if tp == 1 else 4, position)
                    self.assertTrue(all(0 < page < num_blocks for row in pages for page in row))
        # This is raw-layout capacity. The runtime must still check its actual allocation.

    def test_bitwise_status_never_uses_a_tolerance(self):
        self.assertEqual(require_bitwise([dict(equal=True)]), 'BITWISE_PASS')
        self.assertEqual(require_bitwise([dict(equal=True), dict(equal=False)]), 'UNRESOLVED')
        self.assertEqual(require_bitwise([]), 'UNRESOLVED')

    def test_config_retains_r4_and_disables_graphs_and_compilation(self):
        for size, short in [('1.4B', '14'), ('2.6B', '26')]:
            for tp in (1, 4):
                path = ROOT / f'outputs/control_async_base{short}_P{tp}_v1/{size}_R4_TP{tp}'
                source = read(path / 'engine_config.json')
                new = engine_config(source, Path('/fixture'), size, tp)
                self.assertFalse(source['enforce_eager'])
                self.assertTrue(new['enforce_eager'])
                self.assertEqual(new['compilation_config']['mode'], 0)
                self.assertEqual(new['compilation_config']['cudagraph_mode'], 'NONE')
                self.assertEqual(new['hf_overrides']['total_ut_steps'], 4)
                self.assertTrue(new['async_scheduling'])
                self.assertEqual(new['max_num_seqs'], 4)
                self.assertGreaterEqual(new['max_num_batched_tokens'], 4 * 4096)
                self.assertEqual(new['kv_cache_memory_bytes'], 32 * 1024 ** 3)
                self.assertEqual(Path(new['model']), MODELS[size]['path'])

    def test_request_uses_native_allowed_token_not_teacher_hook(self):
        workload = dict(inputs=[dict(id='same_prompt', teacher_ids=[378],
                                    prompt_ids={'1024': [7] * 1024})])
        request = freeze_request(workload, 1024)
        self.assertEqual(len(request['requests']), 4)
        self.assertEqual(request['output_length'], 2)
        self.assertEqual(request['allowed_token_id'], 378)
        workload['inputs'][0]['teacher_ids'][0] = 0
        with self.assertRaises(ValueError):
            freeze_request(workload, 1024)

    def test_config_rejects_a_different_revision_or_loop_count(self):
        path = ROOT / 'outputs/control_async_base14_P1_v1/1.4B_R4_TP1/engine_config.json'
        source = read(path)
        source['hf_overrides']['total_ut_steps'] = 2
        with self.assertRaises(ValueError):
            engine_config(source, Path('/fixture'), '1.4B', 1)
        source = read(path)
        source['model'] = '/different/checkpoint'
        with self.assertRaises(ValueError):
            engine_config(source, Path('/fixture'), '1.4B', 1)

    def test_timed_body_charges_reset_and_collective_parity_precedes_trials(self):
        tree = ast.parse((ROOT / 'c07_fixture_probe.py').read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'c07_service')
        source = ast.unparse(method)
        self.assertLess(source.index('if not self._c07_all_ranks_pass(parity)'), source.index('for ordinal'))
        self.assertLess(source.index('first.record()', source.index('for ordinal')),
                        source.index("hidden.copy_(state['fixture_input'])"))
        self.assertLess(source.index("hidden.copy_(state['fixture_input'])"),
                        source.index('output = self._c07_one_pass(hidden)', source.index('for ordinal')))
        self.assertNotIn('cuda.graph', source)

    def test_cli_does_not_admit_tp2_or_graph_flag(self):
        args = parse_args(['--size', '1.4B', '--tp', '1', '--row', '3', '--depth', '2'])
        self.assertEqual((args.row, args.depth), (3, 2))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parse_args(['--size', '1.4B', '--tp', '2'])

    def test_actual_one_pass_method_keeps_layer_norm_order_and_context(self):
        # Execute the actual fixture method with fake native objects. No Torch import occurs.
        tree = ast.parse((ROOT / 'c07_fixture_probe.py').read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_c07_one_pass')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / 'c07_fixture_probe.py'), 'exec'), namespace)
        events = []
        context = object()
        @contextmanager
        def override(value):
            self.assertIs(value, context)
            events.append('context_enter')
            try:
                yield
            finally:
                events.append('context_exit')
        class Layer:
            def __init__(self, index):
                self.index = index
            def __call__(self, positions, hidden, depth, residual):
                events.append(('layer', self.index, depth, residual))
                return hidden + 1, self.index
        def norm(hidden, residual):
            events.append(('norm', residual))
            return hidden + 10, None
        core = types.SimpleNamespace(layers=[Layer(0), Layer(1)], start_layer=0, end_layer=2, norm=norm)
        instance = types.SimpleNamespace(_c07=dict(fixture_context=context, fixture_positions=[1024]),
                                         get_model=lambda: types.SimpleNamespace(model=core))
        fake = types.ModuleType('vllm.forward_context')
        fake.override_forward_context = override
        with patch.dict(sys.modules, {'vllm.forward_context': fake}):
            result = namespace['_c07_one_pass'](instance, 0)
        self.assertEqual(result, 12)
        self.assertEqual(events, ['context_enter', ('layer', 0, 0, None), ('layer', 1, 0, 0),
                                  ('norm', 1), 'context_exit'])


if __name__ == '__main__':
    unittest.main()
