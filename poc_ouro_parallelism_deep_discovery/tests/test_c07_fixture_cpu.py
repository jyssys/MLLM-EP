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

from poc_ouro_parallelism_deep_discovery.contract import ROOT, MODELS, NATIVE_PYTHON, read
from poc_ouro_parallelism_deep_discovery.c07_fixture import (
    clone_metadata, lease_pages, observe_dense_padding, payload_bytes, require_bitwise,
    selectors, select_decode_metadata, select_v2_padding, snapshot_v2_batch,
    sealed_reference_cohort, token_addresses, v2_batch_rejection, V2CaptureHooks)
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

    def index_select(self, dimension, rows):
        if dimension != 0:
            raise ValueError('CPU fixture only supports row selection.')
        return Tensor([self.values[row] for row in rows])


@dataclass
class Metadata:
    table: Tensor
    nested: dict
    opaque: object


def v2_batch(position=1024):
    return types.SimpleNamespace(req_ids=['r3', 'r0', 'r2', 'r1'], num_reqs=4,
        num_reqs_after_padding=4, num_tokens=4, num_tokens_after_padding=4, num_draft_tokens=0,
        idx_mapping_np=[3, 0, 2, 1], num_scheduled_tokens=[1] * 4,
        query_start_loc_np=[0, 1, 2, 3, 4], num_computed_tokens_np=[position] * 4,
        prefill_len_np=[position] * 4, num_computed_prefill_tokens_np=[position] * 4,
        is_prefilling_np=[False] * 4, is_padding=object())


class SealedLLM:
    def __init__(self, failure=None, paused=False, unfinished=False):
        self.calls, self.paused, self.failure = [], paused, failure
        self.unfinished = unfinished
        self.llm_engine = types.SimpleNamespace(engine_core=self,
            vllm_config=types.SimpleNamespace(parallel_config=types.SimpleNamespace(
                data_parallel_size=1, pipeline_parallel_size=1)),
                                               has_unfinished_requests=lambda: self.unfinished)

    def call_utility(self, name, *args):
        self.calls.append((name, args))
        if name == 'is_scheduler_paused':
            return self.paused
        if name == 'pause_scheduler':
            if args != ('keep', False):
                raise AssertionError('The source fixture must not abort or clear caches.')
            self.paused = True
            if self.failure == 'pause':
                raise ValueError('CPU_PAUSE_FAILURE')
        elif name == 'resume_scheduler':
            if self.failure in ('resume', 'enqueue_and_resume'):
                raise ValueError('CPU_RESUME_FAILURE')
            self.paused = False
        else:
            raise AssertionError(name)

    def enqueue(self, prompts, params, use_tqdm):
        self.calls.append(('enqueue', len(prompts), params, use_tqdm))
        if not self.paused:
            raise AssertionError('Native requests entered before the pause.')
        if self.failure in ('enqueue', 'enqueue_and_resume'):
            raise ValueError('CPU_ENQUEUE_FAILURE')
        return ['r0', 'r1', 'r2'] if self.failure == 'request_count' else ['r0', 'r1', 'r2', 'r3']

    def wait_for_completion(self, use_tqdm):
        self.calls.append(('wait', use_tqdm))
        if self.paused:
            raise AssertionError('Waiting while paused would deadlock.')
        if self.failure == 'wait':
            raise ValueError('CPU_WAIT_FAILURE')
        return ['out0', 'out1', 'out2', 'out3']


class FixtureTests(unittest.TestCase):
    def test_native_sealed_cohort_orders_pause_enqueue_fence_resume_wait(self):
        llm, evidence, params = SealedLLM(), {}, object()
        outputs = sealed_reference_cohort(llm, [object()] * 4, params, evidence)
        self.assertEqual(outputs, ['out0', 'out1', 'out2', 'out3'])
        self.assertEqual(llm.calls, [('is_scheduler_paused', ()), ('pause_scheduler', ('keep', False)),
            ('is_scheduler_paused', ()), ('enqueue', 4, params, False), ('is_scheduler_paused', ()),
            ('resume_scheduler', ()), ('is_scheduler_paused', ()), ('wait', False)])
        self.assertEqual(evidence['status'], 'COMPLETE')
        self.assertEqual([row['event'] for row in evidence['events']], ['INITIAL_UNPAUSED', 'PAUSE_ACK',
                         'FOUR_ENQUEUED', 'QUEUE_FENCE_ACK', 'RESUME_ACK', 'NATIVE_WAIT_START', 'NATIVE_WAIT_RETURNED'])

    def test_native_seal_resumes_in_finally_after_pause_or_enqueue_failure(self):
        for failure in ('pause', 'enqueue', 'request_count'):
            llm, evidence = SealedLLM(failure=failure), {}
            with self.assertRaises((ValueError, c07_fixture.C07ValidationError)):
                sealed_reference_cohort(llm, [object()] * 4, object(), evidence)
            self.assertFalse(llm.paused)
            self.assertEqual(sum(call[0] == 'resume_scheduler' for call in llm.calls), 1)
            self.assertFalse(any(call[0] == 'wait' for call in llm.calls))
            self.assertEqual(evidence['status'], 'FAILED')

    def test_native_seal_does_not_resume_a_preexisting_pause(self):
        llm, evidence = SealedLLM(paused=True), {}
        with self.assertRaisesRegex(ValueError, 'already paused'):
            sealed_reference_cohort(llm, [object()] * 4, object(), evidence)
        self.assertEqual(llm.calls, [('is_scheduler_paused', ())])
        self.assertTrue(llm.paused)

    def test_native_seal_rejects_prior_work_before_any_control_call(self):
        llm, evidence = SealedLLM(unfinished=True), {}
        with self.assertRaisesRegex(ValueError, 'unfinished request'):
            sealed_reference_cohort(llm, [object()] * 4, object(), evidence)
        self.assertEqual(llm.calls, [])

    def test_native_seal_rejects_multiple_engine_ownership_before_control(self):
        llm, evidence = SealedLLM(), {}
        llm.llm_engine.vllm_config.parallel_config.data_parallel_size = 2
        with self.assertRaisesRegex(ValueError, 'DP1 and PP1'):
            sealed_reference_cohort(llm, [object()] * 4, object(), evidence)
        self.assertEqual(llm.calls, [])

    def test_native_seal_wait_failure_occurs_only_after_resume(self):
        llm, evidence = SealedLLM(failure='wait'), {}
        with self.assertRaisesRegex(ValueError, 'CPU_WAIT_FAILURE'):
            sealed_reference_cohort(llm, [object()] * 4, object(), evidence)
        self.assertFalse(llm.paused)
        self.assertEqual(sum(call[0] == 'resume_scheduler' for call in llm.calls), 1)
        self.assertEqual(evidence['status'], 'FAILED')

    def test_native_seal_keeps_primary_and_resume_errors(self):
        llm, evidence = SealedLLM(failure='enqueue_and_resume'), {}
        with self.assertRaisesRegex(ValueError, 'CPU_ENQUEUE_FAILURE'):
            sealed_reference_cohort(llm, [object()] * 4, object(), evidence)
        self.assertEqual(evidence['restore_error']['message'], 'CPU_RESUME_FAILURE')
        self.assertEqual(evidence['error']['message'], 'CPU_ENQUEUE_FAILURE')
        self.assertFalse(any(call[0] == 'wait' for call in llm.calls))

    def test_runner_uses_sealed_source_and_retains_the_control_ledger_on_error(self):
        tree = ast.parse((ROOT / 'scripts/c07_one_pass_fixture.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id == 'sealed_reference_cohort'
                            for node in calls))
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'generate' for node in calls))
        self.assertTrue(any(isinstance(node, ast.Try) and node.finalbody
                            and 'source_cohort_seal.json' in ast.unparse(node.finalbody) for node in ast.walk(tree)))

    def test_rejected_forward_retains_owned_cpu_batch_counts(self):
        tree = ast.parse((ROOT / 'c07_fixture_probe.py').read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        start = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'c07_capture_start')
        observed = next(node for node in start.body if isinstance(node, ast.FunctionDef) and node.name == 'observed')
        batch = snapshot_v2_batch(v2_batch())
        batch['num_tokens_after_padding'] = 8
        state = dict(pending_batch=batch, pending_padding=object(), calls_seen=0,
                     phase='ARMED', rejected=[], original=lambda **kwargs: 'native_result')
        instance = types.SimpleNamespace(_c07_suitability=lambda *args: 'NOT_FOUR_UNPADDED_V2_ROWS')
        namespace = dict(vars(c07_fixture), state=state, self=instance)
        exec(compile(ast.Module(body=[observed], type_ignores=[]), str(ROOT / 'c07_fixture_probe.py'), 'exec'), namespace)
        context = types.SimpleNamespace(attn_metadata={})
        forward_context = types.ModuleType('vllm.forward_context')
        forward_context.get_forward_context = lambda: context
        with patch.dict(sys.modules, {'vllm.forward_context': forward_context}):
            self.assertEqual(namespace['observed'](positions=object()), 'native_result')
        self.assertEqual(state['rejected'][0]['cpu_batch']['num_tokens_after_padding'], 8)
        batch['num_tokens_after_padding'] = 99
        self.assertEqual(state['rejected'][0]['cpu_batch']['num_tokens_after_padding'], 8)

    def test_v2_hooks_observe_real_batch_then_forward_and_restore_class_methods(self):
        events = []
        batch = v2_batch()
        class Model:
            def forward(self, **kwargs):
                events.append(('original_forward', kwargs))
                return 'unchanged'
        class Runner:
            model = Model()
            def prepare_inputs(self, marker):
                events.append(('original_prepare', marker))
                return batch
        runner = Runner()
        original_forward, original_prepare = runner.model.forward, runner.prepare_inputs
        def prepared(actual):
            self.assertIs(actual, batch)
            events.append('observed_batch')
        def observed(**kwargs):
            events.append('observed_forward')
            return lease.original_forward(**kwargs)
        lease = V2CaptureHooks(runner, prepared, observed)
        lease.install()
        self.assertIs(runner.prepare_inputs('native_scheduler'), batch)
        self.assertEqual(runner.model.forward(positions='actual_positions'), 'unchanged')
        lease.restore()
        lease.restore()
        self.assertEqual(runner.prepare_inputs, original_prepare)
        self.assertEqual(runner.model.forward, original_forward)
        self.assertNotIn('prepare_inputs', vars(runner))
        self.assertNotIn('forward', vars(runner.model))
        self.assertEqual(events, [('original_prepare', 'native_scheduler'), 'observed_batch',
                                  'observed_forward', ('original_forward', {'positions': 'actual_positions'})])

    def test_v2_hook_restore_keeps_prior_instance_forward_after_failure(self):
        batch = v2_batch()
        def fail(**kwargs):
            raise ValueError('CPU_V2_FORWARD_FAILURE')
        model = types.SimpleNamespace(forward=fail)
        runner = types.SimpleNamespace(model=model, prepare_inputs=lambda *_: batch)
        before_runner, before_model = dict(vars(runner)), dict(vars(model))
        def observed(**kwargs):
            return lease.original_forward(**kwargs)
        lease = V2CaptureHooks(runner, lambda _: None, observed)
        lease.install()
        with self.assertRaisesRegex(ValueError, 'CPU_V2_FORWARD_FAILURE'):
            runner.model.forward()
        lease.restore()
        self.assertEqual(vars(runner), before_runner)
        self.assertEqual(vars(model), before_model)

    def test_v2_hooks_reject_duplicate_install(self):
        runner = types.SimpleNamespace(model=types.SimpleNamespace(forward=lambda: None),
                                       prepare_inputs=lambda: v2_batch())
        lease = V2CaptureHooks(runner, lambda _: None, lambda: None)
        lease.install()
        try:
            with self.assertRaisesRegex(ValueError, 'already active'):
                lease.install()
        finally:
            lease.restore()

    def test_v2_cpu_snapshot_is_owned_and_keeps_actual_native_row_order(self):
        source = v2_batch()
        snapshot = snapshot_v2_batch(source)
        self.assertIsNone(v2_batch_rejection(snapshot, 1024))
        source.req_ids[0] = 'later_request'
        source.prefill_len_np[0] = 99
        self.assertEqual(snapshot['request_ids'], ['r3', 'r0', 'r2', 'r1'])
        self.assertEqual(snapshot['prefill_len_np'], [1024] * 4)

    def test_v2_admission_rejects_padding_mixed_work_or_computed_progress(self):
        for name, value in [('num_reqs_after_padding', 8), ('num_tokens_after_padding', 8),
                            ('num_draft_tokens', 1), ('num_scheduled_tokens', [1, 1, 1, 2]),
                            ('num_computed_tokens_np', [1024, 1024, 1025, 1024]),
                            ('prefill_len_np', [1024, 1023, 1024, 1024]),
                            ('is_prefilling_np', [False, False, True, False])]:
            source = v2_batch()
            setattr(source, name, value)
            self.assertIsNotNone(v2_batch_rejection(snapshot_v2_batch(source), 1024), name)

    def test_real_dense_padding_flags_are_preserved_not_synthesized(self):
        self.assertEqual(observe_dense_padding([False] * 4)['interpretation'], 'ALL_FALSE')
        flags = [True, False, True, True]
        observation = observe_dense_padding(flags)
        self.assertEqual(observation['interpretation'], 'DENSE_UNUSED_TRUE_FLAGS')
        self.assertEqual(observation['flags'], flags)
        flags[0] = False
        self.assertEqual(observation['flags'], [True, False, True, True])
        with self.assertRaises(ValueError):
            observe_dense_padding([False] * 3)

    def test_selected_padding_keeps_native_values_in_private_storage(self):
        source = Tensor([True, False, True, False])
        tp1 = select_v2_padding(source, [2])
        tp4 = select_v2_padding(source, [0, 1, 2, 3])
        self.assertEqual(tp1.values, [True])
        self.assertEqual(tp4.values, [True, False, True, False])
        source.values[2] = False
        self.assertEqual(tp1.values, [True])
        self.assertEqual(tp4.values, [True, False, True, False])

    def test_decode_metadata_uses_selected_counts_and_zero_prefill_counts(self):
        @dataclass
        class DecodeMetadata:
            num_actual_tokens: int = 4
            max_query_len: int = 1
            query_start_loc: object = None
            seq_lens: object = None
            block_table: object = None
            slot_mapping: object = None
            num_decode_reqs: int = 4
            num_decode_tokens: int = 4
            num_prefill_reqs: int = 0
            num_prefill_tokens: int = 0
            scheduler_metadata: object = None
            max_num_splits: int = 0
        schedule = object()
        original = DecodeMetadata(scheduler_metadata=schedule)
        for rows in (1, 4):
            selected = select_decode_metadata(original, num_rows=rows,
                query_start_loc=list(range(rows + 1)), seq_lens=[1025] * rows,
                block_table=list(range(rows)), slot_mapping=list(range(rows)))
            self.assertEqual(selected.num_actual_tokens, rows)
            self.assertEqual(selected.num_decode_reqs, rows)
            self.assertEqual(selected.num_decode_tokens, rows)
            self.assertEqual(selected.num_prefill_reqs, 0)
            self.assertEqual(selected.num_prefill_tokens, 0)
            self.assertIs(selected.scheduler_metadata, schedule)
            self.assertEqual(original.num_decode_reqs, 4)
        tree = ast.parse((ROOT / 'c07_fixture_probe.py').read_text())
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == 'select_decode_metadata' for node in ast.walk(tree)))
    def test_worker_selects_real_padding_and_uses_v2_hooks_without_v1_fallback(self):
        source = (ROOT / 'c07_fixture_probe.py').read_text()
        self.assertNotIn('runner._model_forward', source)
        self.assertNotIn('runner.input_batch', source)
        self.assertIn('V2CaptureHooks(runner, prepared, observed)', source)
        self.assertIn("is_padding=clone_metadata(context.is_padding, torch.is_tensor, memo)", source)
        self.assertIn("is_padding=select_v2_padding(state['context'].is_padding, selected_rows)", source)

    def test_pinned_dense_route_has_no_padding_reader(self):
        package = Path(NATIVE_PYTHON).parents[1] / 'lib/python3.14/site-packages/vllm'
        for relative in ['model_executor/models/ouro.py', 'model_executor/layers/linear.py',
                         'model_executor/layers/utils.py', 'model_executor/layers/layernorm.py',
                         'model_executor/layers/activation.py', 'model_executor/layers/attention/attention.py',
                         'model_executor/layers/rotary_embedding/base.py',
                         'model_executor/layers/vocab_parallel_embedding.py',
                         'v1/attention/backends/flash_attn.py']:
            tree = ast.parse((package / relative).read_text())
            self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr == 'is_padding'
                                 for node in ast.walk(tree)), relative)
        consumer = ast.parse((package / 'model_executor/layers/fused_moe/modular_kernel.py').read_text())
        self.assertTrue(any(isinstance(node, ast.Attribute) and node.attr == 'is_padding'
                            for node in ast.walk(consumer)))

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
