"""CPU checks for a schedule observer. No native runtime or CUDA import."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def observer_class():
    tree = ast.parse((ROOT / 'cohort_probe.py').read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    definition.bases = [ast.Name(id='object', ctx=ast.Load())]
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace = {}
    exec(compile(module, 'cpu_observer_fixture', 'exec'), namespace)
    return namespace['CohortProbe']


class CohortTests(unittest.TestCase):
    def fixture(self):
        probe = observer_class()()
        self.calls = []
        result = (SimpleNamespace(shape=(3, 1), numel=lambda: 3), 'PROCESSED')
        def execute(schedule, *args, **kwargs):
            self.calls.append(('EXECUTE', schedule, args, kwargs))
            return 'FORWARD_RESULT'
        def sample(*args, **kwargs):
            self.calls.append(('SAMPLE', args, kwargs))
            return result
        probe.model_runner = SimpleNamespace(execute_model=execute,
                                              sampler=SimpleNamespace(sample=sample))
        return probe, execute, sample, result

    def test_original_calls_and_results(self):
        probe, execute, sample, result = self.fixture()
        probe.cohort_trace_start()
        schedule = SimpleNamespace(num_scheduled_tokens={'a': 2, 'b': 1})
        self.assertEqual(probe.model_runner.execute_model(schedule, 5, flag=True), 'FORWARD_RESULT')
        self.assertIs(probe.model_runner.sampler.sample(7, flag=False), result)
        trace = probe.cohort_trace_finish()
        self.assertEqual(self.calls, [('EXECUTE', schedule, (5,), {'flag': True}),
                                     ('SAMPLE', (7,), {'flag': False})])
        self.assertIs(probe.model_runner.execute_model, execute)
        self.assertIs(probe.model_runner.sampler.sample, sample)
        self.assertEqual(trace['events'][0]['scheduled_tokens_by_request'], {'a': 2, 'b': 1})
        self.assertEqual(trace['events'][0]['total_scheduled_tokens'], 3)
        self.assertEqual(trace['events'][1]['sampled_tensor_numel'], 3)

    def test_trace_copies_schedule_metadata(self):
        probe, *_ = self.fixture()
        probe.cohort_trace_start()
        schedule = SimpleNamespace(num_scheduled_tokens={'a': 1024})
        probe.model_runner.execute_model(schedule)
        schedule.num_scheduled_tokens['a'] = 1
        self.assertEqual(probe.cohort_trace_finish()['events'][0]['scheduled_tokens_by_request'], {'a': 1024})

    def test_double_start_and_unarmed_finish_reject(self):
        probe, *_ = self.fixture()
        with self.assertRaises(RuntimeError):
            probe.cohort_trace_finish()
        probe.cohort_trace_start()
        with self.assertRaises(RuntimeError):
            probe.cohort_trace_start()
        probe.cohort_trace_finish()

    def test_fresh_trace_is_independent(self):
        probe, *_ = self.fixture()
        for _ in range(2):
            probe.cohort_trace_start()
            probe.model_runner.sampler.sample()
            self.assertEqual(len(probe.cohort_trace_finish()['events']), 1)

    def test_exceptions_propagate(self):
        probe, *_ = self.fixture()
        def error(*args, **kwargs):
            raise LookupError('ORIGINAL_FAILURE')
        probe.model_runner.execute_model = error
        probe.cohort_trace_start()
        with self.assertRaisesRegex(LookupError, 'ORIGINAL_FAILURE'):
            probe.model_runner.execute_model(SimpleNamespace(num_scheduled_tokens={}))
        probe.cohort_trace_finish()
        self.assertIs(probe.model_runner.execute_model, error)

    def test_frozen_capacity(self):
        protocol = json.loads((ROOT / 'artifacts/native_cohort_protocol_v2.json').read_text())
        required = 2 * 48 * 4 * (4096 + 128) * 16 * 128 * 2 * 8
        self.assertEqual(required / 1024 ** 3, 49.5)
        self.assertGreater(protocol['kv_cache_memory_bytes_per_rank'], required)
        self.assertEqual(protocol['cohort_sizes'], [1, 2, 4, 8])

    def test_no_cuda_reads_or_custom_teacher(self):
        text = (ROOT / 'cohort_probe.py').read_text()
        self.assertNotIn('.cpu(', text)
        self.assertNotIn('.item(', text)
        self.assertNotIn('synchronize', text)
        runner = (ROOT / 'scripts/native_cohort_control.py').read_text()
        self.assertNotIn("collective_rpc('teacher'", runner)
        self.assertIn('allowed_token_ids=[token]', runner)


if __name__ == '__main__':
    unittest.main()
