"""CPU reproduction of the legacy diagnostic's missing teacher_index field."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from poc_ouro_parallelism_deep_discovery.contract import PROJECT, ROOT
from poc_ouro_parallelism_deep_discovery.long_context_diagnostic import finish_native_allowed_capture as diagnostic_call


class CPUValue:
    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return [1]

    def tolist(self):
        return [1] * 4096


def legacy_finish_fixture():
    path = PROJECT / 'poc_looped_recurrence_tp_4models/worker_probe.py'
    tree = ast.parse(path.read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'FourModelProbe')
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'finish_parity_capture')
    namespace = dict(Path=Path, os=types.SimpleNamespace(environ={'LOOPED_KV_OUTPUT': '/cpu-fixture'}),
                     sha=lambda path: 'cpu-stub-no-file-read')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    model = types.SimpleNamespace(compute_logits=object())
    worker = types.SimpleNamespace(original_compute_logits=object(), captured_logits=[CPUValue()],
        captured_hidden=[CPUValue()], get_model=lambda: model,
        model_runner=types.SimpleNamespace(block_tables=types.SimpleNamespace(input_block_tables=[[CPUValue()]])),
        vllm_config=types.SimpleNamespace(compilation_config=types.SimpleNamespace(static_forward_context={})))
    torch = types.ModuleType('torch')
    torch.Tensor = CPUValue
    torch.cat = lambda values: CPUValue()
    torch.cuda = types.SimpleNamespace(synchronize=lambda: None)
    numpy = types.ModuleType('numpy')
    numpy.stack = lambda values: values
    numpy.savez_compressed = lambda *args, **kwargs: None
    distributed = types.ModuleType('vllm.distributed')
    distributed.get_tensor_model_parallel_rank = lambda: 0
    modules = {'torch': torch, 'numpy': numpy, 'vllm.distributed': distributed}
    return worker, lambda label, prefix: namespace['finish_parity_capture'](worker, label, prefix), modules


class DiagnosticTests(unittest.TestCase):
    def test_native_allowed_token_diagnostic_does_not_require_teacher_hook(self):
        worker, finish, modules = legacy_finish_fixture()
        sampler = types.SimpleNamespace(sample=object())
        sample = sampler.sample
        worker.model_runner.sampler = sampler
        self.assertFalse(hasattr(worker, 'teacher_index'))
        with patch.dict(sys.modules, modules):
            result = diagnostic_call(worker, finish, 'long_context_numerical', 32768)
        self.assertEqual(result['rank'], 0)
        self.assertEqual(result['teacher_steps'], 'NOT_APPLICABLE')
        self.assertEqual(result['sampling_control'], 'NATIVE_ALLOWED_TOKEN_IDS')
        self.assertEqual(result['teacher_index_compatibility'], 'TEMPORARY_EXPORT_ONLY')
        self.assertFalse(hasattr(worker, 'teacher_index'))
        self.assertIs(worker.model_runner.sampler, sampler)
        self.assertIs(worker.model_runner.sampler.sample, sample)
        self.assertFalse(hasattr(worker, 'original_sampler_sample'))

    def test_unadapted_legacy_export_reproduces_exact_missing_field(self):
        worker, finish, modules = legacy_finish_fixture()
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(AttributeError, 'teacher_index'):
            finish('long_context_numerical', 32768)

    def test_export_failure_removes_temporary_field(self):
        worker = types.SimpleNamespace()
        def fail(label, prefix):
            self.assertEqual(worker.teacher_index, 0)
            raise ValueError('CPU_EXPORT_FAILURE')
        with self.assertRaisesRegex(ValueError, 'CPU_EXPORT_FAILURE'):
            diagnostic_call(worker, fail, 'capture', 16)
        self.assertFalse(hasattr(worker, 'teacher_index'))

    def test_preexisting_metadata_field_is_restored(self):
        worker = types.SimpleNamespace(teacher_index=17)
        def finish(label, prefix):
            self.assertEqual(worker.teacher_index, 0)
            return dict(teacher_steps=worker.teacher_index)
        result = diagnostic_call(worker, finish, 'capture', 16)
        self.assertEqual(worker.teacher_index, 17)
        self.assertEqual(result['teacher_steps'], 'NOT_APPLICABLE')

    def test_inherited_metadata_field_is_not_shadowed_after_export(self):
        class Worker:
            teacher_index = 23
        worker = Worker()
        result = diagnostic_call(worker, lambda *_: dict(teacher_steps=worker.teacher_index), 'capture', 16)
        self.assertEqual(result['teacher_steps'], 'NOT_APPLICABLE')
        self.assertEqual(worker.teacher_index, 23)
        self.assertNotIn('teacher_index', vars(worker))

    def test_installed_teacher_hook_is_rejected_without_mutation(self):
        original = object()
        for worker in [types.SimpleNamespace(original_sampler_sample=original),
                       types.SimpleNamespace(teacher_tokens=[3])]:
            before = dict(vars(worker))
            with self.assertRaisesRegex(RuntimeError, 'teacher hook'):
                diagnostic_call(worker, lambda *_: self.fail('The legacy exporter must not run.'), 'capture', 16)
            self.assertEqual(vars(worker), before)

    def test_actual_extension_calls_legacy_export_through_adapter(self):
        source = ast.parse((ROOT / 'long_context_probe.py').read_text())
        klass = next(node for node in source.body if isinstance(node, ast.ClassDef))
        class Parent:
            def finish_parity_capture(self, label, prefix):
                self.observed = (label, prefix, self.teacher_index)
                return dict(teacher_steps=self.teacher_index)
        namespace = dict(CampaignProbe=Parent, finish_native_allowed_capture=diagnostic_call)
        exec(compile(ast.Module(body=[klass], type_ignores=[]), str(ROOT / 'long_context_probe.py'), 'exec'), namespace)
        instance = namespace['LongContextDiagnosticProbe']()
        result = instance.finish_parity_capture('capture', 16)
        self.assertEqual(instance.observed, ('capture', 16, 0))
        self.assertEqual(result['teacher_steps'], 'NOT_APPLICABLE')
        self.assertFalse(hasattr(instance, 'teacher_index'))

    def test_native_style_base_injection_keeps_super_resolution(self):
        source = ast.parse((ROOT / 'long_context_probe.py').read_text())
        klass = next(node for node in source.body if isinstance(node, ast.ClassDef))
        class Parent:
            def finish_parity_capture(self, label, prefix):
                return dict(teacher_steps=self.teacher_index)
        namespace = dict(CampaignProbe=Parent, finish_native_allowed_capture=diagnostic_call)
        exec(compile(ast.Module(body=[klass], type_ignores=[]), str(ROOT / 'long_context_probe.py'), 'exec'), namespace)
        class Base:
            pass
        class NativeWorker(Base):
            pass
        # Match installed vLLM worker_base.py:278. No native package import occurs.
        NativeWorker.__bases__ += (namespace['LongContextDiagnosticProbe'],)
        worker = NativeWorker()
        result = worker.finish_parity_capture('capture', 16)
        self.assertEqual(result['teacher_steps'], 'NOT_APPLICABLE')
        self.assertFalse(hasattr(worker, 'teacher_index'))

    def test_runner_selects_adapter_and_keeps_native_allowed_token_path(self):
        tree = ast.parse((ROOT / 'scripts/native_long_context_control.py').read_text())
        extensions = [keyword.value.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                      for keyword in node.keywords if keyword.arg == 'worker_extension_cls']
        self.assertEqual(extensions, ['poc_ouro_parallelism_deep_discovery.long_context_probe.LongContextDiagnosticProbe'])
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == 'teacher' for node in calls))
        self.assertTrue(any(keyword.arg == 'allowed_token_ids' for node in calls for keyword in node.keywords))


if __name__ == '__main__':
    unittest.main()
