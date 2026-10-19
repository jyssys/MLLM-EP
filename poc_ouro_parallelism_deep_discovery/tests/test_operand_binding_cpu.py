"""Test pointer metadata with CPU stand-ins. Do not import Torch."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from poc_ouro_parallelism_deep_discovery.operand_binding import (
    LiftedOperandBindingCapture, ROLES, expected_weight_bindings,
    parameter_snapshot, tensor_metadata,
)


class MetadataTensor:
    def __init__(self, pointer=1040, storage=1024, offset=8, shape=(2, 4),
                 stride=(4, 1), dtype="torch.bfloat16", real=True):
        self.pointer, self.storage, self.offset = pointer, storage, offset
        self.shape, self.steps, self.dtype, self.real = shape, stride, dtype, real
        self.is_meta = False

    def data_ptr(self):
        return self.pointer

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: self.storage)

    def storage_offset(self):
        return self.offset

    def stride(self):
        return self.steps

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected tensor operation: {name}")


def names(layers, mode):
    result = ["L_self_modules_embed_tokens_parameters_weight_",
              "L_self_modules_norm_parameters_weight_"]
    for depth in ([0] if mode == "alias" else range(4)):
        for layer in range(layers):
            result.extend(f"L_self_modules_campaign_execution_modules_{depth}_modules_"
                          f"{layer}_modules_{role}_parameters_weight_" for role in ROLES)
    return result


def capture_fixture(weight=None, runtime=None, original=None):
    weight = weight or MetadataTensor()
    runtime = runtime or weight
    calls = []
    sentinel = object()
    def ordinary(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel
    owner = SimpleNamespace(optimized_call=original or ordinary)
    original_call = owner.optimized_call
    parameters = parameter_snapshot([("model.weight", weight), ("model.alias", weight)])
    capture = LiftedOperandBindingCapture(
        owner, ["position", "weight"], parameters, {"weight": "model.weight"},
        is_real_tensor=lambda value: isinstance(value, MetadataTensor) and value.real,
        context_provider=lambda: {"lane": "CPU_TEST"},
    )
    return owner, capture, runtime, calls, sentinel, original_call


class BindingTests(unittest.TestCase):
    def test_metadata_only_and_alias_names(self):
        owner, capture, weight, calls, sentinel, original = capture_fixture()
        self.assertIs(owner.optimized_call(7, weight), sentinel)
        record = capture.finish()
        self.assertEqual(record["status"], "CAPTURED_MATCH")
        self.assertEqual(record["operands"][1]["parameter_names_on_storage"],
                         ["model.weight", "model.alias"])
        self.assertEqual(record["operands"][1]["storage_offset"], 8)
        self.assertIs(calls[0][0][1], weight)
        self.assertIs(owner.optimized_call, original)
        self.assertNotIn("torch", sys.modules)

    def test_only_one_call_is_observed(self):
        owner, capture, weight, calls, sentinel, original = capture_fixture()
        owner.optimized_call(7, weight)
        owner.optimized_call(8, weight)
        self.assertEqual(len(calls), 2)
        self.assertEqual(capture.finish()["calls_seen"], 1)

    def test_each_metadata_difference_fails(self):
        for field, value in [("pointer", 9000), ("storage", 8000), ("offset", 9),
                             ("shape", (4, 2)), ("steps", (1, 2)), ("dtype", "torch.float32")]:
            with self.subTest(field=field):
                other = MetadataTensor()
                setattr(other, field, value)
                owner, capture, weight, calls, sentinel, original = capture_fixture(runtime=other)
                owner.optimized_call(7, weight)
                self.assertEqual(capture.finish()["status"], "CAPTURED_MISMATCH")

    def test_fake_call_is_forwarded_then_real_call_is_observed(self):
        owner, capture, weight, calls, sentinel, original = capture_fixture()
        owner.optimized_call(7, MetadataTensor(real=False))
        self.assertIsNot(owner.optimized_call, original)
        owner.optimized_call(7, weight)
        self.assertEqual(capture.finish()["status"], "CAPTURED_MATCH")
        self.assertEqual(len(calls), 2)

    def test_finish_without_call_restores(self):
        owner, capture, weight, calls, sentinel, original = capture_fixture()
        self.assertEqual(capture.finish()["status"], "NOT_CAPTURED")
        self.assertIs(owner.optimized_call, original)
        self.assertEqual(calls, [])

    def test_original_exception_propagates(self):
        failure = RuntimeError("original failure")
        def ordinary(*args):
            raise failure
        owner, capture, weight, calls, sentinel, original = capture_fixture(original=ordinary)
        with self.assertRaises(RuntimeError) as result:
            owner.optimized_call(7, weight)
        self.assertIs(result.exception, failure)
        self.assertIs(owner.optimized_call, original)
        self.assertEqual(capture.finish()["original_call_outcome"], "RAISED")

    def test_metadata_error_does_not_replace_original_execution(self):
        class BrokenMetadata(MetadataTensor):
            def stride(self):
                raise ValueError("metadata failure")
        owner, capture, weight, calls, sentinel, original = capture_fixture(runtime=BrokenMetadata())
        self.assertIs(owner.optimized_call(7, weight), sentinel)
        self.assertEqual(capture.finish()["status"], "CAPTURE_ERROR")
        self.assertEqual(len(calls), 1)

    def test_keyword_and_arity_calls_remain_unobserved(self):
        owner, capture, weight, calls, sentinel, original = capture_fixture()
        owner.optimized_call(weight)
        owner.optimized_call(position=7, weight=weight)
        record = capture.finish()
        self.assertEqual(record["status"], "NOT_CAPTURED")
        self.assertEqual(len(record["skipped_calls"]), 2)
        self.assertEqual(len(calls), 2)

    def test_arm_does_not_call_original(self):
        owner, capture, weight, calls, sentinel, original = capture_fixture()
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            LiftedOperandBindingCapture(owner, ["weight"], {}, {"weight": "missing"}, lambda x: True)
        capture.finish()

    def test_all_model_arm_placeholder_maps(self):
        for layers, alias, copies in [(24, 194, 770), (48, 386, 1538)]:
            for mode, count in [("alias", alias), ("copies", copies)]:
                with self.subTest(layers=layers, mode=mode):
                    result = expected_weight_bindings(names(layers, mode), mode, layers)
                    self.assertEqual(len(result), count)
                    self.assertTrue(result[names(layers, mode)[-1]].startswith("model._campaign_execution."))

    def test_incorrect_depth_and_role_maps_fail(self):
        with self.assertRaises(ValueError):
            expected_weight_bindings(names(24, "alias"), "copies", 24)
        bad = [name.replace("modules_3_modules_", "modules_4_modules_", 1)
               if "execution_modules_3_" in name else name for name in names(24, "copies")]
        with self.assertRaises(ValueError):
            expected_weight_bindings(bad, "copies", 24)


class WorkerBindingTests(unittest.TestCase):
    def fixture(self, mode):
        modules = {name: ModuleType(name) for name in [
            "torch", "torch._subclasses", "torch._subclasses.fake_tensor",
            "vllm", "vllm.distributed", "vllm.forward_context",
            "poc_ouro_parallelism_deep_discovery.worker_probe",
        ]}
        modules["torch"].Tensor = MetadataTensor
        modules["torch._subclasses.fake_tensor"].FakeTensor = type("FakeTensor", (), {})
        modules["vllm.distributed"].get_tensor_model_parallel_rank = lambda: 0
        context = modules["vllm.forward_context"]
        context.is_forward_context_available = lambda: False
        context.get_forward_context = lambda: None
        modules["poc_ouro_parallelism_deep_discovery.worker_probe"].CampaignProbe = type("CampaignProbe", (), {})
        module_path = Path(__file__).resolve().parents[1] / "operand_binding_worker.py"
        spec = importlib.util.spec_from_file_location("test_worker_extension", module_path)
        extension = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(extension)
        placeholders = ["L_positions_"] + names(24, mode)
        bindings = expected_weight_bindings(placeholders, mode, 24)
        parameters = {"model.embed_tokens.weight": MetadataTensor(pointer=3000, storage=3000),
                      "model.norm.weight": MetadataTensor(pointer=4000, storage=4000)}
        for depth in range(4):
            for layer in range(24):
                for role_index, (role, path) in enumerate(ROLES.items()):
                    logical = ((depth if mode == "copies" else 0) * 24 + layer) * 8 + role_index
                    pointer = 10000 + 256 * logical
                    parameters[f"model._campaign_execution.{depth}.{layer}.{path}.weight"] = MetadataTensor(pointer=pointer, storage=pointer)
        sentinel = object()
        owner = SimpleNamespace(
            optimized_call=lambda *args: sentinel,
            graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=[
                SimpleNamespace(op="placeholder", target=name) for name in placeholders])),
            _fake_mode=None,
        )
        core = SimpleNamespace(layers=list(range(24)), _campaign_storage_evidence={"mode": mode},
                               aot_compiled_fn=SimpleNamespace(_artifacts=SimpleNamespace(compiled_fn=owner)))
        model = SimpleNamespace(model=core, named_parameters=lambda remove_duplicate: parameters.items())
        worker = extension.OperandBindingProbe()
        worker.get_model = lambda: model
        args = [7] + [parameters[bindings[name]] for name in placeholders[1:]]
        return modules, worker, owner, parameters, args, sentinel

    def test_worker_maps_each_arm_after_loading(self):
        for mode, count in [("alias", 194), ("copies", 770)]:
            with self.subTest(mode=mode):
                modules, worker, owner, parameters, args, sentinel = self.fixture(mode)
                with patch.dict(sys.modules, modules):
                    arm = worker.start_operand_binding_capture()
                    self.assertEqual(arm["weight_placeholder_count"], count)
                    self.assertIs(owner.optimized_call(*args), sentinel)
                    result = worker.finish_operand_binding_capture()
                self.assertEqual(result["status"], "CAPTURED_MATCH")
                self.assertEqual(len(result["execution_storage_groups"]), 192)

    def test_worker_rejects_incorrect_execution_storage(self):
        modules, worker, owner, parameters, args, sentinel = self.fixture("copies")
        name = "model._campaign_execution.1.0.input_layernorm.weight"
        parameters[name] = parameters["model._campaign_execution.0.0.input_layernorm.weight"]
        original = owner.optimized_call
        with patch.dict(sys.modules, modules), self.assertRaises(RuntimeError):
            worker.start_operand_binding_capture()
        self.assertIs(owner.optimized_call, original)

    def test_worker_rejects_unfinished_backend(self):
        modules, worker, owner, parameters, args, sentinel = self.fixture("alias")
        owner._fake_mode = object()
        with patch.dict(sys.modules, modules), self.assertRaises(RuntimeError):
            worker.start_operand_binding_capture()

    def test_full_replay_without_optimized_call_is_not_capture(self):
        modules, worker, owner, parameters, args, sentinel = self.fixture("alias")
        original = owner.optimized_call
        with patch.dict(sys.modules, modules):
            worker.start_operand_binding_capture()
            result = worker.finish_operand_binding_capture()
        self.assertEqual(result["status"], "NOT_CAPTURED")
        self.assertIs(owner.optimized_call, original)


if __name__ == "__main__":
    unittest.main()
