"""CPU-only configuration controls for the unexecuted diagnostic runner."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from poc_ouro_parallelism_deep_discovery.cuda_graph_argument_getter import OPT_IN
from poc_ouro_parallelism_deep_discovery.scripts.native_graph_argument_probe import (
    frozen_request, main, parse_args, reference_for, scoped_config)


class ArgumentRunnerTests(unittest.TestCase):
    def test_import_does_not_load_torch_or_cuda(self):
        code = ("import sys; import poc_ouro_parallelism_deep_discovery.scripts.native_graph_argument_probe; "
                "assert 'torch' not in sys.modules; "
                "assert 'libcudart' not in open('/proc/self/maps').read(); "
                "assert 'libcuda.so' not in open('/proc/self/maps').read()")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_worker_import_gate_precedes_torch_import(self):
        code = ("import sys\ntry:\n import poc_ouro_parallelism_deep_discovery.graph_argument_probe\n"
                "except RuntimeError as e:\n assert 'OURO_CUDA_GRAPH_ARGUMENT_DIAGNOSTIC' in str(e)\n"
                "else:\n raise AssertionError('Missing opt-in was accepted')\n"
                "assert 'torch' not in sys.modules\n"
                "assert 'libcudart' not in open('/proc/self/maps').read()\n")
        environment = dict(os.environ, **{OPT_IN: "0"})
        subprocess.run([sys.executable, "-c", code], env=environment,
                       check=True, capture_output=True, text=True)

    def test_runner_gate_precedes_engine_configuration(self):
        with mock.patch.dict(os.environ, {OPT_IN: "0"}), mock.patch(
                "poc_ouro_parallelism_deep_discovery.scripts.native_graph_argument_probe.configure_native_imports") as imports:
            with self.assertRaisesRegex(RuntimeError, OPT_IN):
                main(["--size", "1.4B", "--tp", "2", "--arm", "alias"])
            imports.assert_not_called()

    def test_guard_gate_precedes_engine_import(self):
        with mock.patch.dict(os.environ, {OPT_IN: "1", "LOOPED_KV_OUTPUT": "/not-an-authorized-output"}), mock.patch(
                "poc_ouro_parallelism_deep_discovery.scripts.native_graph_argument_probe.configure_native_imports") as imports:
            with self.assertRaisesRegex(RuntimeError, "gpu_session guard"):
                main(["--size", "1.4B", "--tp", "2", "--arm", "native"])
            imports.assert_not_called()

    def test_scoped_configuration_preserves_source(self):
        source = {"compilation_config": {"cache_dir": "/old", "cudagraph_capture_sizes": [1, 2, 4, 8]},
                  "worker_extension_cls": "original", "async_scheduling": True,
                  "tensor_parallel_size": 2}
        original = copy.deepcopy(source)
        result = scoped_config(source, Path("/new-output"))
        self.assertEqual(source, original)
        self.assertEqual(result["compilation_config"]["cache_dir"], "/new-output/compile_cache")
        self.assertTrue(result["worker_extension_cls"].endswith("ArgumentGraphDebugProbe"))
        self.assertEqual(result["compilation_config"]["cudagraph_capture_sizes"], [1, 2, 4, 8])
        self.assertTrue(result["async_scheduling"])

    def test_frozen_request_is_exact_and_independent(self):
        row = {"id": "fixture", "prompt_ids": {"1024": list(range(1024))}, "teacher_ids": [1, 2, 3, 4, 5]}
        request = frozen_request({"inputs": [row]})
        self.assertEqual(request["teacher_token_ids"], [1, 2, 3, 4])
        request["prompt_token_ids"][0] = -1
        self.assertEqual(row["prompt_ids"]["1024"][0], 0)
        for change in [lambda r: r["teacher_ids"].clear(),
                lambda r: r["prompt_ids"]["1024"].pop(),
                lambda r: r["teacher_ids"].__setitem__(0, True)]:
            invalid = copy.deepcopy(row)
            change(invalid)
            with self.assertRaises(ValueError):
                frozen_request({"inputs": [invalid]})

    def test_native_reference_and_storage_tp_limit(self):
        args = parse_args(["--size", "2.6B", "--tp", "4", "--arm", "native"])
        name, path, manifest = reference_for(args)
        self.assertEqual(name, "control_async_base26_P4_v1")
        self.assertEqual(path.name, "2.6B_R4_TP4")
        self.assertIsNone(manifest)
        args = parse_args(["--size", "1.4B", "--tp", "1", "--arm", "copies"])
        with self.assertRaisesRegex(ValueError, "TP2 or TP4"):
            reference_for(args)


if __name__ == "__main__":
    unittest.main()
