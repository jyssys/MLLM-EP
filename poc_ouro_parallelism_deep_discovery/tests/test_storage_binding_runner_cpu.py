"""Check runner plans without Torch imports or GPU execution."""

from pathlib import Path
import sys
import unittest

from poc_ouro_parallelism_deep_discovery.scripts.storage_binding_probe import (
    frozen_request, parse_args, scoped_config,
)


class RunnerTests(unittest.TestCase):
    def test_arguments(self):
        args = parse_args(["--size", "2.6B", "--tp", "4", "--arm", "copies"])
        self.assertEqual((args.size, args.tp, args.arm), ("2.6B", 4, "copies"))

    def test_config_preserves_reference(self):
        source = {"tensor_parallel_size": 4, "async_scheduling": True,
                  "worker_extension_cls": "original", "hf_overrides": {"total_ut_steps": 4},
                  "compilation_config": {"cache_dir": "/old", "cudagraph_capture_sizes": [1, 2, 4, 8]}}
        output = Path("/new/evidence")
        config = scoped_config(source, output)
        self.assertEqual(source["compilation_config"]["cache_dir"], "/old")
        self.assertEqual(source["worker_extension_cls"], "original")
        self.assertEqual(config["compilation_config"]["cache_dir"], str(output / "compile_cache"))
        self.assertEqual(config["compilation_config"]["cudagraph_capture_sizes"], [1, 2, 4, 8])
        self.assertTrue(config["async_scheduling"])

    def test_frozen_request(self):
        row = {"id": "frozen", "prompt_ids": {"1024": list(range(1024))},
               "teacher_ids": [10, 11, 12, 13, 14]}
        request = frozen_request({"inputs": [row]})
        self.assertEqual(request["teacher_token_ids"], [10, 11, 12, 13])
        self.assertEqual(len(request["prompt_token_ids"]), 1024)
        self.assertNotIn("torch", sys.modules)

    def test_wrong_workload_fails(self):
        with self.assertRaises(ValueError):
            frozen_request({"inputs": [{"id": "bad", "prompt_ids": {"1024": [1]},
                                         "teacher_ids": [1, 2, 3, 4]}]})


if __name__ == "__main__":
    unittest.main()
