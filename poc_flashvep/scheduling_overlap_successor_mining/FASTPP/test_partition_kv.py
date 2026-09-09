"""CPU-only regression for the exact local native KV layer-count method.

Extracts its AST to avoid importing CUDA-capable serving modules. Uses the
pinned vLLM's actual get_pp_indices implementation, not a replacement partitioner.
"""
import ast
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def extract(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    function = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class PartitionKVTest(unittest.TestCase):
    def setUp(self):
        self.group = SimpleNamespace(rank_in_group=0, world_size=4)
        class Environment:
            @property
            def VLLM_PP_LAYER_PARTITION(self):
                return os.environ.get("VLLM_PP_LAYER_PARTITION")
        helper = extract(os.environ["SUCCESSOR_VLLM_UTILS"], "get_pp_indices",
                         {"envs": Environment(), "Tuple": tuple})
        self.helper = helper
        distributed, utils = ModuleType("vllm.distributed"), ModuleType("vllm.distributed.utils")
        distributed.get_pp_group = lambda: self.group
        utils.get_pp_indices = helper
        self.modules = patch.dict(sys.modules, {"vllm.distributed": distributed,
                                                "vllm.distributed.utils": utils})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        source = Path(os.environ["SUCCESSOR_FASTPP_SOURCE"])/"python/sglang/srt/configs/model_config.py"
        self.method = extract(source, "get_num_hidden_layers", {"os": os})
        self.model = SimpleNamespace(num_hidden_layers=48)

    def test_default_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            for rank in range(4):
                self.group.rank_in_group = rank
                self.assertEqual(self.method(self.model, 4), 12)

    def test_uneven_uses_owned_layers_without_kv_aliases(self):
        with patch.dict(os.environ, {"VLLM_PP_LAYER_PARTITION": "8,12,14,14"}):
            for rank, expected in enumerate([8, 12, 14, 14]):
                self.group.rank_in_group = rank
                local = self.method(self.model, 4)
                self.assertEqual(local, expected)
                start, end = self.helper(48, rank, 4)
                self.assertEqual(len({layer % local for layer in range(start, end)}), end-start)

    def test_invalid_partition_is_rejected(self):
        with patch.dict(os.environ, {"VLLM_PP_LAYER_PARTITION": "8,12,14,13"}):
            with self.assertRaises(ValueError):
                self.method(self.model, 4)


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    unittest.main()
