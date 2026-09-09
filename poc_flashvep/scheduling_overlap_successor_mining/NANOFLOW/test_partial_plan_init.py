"""CPU-only lifecycle test; no native or torch import."""
from types import SimpleNamespace
import unittest
from nano_plan_adapter import initialize_unplanned_ops, UNPLANNED_TAGS


class Op:
    def __init__(self):
        self.calls = []
    def config_tag(self, tag, params):
        self.calls.append((tag, params))
        self.impl = object()


class PartialPlanInitTest(unittest.TestCase):
    def test_first_use_and_existing_implementation(self):
        pipeline = SimpleNamespace(cuda_graph_enabled=False,
                                   **{name: Op() for name in UNPLANNED_TAGS})
        existing = pipeline.kqv.impl = object()
        initialize_unplanned_ops(pipeline)
        self.assertIs(pipeline.kqv.impl, existing)
        self.assertEqual(pipeline.kqv.calls, [])
        for name, tag in UNPLANNED_TAGS.items():
            if name != "kqv":
                self.assertEqual(getattr(pipeline, name).calls, [(tag, {"use_cuda_graph": False})])
        initialize_unplanned_ops(pipeline)
        self.assertEqual(len(pipeline.pfAttn.calls), 1)


if __name__ == "__main__":
    unittest.main()
