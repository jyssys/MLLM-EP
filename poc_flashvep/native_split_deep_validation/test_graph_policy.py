"""Replay the native cache admission seam without CUDA allocation."""
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from .common import prior
from .graph_policy import preferred_geometry

sys.path.insert(0, str(prior.parent.prior.RLT))


class GraphPolicyTests(unittest.TestCase):
    def cfg(self):
        return dict(length=4096, native_frontier=[3, 3072], P_slots=2, D_slots=8,
                    P_prefill_graph_rows=8192, D_prefill_graph_rows=4096, deep_graph_budget=4)

    def test_primary_geometries_fit_four_entries(self):
        self.assertEqual(preferred_geometry(self.cfg(), 'prefill'),
                         {(4096, 4096), (8192, 4096), (3072, 3072), (6144, 3072)})

    def test_eight_entry_policy_admits_non_power_of_two_D_batch(self):
        cfg = dict(self.cfg(), deep_graph_budget=8)
        self.assertIn((3072, 1024), preferred_geometry(cfg, 'decode'))
        self.assertEqual(preferred_geometry(cfg, 'decode'),
                         {(1024, 1024), (2048, 1024), (3072, 1024), (4096, 1024)})

    def replay(self, reserved):
        from vllm_rlt.worker import cuda_graph as module
        graph = object.__new__(module.RecurrentGraphs)
        graph.entries = {}
        graph.execution = NS(cuda_graph_max_graphs=4)
        graph.fallbacks = 0
        graph.warned_cache_full = False
        graph.cache = NS()
        graph.model = NS(config=NS(max_position_embeddings=8192))
        allowed = preferred_geometry(self.cfg(), 'prefill')
        order = [(8192, 4096), (6144, 3072), (1024, 512), (384, 384),
                 (4096, 4096), (3072, 3072)]
        with patch.object(module, 'cuda_graph_rows', return_value=8192), \
             patch.object(module, '_graph_entry', return_value=NS(graph=None)):
            for count, query in order:
                if reserved and (count, query) not in allowed:
                    continue
                allocation = object()
                metadata = NS(rows=[(allocation, 0, i) for i in range(count)],
                              cu_seqlens_q=[0, count], position_ids=range(count),
                              context_lengths=range(count // query), max_seqlen_q=query)
                module.RecurrentGraphs.select(graph, range(count), metadata)
        return {key[0] for key in graph.entries}

    def test_legacy_reproduces_primary_shape_loss(self):
        self.assertNotIn(4096, self.replay(False))

    def test_reserved_admission_preserves_both_primary_shapes(self):
        self.assertEqual(self.replay(True), {4096, 8192, 3072, 6144})


if __name__ == '__main__':
    unittest.main()
