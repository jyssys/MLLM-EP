"""Compare actual old fingerprint functions with the optimized diagnostic."""
import sys
import unittest
from types import SimpleNamespace as NS
from .common import prior
from .exact_fingerprint import native_fingerprint, partial_fingerprint

sys.path.insert(0, str(prior.parent.prior.RLT))


class ExactFingerprintTests(unittest.TestCase):
    def fixture(self):
        import torch
        tensor = torch.arange(5 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(5, 4, 3, 2, 2).to(torch.bfloat16)
        allocation = NS(block_tables=[[3, 1], [4, 2]],
                        written=[[NS(prefix=5) for _ in range(4)] for _ in range(2)])
        cache = NS(key_cache=tensor, value_cache=tensor + 1, num_layers=4, block_size=3,
                   max_loops=2, device='cpu', _get_allocation=lambda tid: allocation)
        connector = NS(cache=cache, engine=NS(model=NS(recurrent_kv_layers=[1, 2], coda_kv_layers=[3])),
                       hidden=tensor.flatten()[:16].reshape(1, 16),
                       layout=lambda transfer: [(0, 0, 5), (0, 1, 5), (1, 1, 5), (0, 3, 4)],
                       hidden_elements=lambda transfer: 16)
        return connector, NS(tid='fixture', slot=0, request=NS(prompt_token_ids=[1] * 5))

    def test_native_hash_and_partial_block_order_are_identical(self):
        from ..looped_pd_abc.verify import fingerprint
        connector, transfer = self.fixture()
        self.assertEqual(native_fingerprint(connector, transfer), fingerprint(connector, transfer))

    def test_partial_hash_and_plane_order_are_identical(self):
        from ..adaptive_recurrence_cut.connector import CutConnector
        connector, transfer = self.fixture()
        self.assertEqual(partial_fingerprint(connector, transfer), CutConnector.fingerprint(connector, transfer))

    def test_invalid_written_frontier_still_fails(self):
        connector, transfer = self.fixture()
        connector.cache._get_allocation(transfer.tid).written[1][1].prefix = 4
        with self.assertRaises(RuntimeError):
            partial_fingerprint(connector, transfer)


if __name__ == '__main__':
    unittest.main()
