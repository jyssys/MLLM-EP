import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from .common import prior
sys.path.insert(0, str(prior.parent.prior.RLT))
from ..adaptive_recurrence_cut import state_probe as original
from ..adaptive_recurrence_cut import common as probe_common
from ..looped_pd_headroom_recovery import state_probe as candidate


class StateProbeTests(unittest.TestCase):
    def record(self, install):
        class Runner:
            def _execute_batch(self, batch, prepared):
                return 'unchanged'
        from vllm_rlt.request import Operation
        allocations = NS(block_tables=[[0, 1], [2, 3], [4, 5]])
        tensor = torch.arange(8 * 4 * 4 * 2).reshape(8, 4, 4, 2).to(torch.bfloat16)
        cache = NS(block_size=4, num_layers=4, device='cpu', key_cache=tensor,
                   value_cache=tensor + 10, _get_allocation=lambda _: allocations)
        owner = Runner()
        owner.cache_manager = cache
        owner.model = NS(recurrent_kv_layers=(1, 2), coda_kv_layers=(3,))
        request = NS(request_id='state', prompt_token_ids=list(range(7)),
                     sampling_params=NS(seed=1), arc_received_cut=2)
        batch = NS(operation=Operation.CODA, spans=[NS(request=request, token_start=6, token_count=1)])
        prepared = NS(forwards=[NS(forward=NS(hidden=torch.tensor([[1., 2.]], dtype=torch.bfloat16)))])
        rows = []
        with patch.dict(os.environ, ARC_RUN='CPU_fixture'), \
             patch.dict(sys.modules, {'vllm_rlt.worker.model_runner': NS(ModelRunner=Runner)}), \
             patch.object(probe_common, 'append', lambda path, row: rows.append(row)), \
             patch.object(candidate, 'append', lambda path, row: rows.append(row)):
            install(dict(state_probe=True, name='CPU_fixture'))
            self.assertEqual(owner._execute_batch(batch, prepared), 'unchanged')
        return rows

    def test_every_original_plane_and_hidden_hash_is_identical(self):
        baseline = self.record(original.install)
        changed = self.record(candidate.install)
        self.assertEqual([{k: v for k, v in row.items() if k != 'scope'} for row in baseline],
                         [{k: v for k, v in row.items() if k != 'scope'} for row in changed])
        self.assertEqual(len(changed[0]['planes']), 16)
        self.assertTrue(any(row['valid_tokens'] == 6 for row in changed[0]['planes']))


if __name__ == '__main__':
    unittest.main()
