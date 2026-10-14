import gzip
import json
import unittest
import numpy as np
from poc_looped_moe_ep.contract import *
from poc_looped_moe_ep.metrics import traffic,jsd,balanced_candidate


class ContractTests(unittest.TestCase):
    def test_actual_gpu_mask(self):
        self.assertEqual(MASK,'4,5,6,7');self.assertEqual(GPUS,[4,5,6,7])

    def test_native_top2_destination_dedup(self):
        mapping=[e//16 for e in range(64)]
        result=traffic([[16,17],[32,48],[0,1]],0,mapping)
        self.assertEqual(result['remote_paths'],4)
        self.assertEqual(result['dispatch_bytes'],3*2048)
        self.assertEqual(result['combine_bytes'],3*2048)

    def test_arbitrary_weight_preserving_owner_mapping(self):
        mapping=[(e//16+1)%4 for e in range(64)]
        result=traffic([[16,17]],0,mapping)
        self.assertEqual(result['rank_counts'].tolist(),[0,0,2,0])
        self.assertEqual(result['dispatch_bytes'],2048)

    def test_contiguous_publication_actual_seam(self):
        import torch
        from poc_looped_moe_ep.runtime import restoration_buffer
        x=torch.arange(12).reshape(3,4).t()
        self.assertFalse(torch.zeros_like(x).is_contiguous())
        restored=restoration_buffer(x)
        self.assertTrue(restored.is_contiguous());self.assertEqual(tuple(restored.shape),tuple(x.shape))

    def test_recorded_pre_fix_failure_and_post_fix_pass(self):
        old=(ROOT/'results/raw/physical_gate_v1/console.log').read_text()
        self.assertIn('Tensors must be contiguous',old)
        for rank in range(4):
            result=json.loads((ROOT/f'results/raw/physical_gate_v2_contiguous/rank{rank}.json').read_text())
            self.assertTrue(result['pass_all']);self.assertEqual(result['rel_l2'],0)

    def test_oracle_candidate_is_capacity_legal(self):
        counts=np.arange(64)
        candidate=balanced_candidate(counts,[e//16 for e in range(64)])
        self.assertTrue(all(candidate.count(r)==16 for r in range(4)))

    def test_jsd_units_and_symmetric(self):
        self.assertEqual(jsd([1,0],[0,1]),1)
        self.assertAlmostEqual(jsd([2,1],[1,3]),jsd([1,3],[2,1]))

    def test_frozen_optimizer_excludes_heldout(self):
        policy=json.loads((ROOT/'results/frozen_best_static.json').read_text())
        manifest=json.loads((ROOT/'results/frozen_requests.json').read_text())
        held={r['request_id'] for r in manifest['requests'] if r['split']=='held_out'}
        self.assertFalse(held&set(policy['calibration_request_ids']))
        self.assertEqual(policy['held_out_requests_used'],0)

    def test_proxy_objective_is_loopwise_not_global_only(self):
        from poc_looped_moe_ep.scripts.optimize_best_static import score
        counts=np.array([[10,0,0,0],[0,10,0,0]])
        self.assertEqual(score(counts,[0,1,2,3]),20)

    def test_spec_exact_copy(self):
        original=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_moe_ep_poc_spec.md')
        self.assertEqual(sha(original),sha(ROOT/'authoritative_spec.md'))

    def test_no_claim_of_validated_latency_oracle(self):
        result=json.loads((ROOT/'results/placement_comparison.json').read_text())
        self.assertEqual(result['S2_cost_domain'],'STRUCTURAL_ASSIGNMENT_PATHS_ONLY')
        self.assertIn('UNVALIDATED',result['S1_to_S2_EP_latency_headroom'])


if __name__=='__main__':unittest.main()
