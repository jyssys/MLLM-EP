import unittest

from poc_dispatch_combine_asymmetry.contract import geometry
from poc_dispatch_combine_asymmetry.scripts.analyze import critical, ci
from poc_dispatch_combine_asymmetry.selection import is_native_prefill
from poc_dispatch_combine_asymmetry.accounting import account


class Invariants(unittest.TestCase):
    def test_destination_dedup_not_expert_pair_bytes(self):
        r=account([[0,1,2,32,33,64,65,96]],0)
        self.assertEqual(r['valid_paths'],8)
        self.assertEqual(r['logical_dispatch_bytes'],4*4096)
        self.assertEqual(r['logical_remote_bytes'],3*4096)
        self.assertEqual(r['logical_dispatch_bytes'],r['logical_combine_bytes'])

    def test_empty_and_invalid_source(self):
        self.assertEqual(account([[-1]*8],2)['logical_dispatch_bytes'],0)
        with self.assertRaises(ValueError):account([[0]*8],4)
        with self.assertRaises(ValueError):account([[128]*8],0)
        with self.assertRaises(ValueError):account([[-2]*8],0)
    def test_idle_forward_excluded(self):
        self.assertFalse(is_native_prefill([0,0],2029))
        self.assertFalse(is_native_prefill([1,0],2029))
        self.assertTrue(is_native_prefill([2029,0],2029))
    def test_wrong_geometry_rejected(self):
        with self.assertRaises(ValueError):
            geometry(dict(hidden_size=2048,moe_intermediate_size=512,num_experts=256,num_experts_per_tok=8,num_hidden_layers=19))

    def test_no_cross_rank_maxima(self):
        rows=[]
        for rank in range(4):
            for layer in range(48):
                d,e,c=(8,1,1) if rank==0 else (1,6,1)
                rows.append(dict(ep_rank=rank,layer=layer,dispatch_ms=d,expert_ms=e,combine_ms=c,moe_ms=d+e+c,subcomponents={}))
        result=critical(rows)
        self.assertEqual(result['critical_rank'],0)
        self.assertEqual(result['expert_ms'],1)
        self.assertEqual(result['moe_ms'],10)

    def test_missing_rank_rejected(self):
        with self.assertRaises(ValueError):critical([])

    def test_duplicate_layer_rejected(self):
        rows=[dict(ep_rank=r,layer=l%47,moe_ms=1) for r in range(4) for l in range(48)]
        with self.assertRaises(ValueError):critical(rows)

    def test_request_not_layer_bootstrap(self):
        self.assertEqual(ci([1,1,1])['n'],3)
        self.assertEqual(ci([1,1,1])['ci95'],[1,1])


if __name__=='__main__':unittest.main()
