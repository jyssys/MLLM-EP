"""Keep allocation ownership separate from new logical block ordering."""
import random,unittest
from types import SimpleNamespace
from .canonical_blocks import canonicalize


class CanonicalBlockTests(unittest.TestCase):
    def test_only_return_order_changes(self):
        rng=random.Random(20261115)
        for n in (0,1,32,500,501):
            original=[SimpleNamespace(block_id=i+1,ref_cnt=1,is_null=False) for i in range(n)]
            returned=list(original);rng.shuffle(returned);identities={id(b) for b in returned}
            result=canonicalize(returned)
            self.assertIs(result,returned)
            self.assertEqual({id(b) for b in result},identities)
            self.assertEqual([b.block_id for b in result],list(range(1,n+1)))
            self.assertTrue(all(b.ref_cnt==1 and not b.is_null for b in result))
    def test_same_orientation_allows_exact_range_coalescing(self):
        import numpy as np
        from .manual_coalescing import coalesce
        source=[SimpleNamespace(block_id=i+1) for i in range(16)]
        target=[SimpleNamespace(block_id=100+i) for i in reversed(range(16))]
        remote=np.array([[b.block_id*64,64,0] for b in source],dtype=np.uint64)
        local=np.array([[b.block_id*64,64,0] for b in target],dtype=np.uint64)
        self.assertEqual(len(coalesce(local,remote,[0]*16,[0]*16)[0]),16)
        canonicalize(target)
        local=np.array([[b.block_id*64,64,0] for b in target],dtype=np.uint64)
        merged=coalesce(local,remote,[0]*16,[0]*16)
        self.assertEqual(len(merged[0]),1)
        self.assertEqual(merged[0][0,1],16*64)


if __name__=='__main__':unittest.main()
