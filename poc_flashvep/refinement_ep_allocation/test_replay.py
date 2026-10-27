"""Structural unit tests. These do not represent model measurements."""
import unittest
import numpy as np
from .allocation_replay import deletion,accounting

class ReplayTests(unittest.TestCase):
    def test_equal_budget_and_minimum_k(self):
        ids=np.asfortranarray(np.arange(128*8).reshape(128,8)%256)
        weights=np.arange(1,128*8+1,dtype=np.float32).reshape(128,8)
        masked=np.arange(128)%2==0
        prior=np.full(128,np.nan)
        for fraction in (.05,.20):
            counts=[]
            for policy in ('P2','P3','P4','P5'):
                removed=deletion(weights,ids,masked,prior,policy,fraction,3,1)
                self.assertTrue(np.all(removed.sum(1)<=7))
                counts.append(int(removed.sum()))
            self.assertEqual(counts,[4*int(32*8*fraction)]*4)

    def test_full_geometry(self):
        ids=np.tile(np.arange(8),(32,1))
        full=accounting(ids,np.zeros_like(ids,dtype=bool))
        self.assertEqual(full['edges'],256)
        self.assertEqual(full['remote_pairs'],24)
        self.assertEqual(full['total_padded_rows'],256)

if __name__=='__main__':
    unittest.main()
