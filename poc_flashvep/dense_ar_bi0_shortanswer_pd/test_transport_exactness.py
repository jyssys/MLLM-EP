"""Check byte mappings before a coalesced Native transport experiment."""
import random
import unittest
import numpy as np
from .nixl_transport import coalesce


class ExactTransfer(unittest.TestCase):
    def test_random_byte_mapping(self):
        rng=random.Random(20261119)
        for _ in range(200):
            left=rng.sample(range(40),20);right=rng.sample(range(40),20)
            a=np.array([(1000+i*8,8,0) for i in left],dtype=np.uint64)
            b=np.array([(2000+i*8,8,0) for i in right],dtype=np.uint64)
            x,y=coalesce(a,b,[0]*20,[0]*20)
            before={(int(d)+j,int(s)+j) for (d,n,_),(s,_,_) in zip(a,b) for j in range(int(n))}
            after={(int(d)+j,int(s)+j) for (d,n,_),(s,_,_) in zip(x,y) for j in range(int(n))}
            self.assertEqual(before,after)

    def test_no_registered_layer_boundary_crossing(self):
        a=np.array([[1000,8,0],[1008,8,0]],dtype=np.uint64)
        x,_=coalesce(a,a,[0,1],[0,1]);self.assertEqual(len(x),2)

    def test_reverse_contiguous_blocks(self):
        a=np.array([[1032,8,0],[1024,8,0],[1016,8,0],[1008,8,0]],dtype=np.uint64)
        b=a.copy();b[:,0]+=1000
        x,y=coalesce(a,b,[0]*4,[0]*4)
        self.assertEqual(x.tolist(),[[1008,32,0]]);self.assertEqual(y.tolist(),[[2008,32,0]])

    def test_overlapping_destination_rejects_merge(self):
        a=np.array([[1000,8,0],[1000,8,0]],dtype=np.uint64)
        with self.assertRaises(AssertionError):coalesce(a,a,[0,0],[0,0])


if __name__=='__main__':unittest.main()
