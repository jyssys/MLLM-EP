"""Compare every transferred byte under random physical block layouts."""
import random,unittest
from .manual_coalescing import coalesce


class CoalescingTests(unittest.TestCase):
    def test_exact_mapping(self):
        import numpy as np
        rng=random.Random(14)
        for _ in range(200):
            ids=rng.sample(range(40),20);remote=rng.sample(range(40),20)
            a=np.array([(1000+i*8,8,0) for i in ids],dtype=np.uint64)
            b=np.array([(2000+i*8,8,0) for i in remote],dtype=np.uint64)
            x,y=coalesce(a,b,np.zeros(20),np.zeros(20))
            before={(int(d)+j,int(s)+j) for (d,n,k),(s,_,_) in zip(a,b) for j in range(int(n))}
            after={(int(d)+j,int(s)+j) for (d,n,k),(s,_,_) in zip(x,y) for j in range(int(n))}
            self.assertEqual(before,after)
    def test_contiguous_reverse_blocks_merge(self):
        import numpy as np
        a=np.array([[1032,8,0],[1024,8,0],[1016,8,0],[1008,8,0]],dtype=np.uint64)
        b=a.copy();b[:,0]+=1000
        x,y=coalesce(a,b,[0]*4,[0]*4)
        self.assertEqual(x.tolist(),[[1008,32,0]]);self.assertEqual(y.tolist(),[[2008,32,0]])
    def test_registered_region_boundary_is_not_crossed(self):
        import numpy as np
        a=np.array([[1000,8,0],[1008,8,0]],dtype=np.uint64);b=a.copy()
        x,y=coalesce(a,b,[0,1],[0,1]);self.assertEqual(len(x),2)
    def test_overlapping_destinations_retain_original_path(self):
        import numpy as np
        a=np.array([[1000,8,0],[1000,8,0]],dtype=np.uint64)
        with self.assertRaises(AssertionError):coalesce(a,a,[0,0],[0,0])


if __name__=='__main__':unittest.main()
