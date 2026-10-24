"""CPU checks for ownership partitions, stable merge, and page accounting."""
import unittest
import numpy as np

class Contracts(unittest.TestCase):
    def test_partitions(self):
        for length in (1,15,17,2049,8193,32769,65409):
            for world in (2,4,8):
                bounds=[length*i//world for i in range(world+1)]
                self.assertEqual(bounds[0],0);self.assertEqual(bounds[-1],length)
                self.assertEqual(sum(b-a for a,b in zip(bounds,bounds[1:])),length)
                self.assertTrue(all(a<=b for a,b in zip(bounds,bounds[1:])))
    def test_merge(self):
        rng=np.random.default_rng(173)
        scores=rng.normal(size=(3,37))*5
        values=rng.normal(size=(37,5))
        native=np.exp(scores-scores.max(1,keepdims=True))
        native=native/native.sum(1,keepdims=True)@values
        for world in (2,4,8):
            zs=[];os=[]
            for index in np.array_split(np.arange(37),world):
                a=scores[:,index];m=a.max(1,keepdims=True);e=np.exp(a-m)
                zs.append((m+np.log(e.sum(1,keepdims=True)))[:,0])
                os.append(e/e.sum(1,keepdims=True)@values[index])
            z=np.stack(zs);weights=np.exp(z-z.max(0));weights/=weights.sum(0)
            result=(np.stack(os)*weights[:,:,None]).sum(0)
            np.testing.assert_allclose(native,result,rtol=1e-12,atol=1e-12)
    def test_logical_payload(self):
        per_token=2*2*24*4*16*128
        self.assertEqual(per_token,786432)
        self.assertEqual(per_token*16384,12*2**30)
        self.assertEqual(18432*24*16*16*128*2*2,54*2**30)

if __name__=='__main__':unittest.main()
