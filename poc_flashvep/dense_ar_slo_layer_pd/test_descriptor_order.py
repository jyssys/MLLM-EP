"""Check exact pair preservation without a GPU."""
import random,unittest
from .descriptor_order import ordered_pairs,runs


class DescriptorTests(unittest.TestCase):
    def test_reverse_run(self):
        a=[4,3,2,1];b=[24,23,22,21]
        self.assertEqual(runs(a,b),4)
        x,y=ordered_pairs(a,b);self.assertEqual(runs(x,y),1)
        self.assertEqual(sorted(zip(a,b)),sorted(zip(x,y)))
    def test_unrelated_fragmentation_cannot_be_removed(self):
        a=[1,2,3];b=[30,10,20]
        x,y=ordered_pairs(a,b);self.assertEqual(runs(x,y),3)
    def test_fuzz_pair_mapping(self):
        rng=random.Random(14)
        for size in range(100):
            a=rng.sample(range(1000),size);b=rng.sample(range(1000),size)
            x,y=ordered_pairs(a,b)
            self.assertEqual(sorted(zip(a,b)),list(zip(x,y)))
            self.assertEqual(x,sorted(a))


if __name__=='__main__':unittest.main()
