import unittest
from .contract import route


class Routes(unittest.TestCase):
    def test_layer_reentry(self):
        r = route('layer_pp4',4)
        self.assertEqual(len(r),16)
        self.assertEqual([x[0] for x in r], [0,1,2,3]*4)

    def test_loop_ownership(self):
        r = route('loop_pp4',4)
        self.assertEqual(r,[(0,0,0,24),(1,1,0,24),(2,2,0,24),(3,3,0,24)])

    def test_true_fixed_truncation(self):
        for depth in [2,3,4]:
            for arm in ['layer_pp4','loop_pp4']:
                r = route(arm,depth)
                self.assertEqual(sum(stop-start for _,_,start,stop in r),24*depth)
                self.assertEqual(max(loop for _,loop,_,_ in r),depth-1)


if __name__=='__main__': unittest.main()
