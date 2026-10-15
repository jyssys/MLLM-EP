"""CPU tests for execution order, request statistics, and kernel labels."""
import unittest
import numpy as np
from poc_looped_recurrence_tp_4models.tied_tape import make_forward
from poc_looped_recurrence_tp_4models.analysis import bootstrap,category,union_ms
from poc_looped_recurrence_tp_4models.contract import GPUS,MODELS


class TapeTest(unittest.TestCase):
    def test_storage_and_state_order(self):
        class Core:
            def __init__(self):
                self.tape=[]
                self.layers=[self.layer(i) for i in range(48)]
            def layer(self,i):
                def call(pos,h,loop,residual):
                    self.tape.append((i,loop,residual is None,id(self.layers[i])))
                    return h+i+loop,1 if residual is None else residual+1
                return call
            def embed_input_ids(self,ids):return ids+1
            def norm(self,h,residual):
                self.tape.append(('norm',residual));return h+residual,None
        for R in [1,2,4]:
            core=Core();function,source=make_forward(R);actual=function(core,2,0)
            reference=Core();h=reference.embed_input_ids(2)
            for loop in range(R):
                residual=None
                for layer in reference.layers:h,residual=layer(0,h,loop,residual)
                h,_=reference.norm(h,residual)
            self.assertEqual(actual,h)
            self.assertEqual([x[:3] for x in core.tape],[x[:3] for x in reference.tape])
            self.assertNotIn('for ',source)
            for i in range(48):
                identities=[x[3] for x in core.tape if x[0]==i]
                self.assertEqual(len(set(identities)),1)
                self.assertEqual(len(identities),R)
    def test_statistics(self):
        stats=bootstrap([1,1,1]);self.assertEqual(stats['n'],3)
        self.assertEqual((stats['mean'],stats['low'],stats['high']),(1,1,1))
        self.assertEqual(union_ms([dict(start_us=0,end_us=2000),dict(start_us=1000,end_us=3000)]),3)
    def test_collective_labels(self):
        self.assertEqual(category('cross_device_reduce_1stage'),'TP_COLLECTIVE')
        self.assertEqual(category('ncclDevKernel_AllGather'),'TP_COLLECTIVE')
        self.assertEqual(category('flash::FlashAttnFwdCombine'),'ATTENTION_WITH_KV_READ')
    def test_scope(self):
        self.assertEqual(GPUS,[4,5,6,7]);self.assertEqual(len(MODELS),3)


if __name__=='__main__':unittest.main()
