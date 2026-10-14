import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from poc_dllm_prefix_canvas_cp.contract import DINFER
sys.path.insert(0,str(DINFER/'python'))
from poc_dllm_prefix_canvas_cp.observer import Observer


def context(mode):
    model=SimpleNamespace(forward=lambda **kw:None,model=SimpleNamespace(layers=[]))
    dllm=SimpleNamespace(block_runner=SimpleNamespace(prefill=lambda:None,decode=lambda:None),
                         diff_iteration=SimpleNamespace(forward=lambda *a,**kw:None))
    return Observer(model,dllm,0,mode,512)


class ObserverBindings(unittest.TestCase):
    def test_native_token_update_is_not_rebound_to_collective(self):
        from dinfer.decoding.utils import TokenArray
        x=object.__new__(TokenArray);x.data=torch.zeros((1,2),dtype=torch.long)
        with patch.object(Observer,'event',return_value=object()),context('collectives'):
            x[:,0:2]=torch.ones((1,2),dtype=torch.long)
        self.assertTrue(torch.equal(x.data,torch.ones_like(x.data)))

    def test_sdpa_is_not_rebound_to_slice_scatter(self):
        import torch.nn.functional as F
        q=torch.ones(1,1,1,2);k=torch.ones(1,1,2,2);v=torch.ones_like(k)
        with patch.object(Observer,'event',return_value=object()),context('trace'):
            result=F.scaled_dot_product_attention(q,k,v)
        self.assertTrue(torch.equal(result,q))


if __name__=='__main__':unittest.main()
