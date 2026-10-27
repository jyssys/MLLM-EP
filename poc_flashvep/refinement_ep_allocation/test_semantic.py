"""Synthetic tensor unit tests, not proxy-model experiments."""
import unittest
import torch
from .semantic_emulator import select_edges

class SelectionTests(unittest.TestCase):
    def test_exact_budget_and_minimum_k(self):
        weights=torch.arange(1,65,dtype=torch.float32).reshape(8,8)
        weights=weights/weights.sum(-1,keepdim=True)*2.5
        old=weights.clone()
        for kind in ('P1','P2','P3'):
            for fraction in (0.,.05,.20):
                removed=select_edges(weights,torch.arange(8)%2==0,
                    torch.tensor([.8,.9,.94,float('nan'),.2,.6,.7,.85]),kind,fraction,0,1,3)
                self.assertEqual(int(removed.sum()),int(64*fraction))
                self.assertTrue(torch.all(removed.sum(-1)<=7))
                self.assertTrue(torch.equal(weights,old))
                self.assertTrue(torch.equal(removed,select_edges(weights,torch.arange(8)%2==0,
                    torch.tensor([.8,.9,.94,float('nan'),.2,.6,.7,.85]),kind,fraction,0,1,3)))

    def test_router_ignores_refinement_values(self):
        w=torch.ones(8,8)
        a=select_edges(w,torch.zeros(8,dtype=torch.bool),torch.zeros(8),'P2',.2,0,1,3)
        b=select_edges(w,torch.ones(8,dtype=torch.bool),torch.ones(8),'P2',.2,0,1,3)
        self.assertTrue(torch.equal(a,b))

    def test_noncontiguous_geometry(self):
        w=torch.arange(64,dtype=torch.float32).reshape(8,8).T
        removed=select_edges(w,torch.ones(8,dtype=torch.bool),torch.zeros(8),'P2',.20,0,1,3)
        self.assertEqual(int(removed.sum()),12)

    def test_uniform_selection_and_rng_isolation(self):
        w=torch.ones(1024,8)
        state=torch.random.get_rng_state()
        mask=select_edges(w,torch.zeros(1024,dtype=torch.bool),torch.zeros(1024),'P1',.20,0,1,3)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertGreater(int((mask.sum(1)>0).sum()),500)
        self.assertEqual(int(mask.sum()),int(8192*.20))

if __name__=='__main__':
    unittest.main()
