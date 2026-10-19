"""Deterministic CPU regressions. These are not pretrained model evidence."""
import unittest
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
configure_imports()
import torch
from torch import nn
from poc_ouro_parallelism_deep_discovery.alias_control import clone_module

class BoundOperation(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor([2.,3.],dtype=torch.bfloat16),requires_grad=False)
        self._forward_method=self.compute
    def compute(self,x):return x*self.weight
    def forward(self,x):return self._forward_method(x)

class Tests(unittest.TestCase):
    def test_bound_method_consumes_copied_weight(self):
        original=BoundOperation();clone=clone_module(original,set())
        self.assertIs(clone._forward_method.__self__,clone)
        self.assertNotEqual(original.weight.data_ptr(),clone.weight.data_ptr())
        x=torch.ones(2,dtype=torch.bfloat16)
        self.assertTrue(torch.equal(original(x),clone(x)))
        clone.weight.data[0]=7
        self.assertEqual(original(x)[0].item(),2)
        self.assertEqual(clone(x)[0].item(),7)

    def test_named_attention_object_stays_shared(self):
        original=nn.ModuleDict({'operator':BoundOperation(),'attention':nn.Identity()})
        clone=clone_module(original,{id(original['attention'])})
        self.assertIs(clone['attention'],original['attention'])
        self.assertIsNot(clone['operator'],original['operator'])

    def test_buffer_container_is_independent(self):
        original=BoundOperation();original.register_buffer('immutable_table',torch.arange(3))
        clone=clone_module(original,set())
        self.assertIsNot(clone._buffers,original._buffers)
        self.assertIs(clone.immutable_table,original.immutable_table)

if __name__=='__main__':unittest.main()
