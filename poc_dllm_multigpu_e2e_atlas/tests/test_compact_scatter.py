import unittest
import torch
from poc_dllm_multigpu_e2e_atlas.compact_scatter import compact_scatter


class CompactScatterTests(unittest.TestCase):
    def test_large_backing_view_keeps_exact_values_without_alias(self):
        for length in [64,128,256]:
            base=torch.arange(20*2*length*4,dtype=torch.float32).view(20,2,1,1,length,4)
            original=base.clone();view=base[3,1];source=torch.randn(1,1,32,4)
            native=view.slice_scatter(source,dim=2,start=length-32,end=length)
            result=compact_scatter(view,source,2,length-32,length)
            self.assertTrue(torch.equal(result,native));self.assertTrue(torch.equal(base,original))
            self.assertEqual(native.untyped_storage().nbytes(),base.untyped_storage().nbytes())
            self.assertEqual(result.untyped_storage().nbytes(),view.numel()*view.element_size())
            self.assertNotEqual(result.untyped_storage().data_ptr(),view.untyped_storage().data_ptr())


if __name__=='__main__':unittest.main()
