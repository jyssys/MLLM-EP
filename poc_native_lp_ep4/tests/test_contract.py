import unittest
from poc_native_lp_ep4.contract import partition,groups_for


class ContractTest(unittest.TestCase):
    def test_partition_covers_every_layer_once(self):
        for groups in (1,4,8,12,16,24,48):
            self.assertEqual([i for lo,hi in partition(groups) for i in range(lo,hi)],list(range(48)))
            self.assertEqual(len(partition(groups)),groups)

    def test_upstream_auto_boundary(self):
        self.assertEqual(groups_for("LP_AUTO",2029),4)
        self.assertEqual(groups_for("LP_AUTO",2049),8)
        self.assertEqual(groups_for("LP_G16",2029),16)

    def test_decode_metadata_keeps_correct_kv_rows(self):
        import torch
        from poc_native_lp_ep4.native import decode_metadata
        cls=type("FlashAttentionMetadata",(),{})
        m=cls();m.use_cascade=False
        m.seq_lens=torch.tensor([12,2029,14])
        m.block_table=torch.tensor([[1,2],[5,6],[7,8]])
        m.slot_mapping=torch.arange(2031)
        out=decode_metadata(m,[0,2],torch.tensor([0,2030]))
        self.assertEqual(out.query_start_loc.tolist(),[0,1,2])
        self.assertEqual(out.seq_lens.tolist(),[12,14])
        self.assertEqual(out.block_table.tolist(),[[1,2],[7,8]])
        self.assertEqual(out.slot_mapping.tolist(),[0,2030])
        self.assertEqual(m.slot_mapping.numel(),2031)


if __name__=="__main__": unittest.main()
