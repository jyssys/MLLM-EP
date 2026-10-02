import unittest
from poc_native_lp_ep4.scripts.metrics import frame_audit


class FrameAuditTest(unittest.TestCase):
    def row(self, active, ranks):
        return {'run':'x','wave':1,'arm':'LP_G24','modality':'VISUAL_PREFILL',
            'iteration':['decode_step',43],'layer':0,'ranks':len(ranks),
            'rank_durations':{str(r):{} for r in ranks},'prefill_active':active}

    def test_partial_inactive_frames_remain_visible(self):
        audit=frame_audit([self.row(True,[0,1,2,3]),self.row(False,[0,1,2])])
        self.assertTrue(audit['all_active_prefill_frames_complete'])
        self.assertFalse(audit['all_captured_frames_complete'])
        self.assertEqual(audit['partial_frame_count'],1)
        self.assertEqual(audit['partial_active_prefill_frame_count'],0)

    def test_missing_active_rank_fails_closed(self):
        audit=frame_audit([self.row(True,[0,1,2])])
        self.assertFalse(audit['all_active_prefill_frames_complete'])
        self.assertEqual(audit['partial_active_prefill_frame_count'],1)


if __name__=='__main__':unittest.main()
