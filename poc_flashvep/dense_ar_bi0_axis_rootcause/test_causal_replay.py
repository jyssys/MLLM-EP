import unittest
from .causal_replay import simulate


class CausalReplayTest(unittest.TestCase):
    def test_credit1_releases_at_first_not_decode_finish(self):
        v=simulate([0.,0.],[2.,2.],[3.,3.],[1.,1.],1)
        self.assertEqual(v['rows'][0]['first'],5)
        self.assertEqual(v['rows'][1]['P_start'],5)

    def test_serial_admission_is_a_separate_barrier(self):
        v=simulate([0.,0.],[2.,2.],[3.,3.],[1.,1.],4,serial=True)
        self.assertEqual(v['rows'][1]['P_start'],6)

    def test_source_and_destination_can_overlap_with_credit4(self):
        v=simulate([0.,0.],[2.,2.],[3.,3.],[1.,1.],4)
        self.assertEqual(v['rows'][1]['P_start'],2)
        self.assertEqual(v['rows'][1]['D_start'],6)


if __name__=='__main__':unittest.main()
