import unittest
from poc_native_lp_ep4.scripts.analyze import victim_itls


class CompletionTimeTest(unittest.TestCase):
    def test_engine_clock_not_poll_backlog_and_canonical_order(self):
        wave={'resident_ids':['9-random','10-random'],
              'emitted':{'9':[1,1,1],'10':[2,2,2]},'steps':[]}
        for i in range(3):
            wave['steps'].append({'engine_end_ns':i*1000000,
                'end_ns':i*i*500000000,
                'outputs':[{'id':'10','tokens':[2]},{'id':'9','tokens':[1]}]})
        arrays=victim_itls(wave,3)
        self.assertEqual([a.tolist() for a in arrays],[[1.,1.],[1.,1.]])


if __name__=='__main__':unittest.main()
