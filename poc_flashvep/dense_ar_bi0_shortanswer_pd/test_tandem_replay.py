import unittest
from .tandem_replay import replay


class QueueState(unittest.TestCase):
    def test_action_updates_later_source_and_destination_wait(self):
        a=replay([0,0,0],[2,2,2],[1,1,1],[0,0,0])
        b=replay([0,0,0],[1,2,2],[2,1,1],[0,0,0])
        self.assertEqual(a[0]['D_first'],b[0]['D_first'])
        self.assertNotEqual(a[1]['P_start'],b[1]['P_start'])
        self.assertNotEqual(a[2]['D_first'],b[2]['D_first'])
    def test_source_lifetime_credit_waits_until_first_output(self):
        rows=replay([0,0,0],[1,1,1],[4,4,4],[2,2,2],credit=1)
        self.assertEqual(rows[1]['P_start'],rows[0]['D_first'])
        self.assertGreater(rows[1]['D_start'],rows[1]['P_finish'])
    def test_no_future_arrival_dispatch(self):
        row=replay([10],[1],[1],[1])[0]
        self.assertEqual(row['P_start'],10)
        self.assertEqual(row['TTFT'],2)


if __name__=='__main__':unittest.main()
