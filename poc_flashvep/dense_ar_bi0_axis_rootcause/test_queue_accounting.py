import unittest
from .queue_accounting import decompose


class QueueAccountingTest(unittest.TestCase):
    def test_fifo_chain_credit_and_host_are_distinct(self):
        row=decompose(arrival=0,previous_ready_received=3,credit_return=5,dispatch=5.2,prepare=5.3)
        self.assertAlmostEqual(row['previous_source_chain_ms'],3000)
        self.assertAlmostEqual(row['direct_credit_block_ms'],2000)
        self.assertAlmostEqual(row['eligible_to_host_dispatch_ms'],200)
        self.assertAlmostEqual(row['dispatch_to_first_prepare_ms'],100)
        self.assertAlmostEqual(row['sum_ms'],5300)

    def test_credit_is_not_binding(self):
        row=decompose(4,5,2,5.1,5.2)
        self.assertEqual(row['direct_credit_block_ms'],0)


if __name__=='__main__':unittest.main()
