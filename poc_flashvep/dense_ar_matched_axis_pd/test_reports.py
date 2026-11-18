"""Check draft labels and ratio formatting without launching GPU work."""
import unittest
from .report_drafts import interval,number,table


class ReportTests(unittest.TestCase):
    def test_missing_ratio_is_not_zero(self):
        text=interval(dict(ratio=None,ratio_CI95=None))
        self.assertIn('NOT_MEASURED',text)
        self.assertNotIn('0.000000',text)

    def test_interval_keeps_both_bounds(self):
        self.assertEqual(interval(dict(ratio=1.02,ratio_CI95=[.98,1.08])),
            '1.020000 [0.980000, 1.080000]')

    def test_zero_is_a_real_number(self):
        self.assertEqual(number(0),'0.000')
        self.assertEqual(number(None),'NOT_MEASURED')

    def test_table_uses_commonmark_separator(self):
        self.assertEqual(table(['A','B'],[[1,2]]),'| A | B |\n| --- | --- |\n| 1 | 2 |')


if __name__=='__main__':
    unittest.main()
