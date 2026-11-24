"""The new control cannot replace Main100 or reuse an image/document from it as fresh holdout."""
import unittest


class WholeContractTest(unittest.TestCase):
    def test_fresh_holdout_is_context_disjoint_and_not_selected_by_outputs(self):
        import json
        from .whole_deflection_control import fresh_holdout
        from .common import ROOT,digest
        from .dataset_build import keys
        rows,audit=fresh_holdout()
        main=ROOT/'data/main100_manifest_template_v2.json'
        old=json.loads(main.read_text())['requests']
        calibration=json.loads((ROOT/'data/calibration_manifest_template_v2.json').read_text())['requests']
        excluded=set().union(*(keys(r) for r in old+calibration))
        self.assertEqual(len(rows),50)
        self.assertGreaterEqual(audit['available_context_disjoint_requests'],50)
        self.assertTrue(all(not keys(r)&excluded for r in rows))
        self.assertTrue(audit['no_output_or_performance_filter'])
        self.assertTrue(audit['not_a_new_Main100'])
        self.assertEqual(digest(main),'910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1')

    def test_each_method_occupies_each_order_position_once(self):
        from .whole_deflection_control import METHODS,orders
        schedule=orders();self.assertEqual(schedule,orders())
        self.assertEqual(len(schedule),3)
        for position in range(3):self.assertEqual(sorted(row[position] for row in schedule),sorted(METHODS))


if __name__=='__main__':unittest.main()
