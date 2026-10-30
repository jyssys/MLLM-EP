"""Keep uncertainty and missing-data handling separate from GPU measurements."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from .publication import fmt,table,paired,patch_report

class Reporting(unittest.TestCase):
    def test_report_edits_use_apply_patch(self):
        with tempfile.TemporaryDirectory(prefix='pd_report_patch_test_') as temp,patch('poc_flashvep.adaptive_loop_token_pd_2gpu.publication.REPO',Path(temp)):
            patch_report('unit_test_report.md','A measured value is unavailable.\n')
            p=Path(temp)/'poc_flashvep/reports/unit_test_report.md'
            self.assertEqual(p.read_text(),'A measured value is unavailable.\n')
            patch_report('unit_test_report.md','No GPU result exists in this unit test.\n')
            self.assertEqual(p.read_text(),'No GPU result exists in this unit test.\n')
    def test_missing_is_not_zero(self):
        self.assertEqual(fmt(None),'NOT_MEASURED')
        self.assertEqual(fmt('NOT_AVAILABLE'),'NOT_MEASURED')
        self.assertEqual(fmt(0),'0.00')
    def test_paired_independent_starts(self):
        rows=[]
        for start in (1,2,3):
            for arm in ('Loop_Dynamic','DP','PD_G1','Loop_Static','Token_Dynamic'):
                rows.append(dict(model='ouro',workload='W1',arm=arm,phase='repeat',start=str(start),
                    status='COMPLETE',SLO_goodput_per_s='2' if arm=='Loop_Dynamic' else '1'))
        with tempfile.TemporaryDirectory(prefix='pd_reporting_test_') as temp,patch('poc_flashvep.adaptive_loop_token_pd_2gpu.publication.ROOT',Path(temp)):
            result=paired(rows)
        selected=next(r for r in result if r['model']=='ouro' and r['baseline']=='DP')
        self.assertEqual(selected['paired_independent_starts'],3)
        self.assertEqual(selected['relative_goodput_gain_percent'],100.)
        self.assertEqual(selected['CI95_low_percent'],100.)
        self.assertEqual(selected['CI95_high_percent'],100.)
    def test_one_start_has_no_qualifying_interval(self):
        rows=[dict(model='ouro',workload='W1',arm=arm,phase='repeat',start='1',status='COMPLETE',SLO_goodput_per_s='1')
              for arm in ('Loop_Dynamic','DP')]
        with tempfile.TemporaryDirectory(prefix='pd_reporting_test_') as temp,patch('poc_flashvep.adaptive_loop_token_pd_2gpu.publication.ROOT',Path(temp)):
            selected=next(r for r in paired(rows) if r['model']=='ouro' and r['baseline']=='DP')
        self.assertEqual(selected['CI95_low_percent'],'NOT_AVAILABLE')

if __name__=='__main__':unittest.main()
