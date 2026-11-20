import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import figures


class FigureEligibilityTests(unittest.TestCase):
    def test_one_confirmed_point_has_visible_error_bar(self):
        from unittest.mock import Mock
        ax=Mock();figures.draw_ci_points(ax,[1.65,2.35],[float('nan'),2.],[float('nan'),4.],'blue')
        args,kwargs=ax.errorbar.call_args
        self.assertEqual(args[0].tolist(),[2.35]);self.assertEqual(args[1].tolist(),[3.])
        self.assertEqual(kwargs['yerr'].tolist(),[1.])

    def test_failed_start_excludes_cell_not_only_unfavorable_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'analysis').mkdir()
            for index,passed in enumerate((True,False,True)):
                name=f'x{index}';folder=root/'raw'/name;folder.mkdir(parents=True)
                (folder/'configuration.json').write_text(json.dumps(dict(method='Native',offered_QPS=2.35)))
                (root/'analysis'/f'{name}_metrics.json').write_text(json.dumps(dict(name=name)))
                (root/'analysis'/f'{name}_quality_gate.json').write_text(json.dumps(dict(CI_passed=passed)))
            with patch.object(figures,'ROOT',root):groups,excluded=figures.measured_group('*_metrics.json')
            self.assertFalse(groups)
            self.assertEqual(len(excluded),1)
            self.assertEqual(len(excluded[0]['names']),3)
