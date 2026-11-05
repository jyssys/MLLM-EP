import unittest
from . import report


class ReportRegressionTests(unittest.TestCase):
    def test_global_artifact_loader_is_not_shadowed_by_a_load_label(self):
        self.assertNotIn('load', report.main.__code__.co_varnames)


if __name__ == '__main__':
    unittest.main()
