"""Equivalent QPS JSON types do not permit any actual trace change."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .common import write
from .native_common_pilot_report import same_arrival_contract


class ArrivalContractTest(unittest.TestCase):
    def test_json_integer_and_float_qps_keep_identical_exact_arrivals(self):
        with TemporaryDirectory() as folder:
            left=Path(folder)/'a.json';right=Path(folder)/'b.json'
            base=dict(QPS=2,seed=2,serial=False,trace=[dict(request_id='r',offset_seconds=.123456789)])
            write(left,base);write(right,dict(base,QPS=2.0))
            self.assertTrue(same_arrival_contract(left,right))
            for changed in [dict(base,seed=3),dict(base,QPS=3),
                dict(base,trace=[dict(request_id='other',offset_seconds=.123456789)]),
                dict(base,trace=[dict(request_id='r',offset_seconds=.123456789001)])]:
                write(right,changed)
                self.assertFalse(same_arrival_contract(left,right))


if __name__=='__main__':unittest.main()
