"""Do not convert physical window overlap or empty engine events into useful work."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .suffix_quantum_cost_report import suffix_events,physical_fit


class SuffixEvidenceTest(unittest.TestCase):
    def test_empty_schedule_events_remain_separate(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[dict(index=0,scheduled_requests=1,scheduled_tokens=128,GPU_enclosure_ms=3,host_begin_ns=0,host_end_ns=3000000),
                dict(index=1,scheduled_requests=0,scheduled_tokens=0,GPU_enclosure_ms=.4,host_begin_ns=0,host_end_ns=400000)]
            with (root/'forward_enclosures.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            (root/'geometry.json').write_text(json.dumps(dict(rows=[dict(index=0,requests={'chatcmpl-real_trial:D-unique':128}),dict(index=1,requests={})])))
            real,empty=suffix_events(root,[dict(run='real_trial',suffix_tokens='128',input_tokens='2961')])
            self.assertEqual(len(real),1);self.assertEqual(len(empty),1)
            self.assertEqual(real[0]['scheduled_query_tokens'],128)
            self.assertAlmostEqual(real[0]['GPU_enclosure_ms'],3.)

    def test_fit_has_different_count_and_duration_denominators(self):
        value=physical_fit([1,2,100],10)
        self.assertAlmostEqual(value['fitting_window_fraction'],1/3)
        self.assertAlmostEqual(value['idle_duration_fraction_in_fitting_windows'],100/103)
        self.assertEqual(value['evidence'],'OFFLINE_ORACLE')
        self.assertIn('Not safe admission',value['scope'])
        with self.assertRaises(AssertionError):physical_fit([1,2],0)


if __name__=='__main__':unittest.main()
