"""Check kernel-union clock mapping without CUDA or live worker state."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import online_profile


class ActivityAnalysisTests(unittest.TestCase):
    def test_merge_does_not_sum_overlaps(self):
        self.assertEqual(online_profile.merge([(3,5),(1,4),(7,8)]),[[1,5],[7,8]])

    def test_common_clock_window_and_edge_trim(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'raw'/'synthetic';folder.mkdir(parents=True)
            (folder/'window.json').write_text(json.dumps(dict(start=100.)))
            stamps=dict(left=dict(host_lower_ns=100000000000,host_upper_ns=100000010000,cupti_ns=1000000000),
                right=dict(host_lower_ns=110000000000,host_upper_ns=110000010000,cupti_ns=11000000000),scope='Synthetic test only')
            for role in ('prefill','decode'):
                (folder/f'activity_window_{role}.json').write_text(json.dumps(stamps))
                (folder/f'scheduler_queues_{role}.json').write_text(json.dumps(dict(rows=[dict(time=100.,waiting_total=1)])))
                activity=folder/f'activity_{role}';activity.mkdir()
                (activity/'activity_counters.json').write_text(json.dumps(dict(passed=True,counters=dict(dropped=0,invalid=0))))
                with (activity/'kernels.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['kind','start_ns','end_ns'])
                    for a,b in ((3,5),(4,6),(8,10)):writer.writerow(['kernel',a*10**9,b*10**9])
            with patch.object(online_profile,'ROOT',root):result=online_profile.kernel_window('synthetic')
            self.assertAlmostEqual(result['roles']['decode']['window_seconds'],6.)
            self.assertAlmostEqual(result['roles']['decode']['kernel_union_seconds'],4.)
            self.assertAlmostEqual(sum(b['kernel_union_seconds'] for b in result['roles']['decode']['kernel_union_bins_1s']),4.)
            self.assertAlmostEqual(result['roles']['decode']['kernel_busy_fraction'],2/3)
            self.assertAlmostEqual(result['roles']['decode']['timestamp_bracket_max_ms'],.01)


if __name__=='__main__':unittest.main()
