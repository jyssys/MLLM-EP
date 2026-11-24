"""Preserve empty-schedule engine work instead of assigning it to prefill."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .prefix_quantum_cost import prefix_events


class PrefixCostEventsTest(unittest.TestCase):
    def fixture(self,folder,rows,geometry):
        with (folder/'forward_enclosures.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (folder/'geometry.json').write_text(json.dumps(dict(rows=geometry)))

    def test_real_request_and_two_empty_control_steps_remain_separate(self):
        with TemporaryDirectory() as temporary:
            folder=Path(temporary)
            rows=[dict(index=0,scheduled_requests=1,scheduled_tokens=256,encoder_items=0,GPU_enclosure_ms=28.795959),
                dict(index=1,scheduled_requests=0,scheduled_tokens=0,encoder_items=0,GPU_enclosure_ms=.390625),
                dict(index=2,scheduled_requests=0,scheduled_tokens=0,encoder_items=0,GPU_enclosure_ms=.298462)]
            self.fixture(folder,rows,[dict(index=0,requests={'actual-profile-q256':256}),
                dict(index=1,requests={}),dict(index=2,requests={})])
            real,control=prefix_events(folder,'E',[128,256,512,1024,2048])
            self.assertEqual(len(real),1);self.assertEqual(len(control),2)
            self.assertEqual(real[0]['quantum'],256)
            self.assertAlmostEqual(sum(r['GPU_enclosure_ms'] for r in control),.689087)

    def test_nonempty_zero_or_inconsistent_geometry_is_still_rejected(self):
        for count,tokens,requests in [(1,0,{'bad':0}),(1,512,{'bad':256}),(0,256,{})]:
            with self.subTest(count=count,tokens=tokens),TemporaryDirectory() as temporary:
                folder=Path(temporary)
                self.fixture(folder,[dict(index=0,scheduled_requests=count,scheduled_tokens=tokens,
                    encoder_items=0,GPU_enclosure_ms=1)], [dict(index=0,requests=requests)])
                with self.assertRaises(AssertionError):prefix_events(folder,'E',[128,256,512,1024,2048])


if __name__=='__main__':unittest.main()
