"""Synthetic interval tests. These fixtures are not experiment evidence."""
import csv,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .envelope_analysis import physical_bound


class EnvelopeTests(unittest.TestCase):
    def test_cross_actor_transfer_and_graph_internal_idle_remain_occupied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'raw'/'fixture';folder.mkdir(parents=True)
            fields=('kind','start_ns','end_ns','device','src_device','dst_device')
            for role,device,rows in (
                ('prefill',0,[('kernel',100,200,0,-1,-1),('memcpy',400,500,0,0,1)]),
                ('decode',1,[('graph',100,300,1,-1,-1),('kernel',600,700,1,-1,-1)])):
                meta=dict(start=dict(host_midpoint=0.,cupti_ns=0),stop=dict(host_midpoint=1.,cupti_ns=1000),
                    status=0,counters=dict(invalid=0,dropped=0),graph_enclosures=role=='decode',
                    model_cuda_device_index=device)
                (folder/f'physical_activity_{role}.json').write_text(json.dumps(meta))
                with (folder/f'physical_activity_{role}.csv').open('w') as f:
                    writer=csv.writer(f);writer.writerow(fields);writer.writerows(rows)
            with patch('poc_flashvep.prefill_heavy_decode_safe.envelope_analysis.ROOT',root):
                value=physical_bound('fixture',padding_ms=0)
            self.assertAlmostEqual(value['prefill']['occupied_union_seconds'],.1)
            self.assertAlmostEqual(value['decode']['occupied_union_seconds'],.4)
            self.assertAlmostEqual(value['decode']['complement_seconds'],.6)
            self.assertFalse(value['decode']['exact_kernel_union'])
            self.assertEqual(value['decode']['complete_D_kernel_idle_distribution'],'NOT_MEASURED')


if __name__=='__main__':unittest.main()
