"""CPU tests. These tests do not establish GPU or model correctness."""
import unittest
from types import SimpleNamespace
from .staged import install


class FrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install(dict(state_capacity_tokens=4096))

    def test_absolute_rows_do_not_wrap(self):
        from vllm_rlt.core.scheduler import Scheduler
        from vllm_rlt.request import RequestPhase
        s=object.__new__(Scheduler)
        s.config=SimpleNamespace(prefill_chunk_size=128)
        s.total_ut_steps=4
        r=SimpleNamespace(prompt_token_ids=list(range(1536)),num_preluded_tokens=1536,
                          arc_export_cut=2,prefill_depth_ranges=[[0,1536],[0,1536],[0,0],[0,0]])
        ranges=[]
        for start in range(0,1536,128):
            t=s._make_wavefront_task(r,RequestPhase.PREFILLING_PRELUDE,128,start=start)
            self.assertEqual(t.prefill_hidden_row_offset,start)
            ranges.extend(range(t.prefill_hidden_row_offset,t.prefill_hidden_row_offset+t.token_count))
        self.assertEqual(ranges,list(range(1536)))
        self.assertIsNone(s._make_wavefront_task(r,RequestPhase.PREFILLING_RECURRENT,128,start=0,depth=2))
        t=s._make_wavefront_task(r,RequestPhase.PREFILLING_RECURRENT,128,start=0,depth=1)
        self.assertEqual(t.token_count,128)
        self.assertTrue(s._prelude_window_open(SimpleNamespace(num_preluded_tokens=512,prompt_token_ids=list(range(1536)))))

    def test_layer_action_space_contains_loop_boundaries(self):
        for depth,physical in ((4,24),(32,4)):
            logical=set(range(depth*physical+1))
            loops={r*physical for r in range(depth+1)}
            self.assertLessEqual(loops,logical)
            self.assertEqual(len(loops),depth+1)

    def test_new_cut_is_committed_before_commands(self):
        from .worker import install_metadata
        engine=SimpleNamespace(_send=lambda *a,**k:(a,k),transfers={'T':SimpleNamespace(request=SimpleNamespace(request_id='R'))})
        install_metadata(engine,4);engine._arc_cuts['R']=2
        for kind in ('reserve','prefill','commit'):
            args,fields=engine._send('P',kind,tid='T',peer='D')
            self.assertEqual(fields['arc_cut'],2)


if __name__=='__main__':unittest.main()
