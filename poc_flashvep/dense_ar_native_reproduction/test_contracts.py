"""CPU-only checks for benchmark accounting and execution options."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .engine_options import options
from . import metrics
from .remote_token_bound import bounded_count


class Contracts(unittest.TestCase):
    def test_remote_first_token_is_not_imported(self):
        for prompt in (1910,8192,16384,38912):
            self.assertEqual(bounded_count(prompt+1,prompt,0),prompt)
            self.assertEqual(bounded_count(prompt+1,prompt,prompt),0)
            self.assertEqual(bounded_count(prompt+1,prompt,16),prompt-16)
    def test_scheduler_and_connector(self):
        for role in ('single','replica0','replica1','prefill','decode'):
            cfg=options(dict(token_budget=8192,max_num_seqs=1024),role)
            self.assertEqual(cfg['max_num_batched_tokens'],8192)
            self.assertEqual(cfg['max_num_seqs'],1024)
            self.assertFalse(cfg['enable_prefix_caching'])
            self.assertTrue(cfg['enable_chunked_prefill'])
            self.assertEqual(cfg['kv_cache_memory_bytes'],42*1024**3)
            self.assertEqual('kv_transfer_config' in cfg,role in ('prefill','decode'))

    def test_sensitivity_is_explicit(self):
        for b in (4096,8192,16384):
            for s in (256,1024):
                cfg=options(dict(token_budget=b,max_num_seqs=s))
                self.assertEqual((cfg['max_num_batched_tokens'],cfg['max_num_seqs']),(b,s))

    def test_request_SLO_not_percentile_SLO(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'raw/unit';folder.mkdir(parents=True);(root/'analysis').mkdir()
            def write(path,value):path.write_text(json.dumps(value))
            write(folder/'configuration.json',dict(count=2,method='Native',offered_QPS=1))
            write(folder/'window.json',dict(start=0,end=10,last_arrival=1))
            rows=[dict(prompt_id='x',task='qa',arrival=0,TTFT_ms=100,TPOT_ms=20,E2E_ms=140,output_tokens=3,
                token_events=[dict(time=.1,count=1),dict(time=.12,count=2),dict(time=.14,count=3)]),
                dict(prompt_id='y',task='summary',arrival=1,TTFT_ms=100,TPOT_ms=25,E2E_ms=150,output_tokens=3,
                token_events=[dict(time=1.1,count=1),dict(time=1.14,count=2),dict(time=1.15,count=3)])]
            (folder/'requests.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with patch.object(metrics,'ROOT',root):result=metrics.summarize('unit')
            self.assertEqual(result['Joint_SLO_attainment'],.5)
            self.assertEqual(result['SLO_qualified_output_tokens_per_second'],.3)
            self.assertEqual(result['generated_output_tokens_per_second'],.6)
            self.assertFalse(result['sustainable_capacity'])


if __name__=='__main__':unittest.main()
