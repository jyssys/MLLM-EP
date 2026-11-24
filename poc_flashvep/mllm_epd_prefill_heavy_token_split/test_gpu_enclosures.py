"""CUDA-event observation must not synchronize or skip a real step."""
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class GPUEnclosureTest(unittest.TestCase):
    def test_real_calls_and_return_values_survive_without_per_step_sync(self):
        import torch
        from vllm.v1.core.sched.scheduler import Scheduler
        from . import gpu_enclosures as module
        state=dict(tick=0,syncs=0,calls=0)
        class Event:
            def __init__(self,enable_timing):pass
            def record(self):state['tick']+=1;self.tick=state['tick']
            def synchronize(self):state['syncs']+=1
            def elapsed_time(self,other):return float(other.tick-self.tick)
        output=SimpleNamespace(num_scheduled_tokens={'r':128},total_num_scheduled_tokens=128,
            scheduled_encoder_inputs={})
        request=SimpleNamespace(num_computed_tokens=0,num_prompt_tokens=128,status='WAITING')
        scheduler=SimpleNamespace(waiting=[request],running=[])
        def schedule(*args,**kwargs):return output
        class Runner:
            def execute_model(self,*args,**kwargs):state['calls']+=1;return 'original'
        runner=Runner()
        with TemporaryDirectory() as folder,patch.object(module,'ROOT',Path(folder)), \
                patch.dict(os.environ,MLLM_EPD_ROLE='prefill'), \
                patch.object(torch.cuda,'Event',Event),patch.object(torch.cuda,'synchronize') as sync, \
                patch.object(Scheduler,'schedule',schedule):
            observer=module.GPUEnclosures(runner,'unit_original_calls')
            observer.begin()
            for _ in range(2):
                self.assertIs(Scheduler.schedule(scheduler),output)
                self.assertEqual(runner.execute_model(output),'original')
            self.assertEqual(state['calls'],2)
            self.assertEqual(state['syncs'],1)
            self.assertEqual(sync.call_count,1)
            result=observer.close()
            self.assertEqual(result['events'],2)
            self.assertEqual(state['syncs'],2)
            self.assertIs(Scheduler.schedule,schedule)
            self.assertNotIn('execute_model',vars(runner))
            with (observer.folder/'forward_enclosures.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertTrue(all(float(r['GPU_enclosure_ms'])>0 for r in rows))


if __name__=='__main__':unittest.main()
