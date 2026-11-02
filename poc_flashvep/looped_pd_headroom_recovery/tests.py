"""CPU regressions for new isolation and profiling helpers."""
import subprocess
import sys
import unittest
from .runtime import ProcessAdapter


class RuntimeTests(unittest.TestCase):
    def test_exact_interval_bins_match_the_original_union(self):
        import random
        from .interval_bins import bins
        from .profile_analysis import union
        rng=random.Random(71)
        kernels=[]
        for _ in range(400):
            start=rng.uniform(-1,7)
            kernels.append(dict(start_monotonic=start,end_monotonic=start+rng.uniform(0,.8)))
        for start,stop,percentage in bins(kernels,0,6.13,.1):
            reference=union([(max(start,k['start_monotonic']),min(stop,k['end_monotonic']))
                for k in kernels if k['end_monotonic']>start and k['start_monotonic']<stop])
            self.assertAlmostEqual(percentage,100*reference/(stop-start),places=10)
    def test_existing_writer_preserves_order_and_uses_background_fsync(self):
        import json,tempfile,threading
        from pathlib import Path
        from unittest.mock import patch
        from ..looped_pd_abc.telemetry import EvidenceWriter
        main=threading.get_ident();writers=[]
        with tempfile.TemporaryDirectory() as d,patch('os.fsync',side_effect=lambda fd:writers.append(threading.get_ident())):
            writer=EvidenceWriter();path=Path(d)/'records.jsonl'
            for i in range(12):writer.append(path,dict(index=i))
            writer.close()
            self.assertEqual([r['index'] for r in map(json.loads,path.read_text().splitlines())],list(range(12)))
            self.assertEqual(writer.records,writer.background_records)
            self.assertTrue(writers)
            self.assertTrue(all(v!=main for v in writers))
    def test_writer_routes_captured_harness_aliases_only(self):
        from types import SimpleNamespace as NS
        from .durable_evidence import route_aliases
        old=lambda:None;new=lambda:None;other=lambda:None
        a=NS(append=old,write=old);b=NS(append=other,write=other)
        route_aliases([('poc_flashvep.old_worker',a),('external_module',b)],old,new,old,new)
        self.assertIs(a.append,new);self.assertIs(a.write,new)
        self.assertIs(b.append,other)
    def test_completed_but_unretired_rows_are_not_compute_ready(self):
        from types import SimpleNamespace as NS
        from .runtime import all_submitted_ids
        a=NS(ready=lambda:True,batch=NS(requests=[NS(request=NS(request_id='completed'))]))
        b=NS(ready=lambda:False,batch=NS(requests=[NS(request=NS(request_id='pending'))]))
        self.assertEqual(all_submitted_ids([a,b]),{'completed','pending'})
    def test_profile_cleanup_rejects_unregistered_or_reused_pid(self):
        import tempfile,json,os
        from pathlib import Path
        from .run import registered_profile_targets
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'stage_owner_prefill.json').write_text(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),uid=os.getuid(),payload=str(root/'fake.pickle'))))
            self.assertEqual(registered_profile_targets(root),[])
    def test_packed_descriptors_match_legacy_after_coalescing(self):
        import random,sys
        from .common import RLT
        sys.path.insert(0,str(RLT))
        from .descriptor import replay,valid_segments
        from ..looped_pd_abc.boundary import valid_segments as old
        rng=random.Random(9)
        for layers,depths,core,coda in [(24,4,list(range(24)),[]),(8,32,[2,3,4,5],[6,7])]:
            for length in (1,16,19,32,35):
                pages=(length+15)//16
                allocated=rng.sample(range(400),pages*depths)
                destination=rng.sample(range(500),pages*depths)
                tables=[allocated[d*pages:(d+1)*pages] for d in range(depths)]
                targets=[destination[d*pages:(d+1)*pages] for d in range(depths)]
                a=dict(block_size=16,num_layers=layers,layer_bytes=160,block_bytes=160*layers,
                    token_bytes=10,key_ptr=10000000,value_ptr=20000000,device=0,hidden_ptr=50000000,hidden_bytes=40)
                b=dict(a,key_ptr=60000000,value_ptr=80000000,hidden_ptr=90000000,device=1)
                for start,end in ((0,length),(1,length),(0,min(16,length))):
                    value=dict(source_info=a,target_info=b,source_tables=tables,target_tables=targets,start=start,end=end,
                        recurrent_layers=core,coda_layers=coda,prompt_tokens=length,final=True,source_slot=0,target_slot=1)
                    self.assertEqual(replay(value,old)[0],replay(value,valid_segments)[0])
        value['source_tables'][1]=value['source_tables'][0]
        with self.assertRaisesRegex(ValueError,'non-aliased'):
            replay(value,valid_segments)
    def test_complete_profile_cohort_is_not_a_response_stall(self):
        import tempfile,json
        from pathlib import Path
        from .run import measured_requests_complete
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'requests.jsonl').write_text('\n'.join(json.dumps(dict(warmup=w)) for w in (True,False,False))+'\n')
            self.assertTrue(measured_requests_complete(root,2))
            self.assertFalse(measured_requests_complete(root,3))
    def test_cleanup_tolerates_a_process_exit_race(self):
        from unittest.mock import patch
        from .run import belongs_to_session
        with patch('pathlib.Path.stat',side_effect=FileNotFoundError):
            self.assertFalse(belongs_to_session(123,123))
    def test_replica_ingress_sends_without_blocking_control(self):
        from types import SimpleNamespace as NS
        from .ingress import enqueue
        messages=[]
        clients=[NS(requests={},generations={},send=lambda kind,**fields:messages.append((kind,fields))) for _ in range(2)]
        pool=NS(clients=clients,assignment={})
        for i in range(4):enqueue(pool,str(i),[1,2,3],NS(max_tokens=2))
        self.assertEqual(pool.assignment,{'0':0,'1':1,'2':0,'3':1})
        self.assertEqual(len(messages),4)
        self.assertTrue(all(kind=='add' for kind,fields in messages))
        self.assertEqual(sum(len(c.requests) for c in clients),4)
    def test_union_does_not_sum_overlapping_kernel_times(self):
        from .profile_analysis import union
        self.assertEqual(union([(0,10),(2,4),(8,14),(20,23)]),17)
        self.assertEqual(union([]),0)
    def test_vector_bytecheck_preserves_legacy_stream(self):
        import torch
        from types import SimpleNamespace as NS
        from .bytecheck import fingerprint
        from ..looped_pd_abc.verify import fingerprint as legacy
        keys=torch.arange(40*8*4*2*3,dtype=torch.float32).reshape(40,8,4,2,3).to(torch.bfloat16)
        cache=NS(num_layers=8,max_loops=4,block_size=4,key_cache=keys,value_cache=keys+1)
        allocation=NS(block_tables=tuple(tuple(range(d*7,d*7+6)) for d in range(4)))
        cache._get_allocation=lambda tid:allocation
        connector=NS(cache=cache,engine=NS(model=NS(recurrent_kv_layers=(2,3,4,5),coda_kv_layers=(6,7))),
            hidden=torch.arange(24,dtype=torch.bfloat16).reshape(3,8))
        for n in (1,3,4,7,8,19):
            transfer=NS(tid='cpu',request=NS(prompt_token_ids=list(range(n))),slot=1)
            self.assertEqual(fingerprint(connector,transfer),legacy(connector,transfer))
        connector.cut=2
        self.assertEqual(fingerprint(connector,transfer),legacy(connector,transfer))

    def test_plane_selection_preserves_all_operand_bytes(self):
        import torch
        from .state_probe import plane
        tensor=torch.arange(11*4*3*5).reshape(11,4,3,5)
        blocks=torch.tensor([9,1,6,3])
        for layer in range(4):
            old=tensor.index_select(0,blocks)[:,layer]
            actual=plane(tensor,blocks,layer)
            self.assertTrue(torch.equal(old,actual))
            self.assertEqual(old.contiguous().numpy().tobytes(),actual.contiguous().numpy().tobytes())

    def test_process_adapter_matches_native_client(self):
        child=subprocess.Popen([sys.executable,'-c','pass'])
        adapted=ProcessAdapter(child)
        self.assertEqual(adapted.pid,child.pid)
        adapted.join(10)
        self.assertFalse(adapted.is_alive())
        self.assertEqual(adapted.exitcode,0)

    def test_actual_request_uses_generated_token_ids(self):
        from pathlib import Path
        source=Path('/home/esjung/external/looped-pd-abc-rlt/vllm_rlt/request.py').read_text()
        self.assertIn('generated_token_ids: list[int]',source)
        self.assertNotIn('num_output_tokens:',source)
        hook=Path(__file__).with_name('runtime.py').read_text()
        self.assertIn('len(r.generated_token_ids)',hook)
        self.assertNotIn('r.num_output_tokens',hook)

    def test_capture_is_selected_stage_only(self):
        from pathlib import Path
        source=Path(__file__).with_name('runtime.py').read_text()
        self.assertIn("stage_config.role != cfg.get('capture_role', 'prefill')",source)
        self.assertIn("'--capture-range=cudaProfilerApi'",source)


if __name__=='__main__':
    unittest.main()
