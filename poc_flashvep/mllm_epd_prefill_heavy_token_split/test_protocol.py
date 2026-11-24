"""CPU regressions for the actual installed multimodal renderer contract."""
import types
import unittest


class Items(dict):
    def get_all_counts(self):return {key:len(value) for key,value in self.items()}


def renderer_ids(cache_gb,request_counter,ids):
    from vllm.renderers.base import BaseRenderer
    mm=types.SimpleNamespace(mm_processor_cache_gb=cache_gb)
    model=types.SimpleNamespace(multimodal_config=mm)
    self=types.SimpleNamespace(model_config=model,
        config=types.SimpleNamespace(cache_config=types.SimpleNamespace(enable_prefix_caching=False)))
    self._validate_mm_uuids=types.MethodType(BaseRenderer._validate_mm_uuids,self)
    return BaseRenderer._process_mm_uuids(self,dict(image=['actual-image']*len(ids)),
        Items(image=['actual-image']*len(ids)),dict(image=ids),f'renderer0-mm-{request_counter}')


class ProtocolTest(unittest.TestCase):
    def test_cache_zero_reproduces_engine_local_identity(self):
        encoder=renderer_ids(0,12,['shared-image-UUID'])
        prefill=renderer_ids(0,7,['shared-image-UUID'])
        self.assertNotEqual(encoder,prefill)
        self.assertEqual(encoder,dict(image=['renderer0-mm-12-image-0']))

    def test_common_cache_one_preserves_cross_engine_identity(self):
        encoder=renderer_ids(1,12,['shared-image-UUID'])
        prefill=renderer_ids(1,7,['shared-image-UUID'])
        self.assertEqual(encoder,prefill)
        self.assertEqual(encoder,dict(image=['shared-image-UUID']))

    def test_request_unique_ids_do_not_reuse_across_requests(self):
        self.assertNotEqual(renderer_ids(1,1,['request-A-image-0']),
            renderer_ids(1,1,['request-B-image-0']))


class NumericalTest(unittest.TestCase):
    def test_feature_metrics_match_float64_reference(self):
        import torch
        from .numerical import embedding_metrics
        torch.manual_seed(17)
        a=torch.randn(1048576).bfloat16()
        b=(a.float()+.01*torch.randn_like(a.float())).bfloat16()
        result=embedding_metrics(a,b)
        x,y=a.double(),b.double()
        expected=float((x-y).norm()/y.norm())
        cosine=float((x*y).sum()/(x.norm()*y.norm()))
        self.assertAlmostEqual(result['relative_L2'],expected,places=8)
        self.assertAlmostEqual(result['cosine'],cosine,places=8)
        self.assertTrue(result['finite'])

    def test_equal_and_zero_features(self):
        import torch
        from .numerical import embedding_metrics
        for value in (torch.zeros(1024),torch.ones(1024)):
            r=embedding_metrics(value,value)
            self.assertEqual(r['relative_L2'],0)
            self.assertEqual(r['cosine'],1)
            self.assertTrue(r['strict'])


class StrictECTest(unittest.TestCase):
    def test_missing_consumer_export_cannot_fall_back_to_vision(self):
        from tempfile import TemporaryDirectory
        from .strict_ec import StrictExampleConnector
        with TemporaryDirectory() as folder:
            connector=object.__new__(StrictExampleConnector)
            connector._storage_path=folder;connector._is_consumer=True
            with self.assertRaises(FileNotFoundError):connector.has_cache_item('missing')

    def test_producer_cache_miss_still_permits_original_encoding(self):
        from tempfile import TemporaryDirectory
        from .strict_ec import StrictExampleConnector
        with TemporaryDirectory() as folder:
            connector=object.__new__(StrictExampleConnector)
            connector._storage_path=folder;connector._is_consumer=False
            self.assertFalse(connector.has_cache_item('new-image'))

    def test_consumer_finds_official_file_path(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from .strict_ec import StrictExampleConnector
        with TemporaryDirectory() as folder:
            connector=object.__new__(StrictExampleConnector)
            connector._storage_path=folder;connector._is_consumer=True
            target=Path(folder)/'image';target.mkdir();(target/'encoder_cache.safetensors').touch()
            self.assertTrue(connector.has_cache_item('image'))


class KVProbeTest(unittest.TestCase):
    def test_decode_input_copies_require_explicit_correctness_opt_in(self):
        from .forward_probe import capture_input_allowed
        self.assertFalse(capture_input_allowed(True,False,0,16))
        self.assertTrue(capture_input_allowed(True,True,0,16))
        self.assertTrue(capture_input_allowed(False,False,0,16))
        self.assertFalse(capture_input_allowed(True,True,16,16))

    def test_logical_HND_indexing(self):
        import torch
        from .forward_probe import gather_prompt_kv
        cache=torch.arange(3*2*4*6).reshape(3,2,4,6)
        blocks=[2,0];length=7
        expected=torch.stack([cache[blocks[t//4],:,t%4,:] for t in range(length)])
        self.assertTrue(torch.equal(gather_prompt_kv(cache,blocks,length),expected))

    def test_noncontiguous_logical_NHD_indexing(self):
        import torch
        from .forward_probe import gather_prompt_kv
        physical=torch.arange(3*4*2*6).reshape(3,4,2,6)
        logical=physical.permute(0,2,1,3);blocks=[1,2];length=6
        expected=torch.stack([physical[blocks[t//4],t%4,:,:] for t in range(length)])
        self.assertTrue(torch.equal(gather_prompt_kv(logical,blocks,length),expected))


class RepresentativeSubsetTest(unittest.TestCase):
    def test_replay_reader_streams_counts_and_rejects_duplicates(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from .client import load_replay_counts
        with TemporaryDirectory() as folder:
            path=Path(folder)/'reference.jsonl'
            rows=[dict(request_id='a',status='MEASURED_ONLINE',output_tokens=16,chunks=[{'unused':'large'}]),
                  dict(request_id='b',status='MEASURED_ONLINE',output_tokens=192),
                  dict(request_id='failed',status='FAILED_EXECUTION')]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with patch.object(Path,'read_text',side_effect=AssertionError('Do not retain complete reference traces')):
                self.assertEqual(load_replay_counts(path),{'a':16,'b':192})
            with path.open('a') as stream:stream.write('\n'+json.dumps(rows[0]))
            with self.assertRaises(AssertionError):load_replay_counts(path)

    def test_serial_duration_does_not_use_the_future_open_loop_anchor(self):
        from .client import duration_anchor
        rows=[dict(scheduled_arrival=100.,finish=100.2)]
        self.assertEqual(duration_anchor(True,101.,rows),100.)
        self.assertEqual(duration_anchor(False,101.,rows),101.)
        self.assertAlmostEqual(rows[0]['finish']-duration_anchor(True,101.,rows),.2)

    def test_selection_is_deterministic_and_preserves_source_split(self):
        from .representative_subset import select
        rows=[];outputs={};discovery=set()
        for source in ['A','B']:
            for split in ['discovery','heldout']:
                for j in range(10):
                    key=f'{source}_{split}_{j}'
                    rows.append(dict(request_id=key,source=source,total_input_tokens=100+j*20,
                        visual_tokens=80+j*20,text_tokens=13,special_tokens=7,image_count=j%3+1,page_count=None))
                    outputs[key]=dict(output_tokens=j*3+2)
                    if split=='discovery':discovery.add(key)
        a,proof,_=select(rows,outputs,discovery,count=10,seed=12,iterations=100,starts=1)
        b,_,_=select(rows,outputs,discovery,count=10,seed=12,iterations=100,starts=1)
        self.assertEqual(a,b)
        self.assertEqual(len(set(a)),10)
        self.assertEqual(sum(rows[i]['request_id'] in discovery for i in a),5)
        self.assertEqual(sum(q['count'] for q in proof['source_split_allocation']),10)

    def test_default_manifest_requires_frozen_hash(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from .common import digest
        from .client import resolve_manifest
        with TemporaryDirectory() as folder:
            path=Path(folder)/'main100.json';path.write_text('{"requests": []}')
            active=Path(folder)/'active.json'
            active.write_text(json.dumps(dict(main_manifest=str(path),main_SHA256=digest(path))))
            self.assertEqual(resolve_manifest(None,active),path)
            path.write_text('{"requests": [1]}')
            with self.assertRaises(AssertionError):resolve_manifest(None,active)

    def test_per_request_replay_keeps_individual_output_counts(self):
        from .client import resolve_output_count
        counts={'A':16,'B':128}
        self.assertEqual(resolve_output_count(counts,dict(request_id='A')),16)
        self.assertEqual(resolve_output_count(counts,dict(request_id='B')),128)
        self.assertIsNone(resolve_output_count(None,dict(request_id='A')))
        self.assertEqual(resolve_output_count(64,dict(request_id='B')),64)
        with self.assertRaises(KeyError):resolve_output_count(counts,dict(request_id='missing'))


class FrontendConfigTest(unittest.TestCase):
    def test_official_renderer_configuration_constraint_is_preserved(self):
        from pathlib import Path
        from .servers import command
        for cache,workers in [(1,1),(0,1),(0,2)]:
            args=command('encoder',18840,Path('/dev/shm/fixture'),19840,
                         mm_cache_gb=cache,renderer_workers=workers)
            self.assertEqual(args[args.index('--mm-processor-cache-gb')+1],str(cache))
            self.assertEqual(args[args.index('--renderer-num-workers')+1],str(workers))
            self.assertEqual(args[args.index('--max-num-batched-tokens')+1],'114688')
        with self.assertRaises(AssertionError):
            command('encoder',18840,Path('/dev/shm/fixture'),19840,mm_cache_gb=1,renderer_workers=2)


class IntervalTest(unittest.TestCase):
    def test_union_removes_kernel_overlap(self):
        import numpy as np
        from .profile_analysis import union,duration
        x=union([[2,4],[1,3],[3,3.5],[7,9]],0,8)
        self.assertTrue(np.array_equal(x,[[1,4],[7,8]]))
        self.assertEqual(duration(x),4)

    def test_idle_intersection_and_empty_activity(self):
        import numpy as np
        from .profile_analysis import union,complement,intersection
        idle=complement([[1,3],[7,8]],0,10)
        self.assertTrue(np.array_equal(idle,[[0,1],[3,7],[8,10]]))
        self.assertTrue(np.array_equal(intersection(idle,union([[2,5],[9,12]],0,10)),[[3,5],[9,10]]))
        self.assertTrue(np.array_equal(complement([],0,10),[[0,10]]))


class ObserverHookTest(unittest.TestCase):
    def test_execute_hook_observes_a_prior_instance_override(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        from .worker_observer import Collector
        class Runner:
            def execute_model(self,output):return 'unchanged-original-result'
        runner=Runner();runner.execute_model=runner.execute_model
        original_slot=runner.execute_model
        schedule=Scheduler.schedule;execute=Runner.execute_model
        observer=object.__new__(Collector)
        observer.runner_class=Runner;observer.runner=runner
        observer.rows=[];observer.calls=[];observer.originals=[]
        observer.state_interval_ns=0
        observer.last_schedule_sample_ns=observer.last_execute_sample_ns=0
        output=types.SimpleNamespace(total_num_scheduled_tokens=1,
            num_scheduled_tokens={'request':1},scheduled_encoder_inputs={})
        try:
            observer.install_state_hooks()
            self.assertEqual(runner.execute_model(output),'unchanged-original-result')
            self.assertEqual(len(observer.calls),1)
        finally:
            Scheduler.schedule=schedule;Runner.execute_model=execute
            runner.execute_model=original_slot

    def test_official_app_registers_middleware_before_stack_bootstrap(self):
        from fastapi.testclient import TestClient
        from vllm.entrypoints.openai import api_server
        from . import host_timeline
        previous=host_timeline.API_RECORDER
        host_timeline.API_RECORDER=host_timeline.APITimeline()
        try:
            parser=api_server.make_arg_parser(api_server.FlexibleArgumentParser())
            args=parser.parse_args(['--model','Qwen/Qwen2.5-VL-7B-Instruct','--middleware',
                'poc_flashvep.mllm_epd_prefill_heavy_token_split.host_timeline.APITimelineMiddleware'])
            app=api_server.build_app(args,supported_tasks=('generate',))
            @app.get('/campaign/cpu_middleware_test')
            async def probe():return {'unchanged_route':True}
            # No EngineClient lifespan is started in this ASGI path unit test.
            response=TestClient(app).get('/campaign/cpu_middleware_test')
            self.assertEqual(response.status_code,200)
            self.assertTrue(response.json()['unchanged_route'])
        finally:host_timeline.API_RECORDER=previous


class ClientHostObserverTest(unittest.IsolatedAsyncioTestCase):
    async def test_callbacks_restore_and_original_append_remains(self):
        import gc
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from . import client_host_observer as module
        previous=list(gc.callbacks)
        with TemporaryDirectory() as folder, patch.object(module,'ROOT',Path(folder)):
            observer=module.ClientHostObserver('cpu_probe',interval=.001)
            await observer.begin()
            try:
                gc.collect()
                target=Path(folder)/'original.jsonl'
                observer.record_append(target,dict(request_id='request',runtime_request_id='runtime'))
                await __import__('asyncio').sleep(.01)
            finally:
                await observer.close()
            self.assertEqual(gc.callbacks,previous)
            self.assertIn('"request_id": "request"',target.read_text())
            self.assertTrue(any(r['kind']=='gc' for r in observer.rows))
            self.assertTrue(any(r['kind']=='heartbeat' for r in observer.rows))
            self.assertTrue(any(r['kind']=='raw_append' for r in observer.rows))


class ClientHeapTest(unittest.TestCase):
    def test_freeze_restores_normal_gc_policy(self):
        import gc
        from .client_gc import StaticHeapFreeze
        self.assertEqual(gc.get_freeze_count(),0)
        before=(gc.isenabled(),gc.get_threshold())
        policy=StaticHeapFreeze()
        try:
            info=policy.begin()
            self.assertGreater(info['frozen_objects'],0)
            self.assertEqual((gc.isenabled(),gc.get_threshold()),before)
        finally:
            policy.close()
        self.assertEqual(gc.get_freeze_count(),0)
        self.assertEqual((gc.isenabled(),gc.get_threshold()),before)

    def test_an_existing_frozen_heap_is_not_modified(self):
        import gc
        from .client_gc import StaticHeapFreeze
        gc.freeze()
        try:
            before=gc.get_freeze_count()
            with self.assertRaises(AssertionError):StaticHeapFreeze().begin()
            self.assertEqual(gc.get_freeze_count(),before)
        finally:
            gc.unfreeze()


class APIHostPolicyTest(unittest.TestCase):
    def test_explicit_switch_interval_changes_only_the_requested_setting(self):
        import os
        import sys
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from . import common,server_entry
        before=sys.getswitchinterval()
        try:
            with TemporaryDirectory() as folder,patch.object(common,'ROOT',Path(folder)),\
                    patch.dict(os.environ,{'MLLM_EPD_API_SWITCH_INTERVAL':'.001'}):
                row=server_entry.api_host_policy()
                self.assertEqual(sys.getswitchinterval(),.001)
                self.assertEqual(row['actual_seconds'],.001)
                self.assertTrue(row['explicit'])
                self.assertEqual(row['pid'],os.getpid())
        finally:
            sys.setswitchinterval(before)


class APIHostScopesTest(unittest.TestCase):
    def test_original_serialization_bytes_and_context_are_preserved(self):
        from vllm.v1.serial_utils import MsgpackEncoder
        from .api_host_scopes import install,HTTP_REQUEST_ID
        obj={'input':[1,2,3],'request_id':'unchanged'}
        expected=MsgpackEncoder().encode(obj)
        recorder=types.SimpleNamespace(active=True,rows=[])
        originals=install(recorder)
        token=HTTP_REQUEST_ID.set('actual-request')
        try:
            actual=MsgpackEncoder().encode(obj)
            self.assertEqual(actual,expected)
            self.assertEqual(len(recorder.rows),1)
            self.assertEqual(recorder.rows[0]['request_id'],'actual-request')
            self.assertEqual(recorder.rows[0]['kind'],'ipc_encode_host')
            recorder.active=False
            self.assertEqual(MsgpackEncoder().encode(obj),expected)
            self.assertEqual(len(recorder.rows),1)
        finally:
            HTTP_REQUEST_ID.reset(token)
            for cls,key,original in reversed(originals):setattr(cls,key,original)


if __name__=='__main__':unittest.main()
