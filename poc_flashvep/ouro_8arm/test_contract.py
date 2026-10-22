"""CPU regression tests for the frozen policy, ownership, and task boundaries."""
import ast
import hashlib
import importlib.util
import unittest
from unittest import mock
import torch
from .contract import *
from .stage import Stage,decision
from poc_flashvep.ouro_request_pp.model_adapter import StaticCache
from .request_runtime import path_for
from .wave import NoRefillWave


class ContractTests(unittest.TestCase):
    def test_profiled_native_environment_finds_ninja(self):
        import os
        import shutil
        import subprocess
        from .launch import execution_environment
        argv=['nsys','profile','--trace=cuda,nvtx',VLLM_PYTHON,'-u','-m','poc_flashvep.ouro_8arm.native_quality']
        with mock.patch.dict(os.environ,{'PATH':'/usr/bin:/bin'}):
            env=execution_environment(argv,[4])
        self.assertEqual(shutil.which('ninja',path=env['PATH']),str(Path(VLLM_PYTHON).parent/'ninja'))
        self.assertTrue(subprocess.check_output(['ninja','--version'],env=env,text=True).strip())

    def test_profile_resume_preserves_success_and_failure(self):
        import tempfile
        from . import replay_campaign
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for folder in ['logs','commands','profiles','state','raw']:(root/folder).mkdir()
            for label,code in [('profile_done_v1',0),('profile_retry_v1',1)]:
                (root/'logs'/f'{label}.log').write_text('Preserved original log.\n')
                write(root/'commands'/f'{label}_completion.json',dict(code=code))
                if code==0:
                    write(root/'raw'/label/'complete.json',dict(success=True))
                    (root/'profiles'/f'{label}.sqlite').touch()
            with mock.patch.object(replay_campaign,'ROOT',root),mock.patch.object(replay_campaign,'execute') as run_,\
                    mock.patch.object(replay_campaign.subprocess,'run') as export_:
                replay_campaign.capture_profile('profile_done_v1',['nsys','-o',str(root/'profiles/profile_done_v1'),'--label','profile_done_v1'],[4],True)
                run_.assert_not_called();export_.assert_not_called()
                replay_campaign.capture_profile('profile_retry_v1',['nsys','-o',str(root/'profiles/profile_retry_v1'),'--label','profile_retry_v1'],[4],True)
                self.assertEqual(run_.call_args.args[0],'profile_retry_v2')
                self.assertIn(str(root/'profiles/profile_retry_v2'),run_.call_args.args[1])
                self.assertEqual(read(root/'commands/profile_retry_v1_completion.json')['code'],1)
                self.assertEqual((root/'logs/profile_retry_v1.log').read_text(),'Preserved original log.\n')
                self.assertEqual(read(root/'state/profile_resume_map.json')['profile_retry_v1'],'profile_retry_v2')

    def test_bounded_serving_layout_trace(self):
        self.assertEqual(KV_allocation_cap(4096),4096)
        self.assertEqual(KV_allocation_cap(32),32)
        self.assertEqual(KV_allocation_cap(32,True),4096)
        with self.assertRaises(AssertionError):KV_allocation_cap(4096,True)

    def test_trace_command_plan_without_GPU(self):
        from . import replay_campaign
        status={arm:dict(status='PASS') for arm in ['SD','TD','LD','OD']}
        with mock.patch('sys.argv',['replay_campaign','--trace']),mock.patch.object(replay_campaign,'run') as run_,\
                mock.patch.object(replay_campaign,'execute') as execute_,mock.patch.object(replay_campaign,'restore'),\
                mock.patch.object(replay_campaign,'read',side_effect=lambda file:status if file.name=='dynamic_probe_status.json' else read(file)),\
                mock.patch.object(replay_campaign.subprocess,'run'):
            replay_campaign.main()
        serving=[call for call in execute_.call_args_list if call.args[0].startswith('profile_serving_shape_')]
        self.assertEqual(len(serving),8)
        for call in serving:
            self.assertIn('--serving-KV-shape',call.args[1])
            self.assertEqual(call.args[1][call.args[1].index('--cap')+1],'32')
            self.assertTrue(set(call.args[2])<={4,5,6,7})
        self.assertEqual(len(execute_.call_args_list),18)
        self.assertEqual(len([call for call in run_.call_args_list if call.args[0].startswith('observer_serving_shape_')]),8)
    def test_subset400_metadata_and_cohorts(self):
        rows=subset400();extra=subset400(True);plan=read(ROOT/'evaluation_subset400.json')
        self.assertEqual(len(rows),400)
        self.assertEqual(len({r['id'] for r in rows}),400)
        self.assertEqual(sum(r['dataset']=='gsm8k' for r in rows),300)
        self.assertEqual(sum(r['dataset']=='math500' for r in rows),100)
        self.assertEqual(len(extra),286)
        self.assertFalse({r['id'] for r in dataset(True)} & {r['id'] for r in extra})
        pilot_cohorts=[c for c in plan['cohorts'] if c['source']=='pilot']
        self.assertEqual([v for c in pilot_cohorts for v in c['ids']],[r['id'] for r in dataset(True)])
        for c in plan['cohorts']:
            self.assertLessEqual(len(c['ids']),4)
            self.assertEqual(len({r['dataset'] for r in rows if r['id'] in c['ids']}),1)

    def test_metric_ratios(self):
        from .replay_analysis import ratio_CI
        point,low,high=ratio_CI([2.,4.,6.,8.],[1.,2.,3.,4.])
        self.assertEqual(point,2.)
        self.assertEqual(low,2.)
        self.assertEqual(high,2.)
        self.assertEqual(ratio_CI([2.],[4.]),(.5,None,None))

    def test_task_cohort_stratified_ratio(self):
        import numpy as np
        from .final_tables import ratio_interval
        groups=[np.array([0,1]),np.array([2,3])]
        self.assertEqual(ratio_interval([2.,4.,6.,8.],[1.,2.,3.,4.],groups),(2.,2.,2.))
        self.assertEqual(ratio_interval([1.,2.,3.,4.],[1.,2.,3.,4.],groups),(1.,1.,1.))

    def test_free_running_first_difference(self):
        from .behavior_analysis import first_difference
        self.assertIsNone(first_difference([2,3],[2,3]))
        self.assertEqual(first_difference([2,3],[2,4]),1)
        self.assertEqual(first_difference([2,3],[2,3,4]),2)
        self.assertEqual(first_difference([],[]),None)

    def test_collective_kernel_labels(self):
        from .profile_analysis import collective_kind
        self.assertEqual(collective_kind('ncclDevKernel_AllReduce_Sum_f32'),'NCCL')
        self.assertEqual(collective_kind('vllm::cross_device_reduce_1stage<4>'),'VLLM_CUSTOM_ALL_REDUCE')
        self.assertEqual(collective_kind('vllm::cross_device_reduce_2stage<4>'),'VLLM_CUSTOM_ALL_REDUCE')
        self.assertIsNone(collective_kind('reduce_kernel<float>'))
        self.assertIsNone(collective_kind('unclassified_allreduce_fusion'))

    def test_rank_work_balance(self):
        from .placement_accounting import balance
        self.assertEqual(balance([1,1,1,1]),dict(CV=0.,max_over_mean=1.))
        self.assertAlmostEqual(balance([1,1,1,5])['CV'],3**.5/2)
        self.assertEqual(balance([1,1,1,5])['max_over_mean'],2.5)
        self.assertEqual(balance([0,0]),dict(CV=None,max_over_mean=None))

    def test_no_refill_and_canonical_admission(self):
        wave=NoRefillWave(['A','B','C','D'])
        self.assertEqual(wave.complete('C','C-next'),[])
        self.assertEqual(wave.complete('A','A-next'),[])
        self.assertEqual(wave.complete('D',None),[])
        self.assertEqual(wave.complete('B','B-next'),['A-next','B-next','C-next'])
        self.assertEqual(wave.expected,3)
        self.assertEqual(wave.complete('B',None),[])
        self.assertEqual(wave.complete('C','C-later'),[])
        self.assertEqual(wave.complete('A','A-later'),['A-later','C-later'])
    def test_syntax(self):
        for file in (PROJECT/'poc_flashvep/ouro_8arm').glob('*.py'):
            ast.parse(file.read_text(),filename=str(file))

    def test_manifest(self):
        rows=dataset(False)
        self.assertEqual(len(rows),1819)
        self.assertEqual(sum(r['pilot'] for r in rows if r['dataset']=='gsm8k'),64)
        self.assertEqual(sum(r['pilot'] for r in rows if r['dataset']=='math500'),50)
        for row in rows:
            self.assertEqual(row['prompt_sha256'],hashlib.sha256(row['prompt'].encode()).hexdigest())

    def test_owner_slots(self):
        for arm in ['S4','L4','O4','SD','LD','OD']:
            slots=[(loop,i) for _,loop,a,b in path_for(arm) for i in range(a,b)]
            self.assertEqual(len(slots),96)
            self.assertEqual(len(set(slots)),96)

    def test_official_hazard(self):
        file=CDB/'src/looped_cdb/continuous_depth_batching/exit_policy.py'
        spec=importlib.util.spec_from_file_location('official_exit_policy',file)
        module=importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        policy=module.ExitPolicy(module.ExitPolicySpec(module.ExitDecisionRule.CUMULATIVE_HAZARD),.5,2,4,False,False,False)
        for dtype in [torch.float32,torch.bfloat16]:
            for sequence in [[-8.,-8.,-8.,-8.],[0.,0.,0.,0.],[3.,-3.,-2.,-1.],[-3.,-2.,1.,2.]]:
                state=module.TokenExitPolicyState();cdf=0.;survival=1.
                for loop,value in enumerate(sequence):
                    signal=torch.tensor([[[value]]],dtype=dtype)
                    got,cdf,survival,_=decision(signal,loop,cdf,survival)
                    expected=policy.should_exit(state,source_step=loop,signal=signal) or loop==3
                    self.assertEqual(got,expected)
                    self.assertEqual(cdf,state.exit_cdf)
                    self.assertEqual(survival,state.survival)

    def test_minimum_depth_keeps_first_hazard(self):
        end,cdf,survival,_=decision(torch.tensor([3.]),0,0.,1.)
        self.assertFalse(end)
        self.assertGreater(cdf,.9)
        end,_,_,_=decision(torch.tensor([-8.]),1,cdf,survival)
        self.assertTrue(end)

    def test_last_exited_preserves_prefix(self):
        part=Stage.__new__(Stage);part.indices=list(range(6,12));cache=StaticCache(5);part.caches={'r':cache}
        for loop in range(4):
            for index in part.indices:
                slot=24*loop+index
                cache.key_cache[slot]=torch.full((1,16,5,128),float(slot),dtype=torch.bfloat16)
                cache.value_cache[slot]=torch.full((1,16,5,128),float(-slot),dtype=torch.bfloat16)
        before={slot:(x.clone(),cache.value_cache[slot].clone()) for slot,x in enumerate(cache.key_cache) if x is not None}
        part.fill_local('r',1,3)
        for slot,(k,v) in before.items():
            loop,index=divmod(slot,24)
            for now,old,source in [(cache.key_cache[slot],k,cache.key_cache[24+index]),(cache.value_cache[slot],v,cache.value_cache[24+index])]:
                self.assertTrue(torch.equal(now[:,:,:3],old[:,:,:3]))
                self.assertTrue(torch.equal(now[:,:,4:],old[:,:,4:]))
                self.assertTrue(torch.equal(now[:,:,3],source[:,:,3] if loop>1 else old[:,:,3]))

    def test_loop_owner_imports_current_row_only(self):
        source=Stage.__new__(Stage);target=Stage.__new__(Stage)
        source.indices=target.indices=list(range(24));a=StaticCache(5);b=StaticCache(5)
        b.length=3
        source.caches={'r':a};target.caches={'r':b}
        for i in range(24):
            a.key_cache[24+i]=torch.full((1,16,5,128),float(i+1),dtype=torch.bfloat16)
            a.value_cache[24+i]=-a.key_cache[24+i]
            for loop in [2,3]:
                b.key_cache[24*loop+i]=torch.zeros((1,16,5,128),dtype=torch.bfloat16)
                b.value_cache[24*loop+i]=torch.zeros_like(b.key_cache[24*loop+i])
        bank=torch.empty((24,2,1,16,128),dtype=torch.bfloat16)
        source.export_exit('r',1,3,bank);target.import_exit('r',1,[2,3],3,bank)
        self.assertEqual(b.length,4,'A skipped-loop owner must publish the newly filled KV row as valid.')
        for loop in [2,3]:
            for i in range(24):
                for view in [b.key_cache[24*loop+i],b.value_cache[24*loop+i]]:
                    self.assertEqual(int(torch.count_nonzero(view[:,:,:3])),0)
                    self.assertEqual(int(torch.count_nonzero(view[:,:,4:])),0)
                self.assertTrue(torch.equal(b.key_cache[24*loop+i][:,:,3],a.key_cache[24+i][:,:,3]))


if __name__=='__main__':
    unittest.main()
