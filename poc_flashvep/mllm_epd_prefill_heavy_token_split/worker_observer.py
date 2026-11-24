"""RPC-controlled kernel/copy observation. Never change model inputs or outputs."""
import ctypes
import json
import os
from pathlib import Path
import re
import time

from .common import ROOT,write,digest


def prepare_artifact():
    import shutil
    from .common import PRIOR
    source=PRIOR/'references/kernel_only_collector_v3'
    target=ROOT/'references/kernel_only_collector_v3';target.mkdir(parents=True,exist_ok=True)
    record=json.loads((source/'build.json').read_text())
    for key in ('source','library'):
        original=Path(record[key]);destination=target/original.name
        if not destination.exists():shutil.copyfile(original,destination)
        assert digest(destination)==record[key+'_SHA256']
        record[key]=str(destination)
    write(target/'build.json',record)


class Collector:
    def __init__(self,name,role,copies,runner):
        assert re.fullmatch('[A-Za-z0-9_.-]+',name) and name not in ('.','..')
        key=role+('_gpu'+os.environ['CUDA_VISIBLE_DEVICES'] if role in ('agg','pd') else '')
        self.folder=ROOT/'raw/profiles'/name/key
        assert not self.folder.exists(),'Do not overwrite a profile'
        self.folder.mkdir(parents=True)
        build=json.loads((ROOT/'references/kernel_only_collector_v3/build.json').read_text())
        for k in ('source','library','CUPTI_library'):assert digest(build[k])==build[k+'_SHA256']
        self.lib=ctypes.CDLL(build['library']);self.copies=copies;self.runner=runner
        self.lib.activity_prepare.argtypes=[ctypes.c_size_t]
        self.lib.activity_timestamp.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        self.lib.activity_end.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.c_uint64]
        for k in ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes'):
            getattr(self.lib,'activity_'+k).restype=ctypes.c_uint64
        assert self.lib.activity_prepare(1024*1024**2)==0
        self.rows=[];self.calls=[];self.originals=[]
        write(self.folder/'provenance.json',dict(build=build,role=role,pid=os.getpid(),include_copies=copies,
            scope='CUPTI kernels and optional copies. No driver/runtime/external or graph-enclosure tracing. '
                'Kernel union is not SM,Tensors or HBM utilization. Observer qualification is required.'))

    def anchor(self):
        value=ctypes.c_uint64();left=time.monotonic_ns()
        assert self.lib.activity_timestamp(ctypes.byref(value))==0
        return dict(cupti_ns=value.value,host_lower_ns=left,host_upper_ns=time.monotonic_ns())

    def begin(self):
        import torch
        torch.cuda.synchronize()
        assert self.lib.activity_enable(int(self.copies))==0
        self.left=self.anchor();self.install_state_hooks()
        return self.left

    def install_state_hooks(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        original_schedule=Scheduler.schedule;original_execute=self.runner.execute_model;me=self
        self.execute_had_instance_slot='execute_model' in vars(self.runner)
        self.execute_original_slot=vars(self.runner).get('execute_model')
        def queue(scheduler):
            waiting=list(scheduler.waiting);running=list(scheduler.running)
            return dict(waiting=len(waiting),running=len(running),
                waiting_status={str(status):sum(str(r.status)==str(status) for r in waiting)
                    for status in {r.status for r in waiting}},
                decode_active=sum(r.num_computed_tokens>=r.num_prompt_tokens for r in running),
                prefill_active=sum(r.num_computed_tokens<r.num_prompt_tokens for r in running),
                decode_ready_waiting=sum(r.num_computed_tokens>=r.num_prompt_tokens for r in waiting))
        def schedule(scheduler,*args,**kwargs):
            begin=time.monotonic_ns();before=queue(scheduler);output=original_schedule(scheduler,*args,**kwargs)
            requests=[]
            for rid,n in output.num_scheduled_tokens.items():
                request=scheduler.requests.get(rid)
                if request is not None:requests.append(dict(request_id=rid,scheduled_tokens=n,
                    prompt_tokens=request.num_prompt_tokens,num_computed_tokens=request.num_computed_tokens,
                    status=str(request.status)))
            me.rows.append(dict(begin_ns=begin,end_ns=time.monotonic_ns(),before=before,after=queue(scheduler),
                scheduled=requests,actual_scheduled_requests=len(output.num_scheduled_tokens),
                scheduled_tokens=output.total_num_scheduled_tokens,
                scheduled_encoder_items=sum(len(v) for v in output.scheduled_encoder_inputs.values())))
            return output
        def execute(output,*args,**kwargs):
            begin=time.monotonic_ns();result=original_execute(output,*args,**kwargs)
            me.calls.append(dict(begin_ns=begin,end_ns=time.monotonic_ns(),
                scheduled_tokens=output.total_num_scheduled_tokens,
                scheduled_requests=output.num_scheduled_tokens,
                scheduled_encoder_items=output.scheduled_encoder_inputs))
            return result
        self.originals=[(Scheduler,'schedule',original_schedule)]
        Scheduler.schedule=schedule;self.runner.execute_model=execute

    def close(self):
        import torch
        for cls,attribute,original in self.originals:setattr(cls,attribute,original)
        if self.execute_had_instance_slot:self.runner.execute_model=self.execute_original_slot
        else:delattr(self.runner,'execute_model')
        torch.cuda.synchronize()
        assert self.lib.activity_disable()==0
        right=self.anchor()
        code=self.lib.activity_end(str(self.folder/'kernels.csv').encode(),0,2**64-1)
        counters={k:int(getattr(self.lib,'activity_'+k)()) for k in
            ('dropped','invalid','count','all_kernels','requested_buffers','completed_buffers','reserved_host_bytes')}
        value=dict(left=self.left,right=right,returncode=code,counters=counters,
            passed=code==0 and not counters['dropped'] and not counters['invalid'])
        write(self.folder/'actual_ROI.json',value)
        write(self.folder/'scheduler.json',dict(rows=self.rows,
            scope='Actual engine scheduler state and query geometry. Not GPU-exclusive request cost.'))
        write(self.folder/'worker_calls.json',dict(rows=self.calls,
            scope='Host model-runner enclosures. Kernels may overlap with copies and other requests.'))
        return value


class CampaignWorkerExtension:
    def campaign_timeline_begin(self,name):
        from .host_timeline import EngineTimeline
        self._campaign_timeline=EngineTimeline(self.model_runner,name)
        return self._campaign_timeline.begin()

    def campaign_timeline_end(self):
        return self._campaign_timeline.close()

    def campaign_llm_probe_begin(self,name,max_logits=16):
        from .forward_probe import ForwardProbe
        self._campaign_llm_probe=ForwardProbe(self,name,max_logits)
        return self._campaign_llm_probe.begin()

    def campaign_llm_probe_end(self):
        return self._campaign_llm_probe.close()

    def campaign_vision_fingerprint(self):
        import hashlib
        import torch
        model=self.model_runner.model
        visual=getattr(model,'visual',None)
        assert visual is not None,type(model).__name__
        def signature(tensor):
            data=tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),bytes=len(data),SHA256=hashlib.sha256(data).hexdigest())
        parameters={name:signature(value) for name,value in visual.named_parameters()}
        buffers={name:signature(value) for name,value in visual.named_buffers()}
        return dict(pid=os.getpid(),model_type=type(model).__name__,
            runner_type=type(self.model_runner).__module__+'.'+type(self.model_runner).__name__,
            vision_type=type(visual).__name__,vision_backend=str(getattr(visual,'attn_backend',None)),
            vision_parameters=parameters,vision_buffers=buffers,
            scope='CORRECTNESS_ONLY. Reads original parameters and buffers. CPU copies exclude this RPC from timing.')

    def campaign_encoder_capture_begin(self,name):
        import hashlib
        from safetensors.torch import save_file
        assert re.fullmatch('[A-Za-z0-9_.-]+',name) and name not in ('.','..')
        folder=ROOT/'raw/encoder_reference'/name
        assert not folder.exists();folder.mkdir(parents=True)
        model=self.model_runner.model
        original=model.embed_multimodal;rows=[];calls=[]
        def captured(**kwargs):
            import torch
            call_id=len(calls);calls.append(dict(call_id=call_id))
            output=original(**kwargs)
            if kwargs.get('pixel_values') is not None and kwargs.get('image_grid_thw') is not None:
                pixels=kwargs['pixel_values'].detach().cpu().contiguous()
                grids=kwargs['image_grid_thw'].detach().cpu()
                items=list(output) if not isinstance(output,torch.Tensor) else list(output.unbind())
                assert len(items)==len(grids)
                offset=0
                for grid,embedding in zip(grids,items):
                    count=int(grid.prod());sample=pixels[offset:offset+count].contiguous();offset+=count
                    key=hashlib.sha256(str(list(grid.tolist())).encode()+sample.view(torch.uint8).numpy().tobytes()).hexdigest()
                    file=folder/(key+'_'+str(len(rows))+'.safetensors')
                    assert not file.exists(),'Do not overwrite a captured embedding'
                    save_file({'embedding':embedding.detach().cpu().contiguous()},str(file))
                    rows.append(dict(input_pixel_SHA256=key,input_pixel_dtype=str(sample.dtype),
                        grid=grid.tolist(),shape=list(embedding.shape),
                        file=str(file),SHA256=digest(file),call_id=call_id,
                        images_in_call=len(grids),patches_in_call=len(pixels)))
                assert offset==len(pixels)
            return output
        model.embed_multimodal=captured
        self._campaign_encoder_capture=dict(folder=folder,original=original,rows=rows)
        return dict(folder=str(folder),scope='CORRECTNESS_ONLY. Original official vLLM encoder output passthrough.')

    def campaign_encoder_capture_end(self):
        capture=self._campaign_encoder_capture
        self.model_runner.model.embed_multimodal=capture['original']
        write(capture['folder']/'manifest.json',dict(rows=capture['rows'],
            evidence='CORRECTNESS_ONLY',scope='Actual colocated official vLLM encoder. Extra CPU copies exclude this run from performance.'))
        return dict(images=len(capture['rows']),folder=str(capture['folder']))

    def campaign_profile_begin(self,name,include_copies=True):
        assert not hasattr(self,'_campaign_collector'),'One profile per independent worker start'
        self._campaign_collector=Collector(name,os.environ['MLLM_EPD_ROLE'],include_copies,self.model_runner)
        return self._campaign_collector.begin()

    def campaign_profile_end(self):
        return self._campaign_collector.close()

    def campaign_worker_info(self):
        import torch
        config=self.model_runner.vllm_config
        mm=getattr(self.model_runner,'mm_budget',None)
        model=self.model_runner.model
        parameters=list(model.named_parameters())
        groups={}
        for name,value in parameters:
            key=name.split('.')[0]
            groups[key]=groups.get(key,0)+value.numel()*value.element_size()
        free,total=torch.cuda.mem_get_info()
        return dict(pid=os.getpid(),role=os.environ['MLLM_EPD_ROLE'],
            model_runner=type(self.model_runner).__module__+'.'+type(self.model_runner).__name__,
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            maximum_allocated_bytes=torch.cuda.max_memory_allocated(),
            device_free_bytes=free,device_total_bytes=total,
            model_parameter_logical_bytes=sum(value.numel()*value.element_size() for _,value in parameters),
            model_parameter_bytes_by_prefix=groups,
            device_properties=str(torch.cuda.get_device_properties(0)),
            model_type=type(self.model_runner.model).__name__,
            scheduler=dict(B=config.scheduler_config.max_num_batched_tokens,S=config.scheduler_config.max_num_seqs,
                async_scheduling=config.scheduler_config.async_scheduling,
                chunked_prefill=config.scheduler_config.enable_chunked_prefill,
                encoder_compute_budget=mm.encoder_compute_budget if mm else None,
                encoder_cache_size=mm.encoder_cache_size if mm else None),
            scope='Worker allocator/device counters and logical parameter bytes. '
                'Parameter sums do not measure runtime workspace or shared-storage ownership. '
                'Not a co-residency or safe-quantum result.')

    def campaign_runtime_state(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        if not hasattr(self,'_campaign_schedule_watch_original'):
            original=Scheduler.schedule;me=self
            def watched(scheduler,*args,**kwargs):
                me._campaign_scheduler_reference=scheduler
                return original(scheduler,*args,**kwargs)
            self._campaign_schedule_watch_original=original
            Scheduler.schedule=watched
        rows=[];obj=getattr(self,'_campaign_scheduler_reference',None)
        if obj is not None:
            cache=obj.encoder_cache_manager
            rows.append(dict(type=type(obj).__name__,running=len(obj.running),waiting=len(obj.waiting),
                waiting_requests=[dict(request_id=r.request_id,prompt_tokens=r.num_prompt_tokens,
                    computed_tokens=r.num_computed_tokens,status=str(r.status),
                    encoder_lengths=[r.get_num_encoder_embeds(i) for i in range(len(r.mm_features or []))]) for r in list(obj.waiting)[:8]],
                encoder_cache_size=cache.cache_size,encoder_free_slots=cache.num_free_slots,
                encoder_freeable_slots=cache.num_freeable_slots,
                encoder_compute_budget=obj.max_num_encoder_input_tokens,
                KV_free_blocks=obj.kv_cache_manager.block_pool.get_num_free_blocks()))
        return dict(pid=os.getpid(),monotonic_ns=time.monotonic_ns(),schedulers=rows,
            watcher_status='SCHEDULER_OBSERVED' if rows else 'WATCH_INSTALLED_PENDING_FIRST_SCHEDULE',
            scope='DIAGNOSTIC read-only scheduler snapshot. Do not include this RPC in measured serving latency.')
