"""Campaign-only diagnostic extension. No model operator is replaced."""
from pathlib import Path
import os
from poc_looped_recurrence_tp_4models.worker_probe import FourModelProbe

if os.environ.get('OURO_CAMPAIGN_STORAGE_CONTROL'):
    from poc_ouro_parallelism_deep_discovery.alias_control import install
    install(os.environ['OURO_CAMPAIGN_STORAGE_CONTROL'])

class CampaignProbe(FourModelProbe):
    def export_key_rank_fixture(self,positions_count=257):
        import re
        import numpy as np
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
        assert get_tensor_model_parallel_world_size()==1
        torch.cuda.synchronize()
        core=self.get_model().model;L=len(core.layers)
        table=self.model_runner.block_tables.input_block_tables[0][0].long()
        positions=torch.arange(positions_count,device=self.device)
        block_ids=table[positions//16];offsets=positions%16
        selected=[];names=[];metadata=[]
        for name,module in self.vllm_config.compilation_config.static_forward_context.items():
            cache=getattr(module,'kv_cache',None)
            if not isinstance(cache,torch.Tensor) or not cache.numel():continue
            logical=int(re.search(r'layers\.(\d+)\.',name).group(1))
            if logical%L not in [0,L//2,L-1]:continue
            assert cache.ndim==4 and cache.shape[1]==16 and cache.shape[2]==16 and cache.shape[3]==256
            sample=cache[block_ids,:,offsets,:128].index_select(1,torch.tensor([0,8,15],device=self.device))
            assert sample.shape==(positions_count,3,128) and torch.isfinite(sample).all()
            selected.append(sample.permute(1,0,2).contiguous().view(torch.int16).cpu().numpy().view(np.uint16))
            names.append(name);metadata.append(dict(logical_layer=logical,physical_layer=logical%L,depth=logical//L+1))
        assert len(selected)==12
        path=Path(os.environ['LOOPED_KV_OUTPUT'])/'actual_key_words.npz'
        assert not path.exists()
        np.savez_compressed(path,K_words=np.stack(selected),positions=np.arange(positions_count),heads=np.array([0,8,15]))
        return dict(rank=get_tensor_model_parallel_rank(),path=str(path),sha256=__import__('hashlib').sha256(path.read_bytes()).hexdigest(),
            names=names,layers=metadata,shape=list(np.stack(selected).shape),
            precision='Original BF16 key words. No dtype reduction or dequantization.',
            scope='Selected actual native KV after a frozen controlled request. Post-request capture only. Not performance or quality.')

    def storage_control_evidence(self):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        core=self.get_model().model
        metadata=dict(core._campaign_storage_evidence)
        original_buffers=dict(core._campaign_physical_copies[0].named_buffers())
        buffer_checks=[]
        for depth in range(1,4):
            buffers=dict(core._campaign_physical_copies[depth].named_buffers())
            assert buffers.keys()==original_buffers.keys()
            shared=all(value.untyped_storage().data_ptr()==original_buffers[name].untyped_storage().data_ptr()
                       for name,value in buffers.items())
            assert shared,'A copied block changed buffer ownership after warmup.'
            buffer_checks.append(dict(depth=depth,buffers=len(buffers),all_buffer_storage_pointers_shared=True))
        unique={p.untyped_storage().data_ptr():p.untyped_storage().nbytes()
                for p in self.get_model().parameters()}
        metadata.update(rank=get_tensor_model_parallel_rank(),unique_parameter_storage_bytes=sum(unique.values()),
            unique_parameter_storages=len(unique),
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            post_warmup_buffer_checks=buffer_checks,
            inspector_count_note='Generic inspect_ownership physical_layers counts registered OuroAttention wrappers here. It is4L, not the L checkpoint layer identities or the number of KV slots.')
        return metadata

    def external_intervals_start(self,capacity=1024):
        """Record complete replay intervals. Do not time individual graph nodes."""
        import time
        import torch
        assert not hasattr(self,'interval_originals')
        self.interval_pool=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                            for _ in range(capacity)]
        # Materialize the event pool before the observed request.
        for first,last in self.interval_pool:first.record();last.record()
        torch.cuda.synchronize()
        self.interval_anchor=torch.cuda.Event(enable_timing=True)
        self.interval_anchor.record()
        self.interval_records=[];self.interval_used=0;self.interval_originals=[]
        def wrap(obj,name,kind):
            original=getattr(obj,name)
            def measured(*args,**kwargs):
                ordinal=self.interval_used;self.interval_used+=1
                assert ordinal<len(self.interval_pool),'External event capacity exceeded'
                first,last=self.interval_pool[ordinal]
                stream=int(torch.cuda.current_stream().cuda_stream)
                host_start=time.perf_counter_ns();first.record()
                result=original(*args,**kwargs)
                last.record();host_end=time.perf_counter_ns()
                self.interval_records.append(dict(ordinal=ordinal,kind=kind,
                    stream_before=stream,stream_after=int(torch.cuda.current_stream().cuda_stream),
                    host_call_start_ns=host_start,host_call_end_ns=host_end))
                return result
            setattr(obj,name,measured);self.interval_originals.append((obj,name,original))
        wrap(self.model_runner,'execute_model','execute_model_outer')
        wrap(self.model_runner,'sample_tokens','sample_tokens_outer')
        wrap(torch.cuda.CUDAGraph,'replay','graph_replay')
        return dict(capacity=capacity,pid=os.getpid(),scope='Events are outside the recorded CUDA graph.')

    def external_intervals_finish(self):
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        for obj,name,original in reversed(self.interval_originals):setattr(obj,name,original)
        torch.cuda.synchronize()
        rows=[]
        for value in sorted(self.interval_records,key=lambda r:r['ordinal']):
            first,last=self.interval_pool[value['ordinal']]
            rows.append(dict(value,start_ms=self.interval_anchor.elapsed_time(first),
                end_ms=self.interval_anchor.elapsed_time(last),duration_ms=first.elapsed_time(last)))
        del self.interval_originals
        return dict(rank=get_tensor_model_parallel_rank(),pid=os.getpid(),records=rows,
            interval_semantics='Each event pair encloses one complete call on its caller stream. Outer calls contain graph intervals. Never add nested intervals.',
            clock='Rank-local GPU event clock. No cross-rank timestamp alignment.',
            waiting='Inter-call gaps are unattributed. They are not proved idle, removable, or recurrence-specific.')

    def profiler_api_start(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        return {'pid':os.getpid(),'method':'cudaProfilerStart','scope':'Diagnostic process-tree capture only'}

    def profiler_api_stop(self):
        import torch
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        torch.cuda.synchronize()
        dist.barrier(group=get_tp_group().device_group)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return {'pid':os.getpid(),'method':'cudaProfilerStop','scope':'After-request synchronization, not a serving dependency'}

    def finish_kernel_profile(self):
        result=super().finish_kernel_profile()
        path=Path(os.environ['LOOPED_KV_OUTPUT'])/f'chrome.rank{result["rank"]}.json.gz'
        self.kernel_profile.export_chrome_trace(str(path))
        result['chrome_trace']=str(path)
        result['stream_identity']='Native Chrome CUDA trace fields. Inspect schema before interval attribution.'
        return result
