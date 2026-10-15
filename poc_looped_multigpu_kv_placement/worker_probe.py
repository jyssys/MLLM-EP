"""Read actual loaded model/cache ownership outside clean request timing."""
class KVWorkerProbe:
    """Native public worker extension. No pickle RPC or model replacement."""
    def inspect_ownership(self,reset=False):return inspect_worker(self,reset)
    def inspect_memory(self,reset=False):return memory_worker(self,reset)
    def start_outer_events(self):
        import torch
        self.outer_records=[];self.outer_original=[]
        self.outer_pool=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(160)]
        self.outer_origin=torch.cuda.Event(enable_timing=True);self.outer_origin.record()
        for name in ['execute_model','sample_tokens']:
            original=getattr(self.model_runner,name)
            def measured(*args,_fn=original,_name=name,**kwargs):
                a,b=self.outer_pool[len(self.outer_records)]
                rows=getattr(args[0],'total_num_scheduled_tokens',None) if args else None
                self.outer_records.append((_name,rows,a,b));a.record()
                try:return _fn(*args,**kwargs)
                finally:b.record()
            self.outer_original.append((name,original));setattr(self.model_runner,name,measured)
        return True
    def finish_outer_events(self):
        import torch
        for name,original in self.outer_original:setattr(self.model_runner,name,original)
        torch.cuda.synchronize()
        return [dict(category=name,scheduled_rows=rows,start_ms=self.outer_origin.elapsed_time(a),
            end_ms=self.outer_origin.elapsed_time(b),duration_ms=a.elapsed_time(b),
            scope='Actual GPU outer interval. May include host gaps. Per-rank clocks are not cross-rank aligned.')
            for name,rows,a,b in self.outer_records]
    def enable_loop_trace(self):
        from poc_looped_multigpu_kv_placement.loop_observer import LoopObserver
        self.loop_observer=LoopObserver(self.get_model());self.loop_observer.__enter__()
        return True
    def finish_loop_trace(self):
        self.loop_observer.__exit__();return self.loop_observer.finish()
    def start_kernel_profile(self):
        import torch
        self.kernel_profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False)
        self.kernel_profile.__enter__();return True
    def finish_kernel_profile(self):
        import gzip,json,os
        from pathlib import Path
        from vllm.distributed import get_tensor_model_parallel_rank
        self.kernel_profile.__exit__(None,None,None)
        rows=[dict(name=e.name,start_us=e.time_range.start,end_us=e.time_range.end,device_type=str(e.device_type),
            device_index=e.device_index,thread=e.thread,id=e.id,cpu_parent=None if e.cpu_parent is None else e.cpu_parent.id)
            for e in self.kernel_profile.events()]
        rank=get_tensor_model_parallel_rank();path=Path(os.environ['LOOPED_KV_OUTPUT'])/f'torch_events.rank{rank}.json.gz'
        with gzip.open(path,'wt') as stream:json.dump(rows,stream)
        return dict(rank=rank,events=len(rows),path=str(path),scope='Diagnostic service, not headline clean E2E')


def memory_worker(worker,reset=False):
    import torch
    torch.cuda.synchronize()
    result=dict(device=str(worker.device),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    if reset:torch.cuda.reset_peak_memory_stats()
    return result


def inspect_worker(worker,reset=False):
    import os
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank,get_tensor_model_parallel_world_size
    runner=worker.model_runner;model=worker.get_model();torch.cuda.synchronize()
    def tensor(t):
        return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),device=str(t.device),
            storage_ptr=t.untyped_storage().data_ptr(),data_ptr=t.data_ptr(),offset=t.storage_offset(),
            logical_bytes=t.numel()*t.element_size(),storage_bytes=t.untyped_storage().nbytes())
    caches=[]
    for name,module in worker.vllm_config.compilation_config.static_forward_context.items():
        value=getattr(module,'kv_cache',None)
        if isinstance(value,list):
            for i,item in enumerate(value):
                if isinstance(item,torch.Tensor) and item.numel():caches.append(dict(name=name,index=i,**tensor(item)))
        elif isinstance(value,torch.Tensor) and value.numel():caches.append(dict(name=name,**tensor(value)))
    parameters={name:tensor(value) for name,value in model.named_parameters()}
    storages={v['storage_ptr']:v['storage_bytes'] for v in parameters.values()}
    cache_storage={v['storage_ptr']:v['storage_bytes'] for v in caches}
    geometry=[]
    for name,module in model.named_modules():
        if type(module).__name__=='OuroAttention':
            geometry.append(dict(name=name,Q_heads=module.num_heads,KV_heads=module.num_kv_heads,
                head_dim=module.head_dim,loop_cache_modules=len(module.attn),
                qkv_weight=list(module.qkv_proj.weight.shape),o_weight=list(module.o_proj.weight.shape)))
    config=runner.kv_cache_config
    result=dict(pid=os.getpid(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        rank=get_tensor_model_parallel_rank(),world=get_tensor_model_parallel_world_size(),
        device=str(worker.device),GPU_uuid=str(torch.cuda.get_device_properties(worker.device).uuid),
        physical_layers=len(geometry),model_class=type(model).__name__,cache_slots=len(caches),
        cache=caches,parameters=parameters,unique_parameter_storage_bytes=sum(storages.values()),
        unique_KV_pool_storage_bytes=sum(cache_storage.values()),geometry=geometry,
        memory_allocated=torch.cuda.memory_allocated(),memory_reserved=torch.cuda.memory_reserved(),
        max_memory_allocated=torch.cuda.max_memory_allocated(),max_memory_reserved=torch.cuda.max_memory_reserved(),
        KV_pool_blocks=config.num_blocks,KV_group_specs=[repr(g) for g in config.kv_cache_groups],
        compilation=repr(worker.vllm_config.compilation_config),
        backend=[dict(name=name,impl=type(getattr(module,'impl',None)).__name__) for name,module in
            worker.vllm_config.compilation_config.static_forward_context.items()],
        actual_parallel=repr(worker.vllm_config.parallel_config))
    if reset:torch.cuda.reset_peak_memory_stats()
    return result
