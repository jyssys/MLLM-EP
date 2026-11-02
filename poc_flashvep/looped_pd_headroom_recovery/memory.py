"""Count exposed CUDA storages once. Keep opaque allocator bytes separate."""


def inventory(engine):
    import torch
    seen=set()
    objects=set()
    def count(value,depth=0):
        if isinstance(value,torch.Tensor):
            if value.device.type!='cuda':return 0
            storage=value.untyped_storage()
            key=(str(value.device),storage.data_ptr())
            if key in seen:return 0
            seen.add(key)
            return storage.nbytes()
        if value is None or depth>7 or id(value) in objects:return 0
        objects.add(id(value))
        if isinstance(value,torch.nn.Module):
            return sum(count(x,depth+1) for x in (*value.parameters(),*value.buffers()))
        if isinstance(value,dict):return sum(count(v,depth+1) for v in value.values())
        if isinstance(value,(list,tuple)):return sum(count(v,depth+1) for v in value)
        if type(value).__module__.startswith('vllm_rlt') and hasattr(value,'__dict__'):
            return sum(count(v,depth+1) for k,v in vars(value).items()
                if k not in ('engine','model','cache','cache_manager','metadata_builder','runtime'))
        return 0
    runner=engine.model_runner
    cache=engine.cache_manager
    weights=count(runner.model)
    KV=count((cache.key_cache,cache.value_cache))
    fields=('requestspaces','prefill_states','prefill_hidden','batch_input_buffers','states','async_state',
        'input_buffers','packed_metadata_buffers','loop_hidden','readback_slots','prompt_token_buffers')
    workspace=count({k:getattr(runner,k,None) for k in fields})
    connector=getattr(engine,'connector',None)
    staging=count(getattr(connector,'hidden',None))
    graph=count((runner.graphs,runner.coda_graphs))
    allocated=torch.cuda.memory_allocated(runner.device)
    return dict(weight_and_model_buffer_storage_bytes=weights,KV_storage_bytes=KV,
        exposed_runner_workspace_storage_bytes=workspace,exposed_NIXL_hidden_staging_storage_bytes=staging,
        exposed_graph_tensor_storage_bytes=graph,
        opaque_or_unattributed_allocator_bytes=allocated-weights-KV-workspace-staging-graph,
        allocator_allocated_bytes=allocated,
        scope='Unique CUDA storage inventory with weights/KV precedence. Opaque graph-pool and library allocations remain in the residual. This is not a complete graph-pool attribution.')
