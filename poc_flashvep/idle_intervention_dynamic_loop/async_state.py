"""Copy exact prompt bytes under a local lease. Hash only stable host buffers."""
import hashlib
import os
import time
import queue
import math
from concurrent.futures import ThreadPoolExecutor
from .common import ROOT,write
from .protocol import should_capture_state


def install(cfg):
    import torch
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.request import Operation
    from ..looped_pd_headroom_recovery.state_probe import plane
    from ..looped_pd_abc.common import append
    from ..native_loop_2d_oracle import native_path
    initialize,execute,step,close,stats=StageEngine.__init__,ModelRunner._execute_batch,StageEngine.step,StageEngine.close,StageEngine.stats

    def observed(request):
        return should_capture_state(cfg,request.sampling_params.seed,
            (ROOT/'raw'/cfg['name']/'measurement_begin.json').exists())

    def initialized(engine,*args,**kwargs):
        result=initialize(engine,*args,**kwargs)
        engine._idle_probe_pending=[];engine._idle_probe_futures=[]
        engine._idle_deferred_hashes=[]
        engine._idle_probe_pool=ThreadPoolExecutor(max_workers=cfg.get('state_hash_threads',8),thread_name_prefix='exact-state-hash')
        engine._idle_probe_stream=torch.cuda.Stream(device=engine.model_runner.device)
        engine.model_runner._idle_probe_engine=engine
        engine._idle_import_hidden={}
        cache,model=engine.cache_manager,engine.model_runner.model
        recurrent=set(getattr(model,'recurrent_kv_layers',range(cache.num_layers)))
        coda=set(getattr(model,'coda_kv_layers',()))
        token_bytes=math.prod(cache.key_cache.shape[3:])*cache.key_cache.element_size()
        valid=sum((cfg['length']-1 if layer in coda else cfg['length'])
            for depth in range(cache.max_loops) for layer in range(cache.num_layers)
            if not depth or layer in recurrent)
        index_bytes=(cache.max_loops*cache.num_layers+1)*math.ceil(cfg['length']/cache.block_size)*8+256
        size=2*valid*token_bytes+model.config.hidden_size*2+index_bytes
        slots=cfg.get('state_pinned_slots',4)
        engine._idle_cpu_snapshots=torch.empty((slots,size),dtype=torch.uint8,pin_memory=True)
        engine._idle_cpu_available=queue.SimpleQueue()
        for index in range(slots):engine._idle_cpu_available.put(index)
        write(ROOT/'raw'/cfg['name']/f'async_pinned_pool_{os.getpid()}.json',dict(
            role=engine.stage_config.role,slots=slots,bytes_per_slot=size,total_pinned_bytes=slots*size,
            scope='Host memory only. Allocate before startup requests. Reuse a slot only after its CPU hashes complete. No timed cudaHostAlloc is required.'))
        return result

    # StageWorker creates its connector after StageEngine construction.
    # Attach at the factory boundary, not during StageEngine.__init__.
    Connector=native_path.ACTIVE_CONNECTOR
    create=Connector.create
    def created(cls,engine,*,config,notify):
        connector=create(engine,config=config,notify=notify)
        connector_progress=connector.progress
        def progressed():
            result=connector_progress();poll(engine);return result
        connector.progress=progressed
        if cfg.get('async_verify_handoff'):
            previous_notify=connector.notify
            def notifying(kind,**fields):
                if kind=='commit' and engine.stage_config.role=='prefill':
                    t=connector.transfers[fields['tid']]
                    if not observed(t.request):return previous_notify(kind,**fields)
                    layout=list(connector.layout(t));record=copy_planes(engine,t.request,layout,'idle_exact_wire_')
                    raw=connector.hidden[t.slot,:connector.hidden_elements(t)].detach().contiguous().view(torch.uint8).cpu().numpy()
                    record.update(wire_only=True,wire_hidden_sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest())
                    engine._idle_probe_pending.append(record)
                return previous_notify(kind,**fields)
            connector.notify=notifying
            acknowledged=connector.acknowledge
            def acknowledge(transfer_id):
                begin=time.monotonic()
                for record in engine._idle_probe_pending:
                    if record['rid']==transfer_id and record.get('wire_only'):
                        record['done'].synchronize()
                poll(engine)
                append(ROOT/'raw'/cfg['name']/f'async_ACK_observer_wait_{os.getpid()}.jsonl',dict(
                    transfer_id=transfer_id,elapsed_ms=(time.monotonic()-begin)*1000,
                    scope='Diagnostic D2H drain before native source retirement. CPU hashes do not block ACK. Not a clean transfer measurement.'))
                return acknowledged(transfer_id)
            if engine.stage_config.role=='prefill':connector.acknowledge=acknowledge
            accepted=engine.accept_import
            def imported(request,hidden,*,cached_tokens=0):
                if not observed(request):return accepted(request,hidden,cached_tokens=cached_tokens)
                t=connector.transfers[request.request_id]
                raw=connector.hidden[t.slot,:connector.hidden_elements(t)].detach().contiguous().view(torch.uint8).cpu().numpy()
                engine._idle_import_hidden[request.request_id]=hashlib.sha256(memoryview(raw).cast('B')).hexdigest()
                return accepted(request,hidden,cached_tokens=cached_tokens)
            if engine.stage_config.role=='decode':engine.accept_import=imported
        return connector
    Connector.create=classmethod(created)

    def slot(engine):
        while engine._idle_cpu_available.empty():
            poll(engine);time.sleep(.001)
        return dict(cpu_slot=engine._idle_cpu_available.get(),cpu_offset=0,
            release_cpu_slot=engine._idle_cpu_available.put)

    def host_copy(engine,record,gpu):
        raw=gpu.detach().contiguous().view(torch.uint8)
        start=record['cpu_offset'];end=start+raw.numel()
        assert end<=engine._idle_cpu_snapshots.shape[1],'Pinned state slot is too small'
        buffer=engine._idle_cpu_snapshots[record['cpu_slot'],start:end].view(raw.shape)
        buffer.copy_(raw,non_blocking=True);record['cpu_offset']=end
        return buffer

    def device_blocks(engine,record,table,pages):
        # A plain torch.tensor(..., device='cuda') waits for the preceding
        # copies on this stream. Reuse pinned indices and submit asynchronously.
        start=(record['cpu_offset']+7)//8*8;end=start+pages*8
        assert end<=engine._idle_cpu_snapshots.shape[1]
        cpu=engine._idle_cpu_snapshots[record['cpu_slot'],start:end].view(torch.int64)
        cpu.copy_(torch.tensor(table[:pages],dtype=torch.int64))
        record['cpu_offset']=end
        return cpu.to(engine.cache_manager.device,non_blocking=True)

    def copy_planes(engine,request,layout,tag):
        cache=engine.cache_manager;runner=engine.model_runner
        allocation=cache._get_allocation(request.request_id);lease=tag+request.request_id
        assert lease not in allocation.transfer_leases
        cache.pin_transfer(request.request_id,lease)
        record=dict(rid=request.request_id,lease=lease,n=len(request.prompt_token_ids),seed=request.sampling_params.seed,
            cut=getattr(request,'arc_received_cut',None),role=engine.stage_config.role,planes=[])
        record.update(slot(engine))
        with torch.cuda.stream(engine._idle_probe_stream):
            for depth,layer,valid in layout:
                assert allocation.written[depth][layer].prefix>=valid
                blocks=device_blocks(engine,record,allocation.block_tables[depth],(valid+cache.block_size-1)//cache.block_size)
                for kind,tensor in (('K',cache.key_cache),('V',cache.value_cache)):
                    gpu=plane(tensor,blocks,layer).flatten(0,1)[:valid].detach().contiguous().view(torch.uint8)
                    cpu=host_copy(engine,record,gpu)
                    record['planes'].append((dict(depth=depth,layer=layer,kind=kind,valid_tokens=valid),cpu))
            record['done']=torch.cuda.Event();record['done'].record(engine._idle_probe_stream)
        return record

    def wire_record(record,hidden_hash,layout=None):
        digest=hashlib.sha256();size=0
        available={(m['depth'],m['layer'],m['kind']):buffer for m,buffer in record['planes']}
        order=(layout if layout is not None else list(dict.fromkeys((m['depth'],m['layer'],m['valid_tokens']) for m,_ in record['planes'])))
        for depth,layer,valid in order:
            for kind in ('K','V'):
                raw=memoryview(available[depth,layer,kind][:valid].numpy()).cast('B')
                digest.update(raw);size+=len(raw)
        append(ROOT/'raw'/cfg['name']/f'async_wire_states_{os.getpid()}.jsonl',dict(
            role=record['role'],seed=record['seed'],KV_sha256=digest.hexdigest(),KV_valid_bytes=size,
            hidden_sha256=hidden_hash,evidence='MEASURED_ASYNC_FULL_VALID_HANDOFF_BYTES',
            scope='Source export layout or the same immutable destination prompt prefix. Hidden bytes sampled before source commit and before destination import. Local read leases protect GPU pages until D2H completes. Not clean timing.'))

    def hash_contents(record):
        if record.get('wire_only'):
            wire_record(record,record['wire_hidden_sha256']);return
        planes=[]
        for metadata,buffer in record['planes']:
            planes.append(dict(metadata,sha256=hashlib.sha256(memoryview(buffer.numpy()).cast('B')).hexdigest()))
        hidden=hashlib.sha256(memoryview(record['hidden'].numpy()).cast('B')).hexdigest()
        append(ROOT/'raw'/cfg['name']/f'prompt_states_{os.getpid()}.jsonl',dict(
            seed=record['seed'],cut=record['cut'],prompt_tokens=record['n'],
            hidden_sha256=hidden,planes=planes,evidence='MEASURED_ASYNC_EXACT_PROMPT_CORRECTNESS',
            scope='All original valid prompt bytes. First-Coda hidden clone and event-ordered read-only KV copies. A local native cache lease prevents reuse until D2H completes. SHA256 uses stable pinned host buffers. Diagnostic only, not clean timing.'))
        if record.get('wire_layout') is not None:
            wire_record(record,record['wire_hidden_sha256'],record['wire_layout'])

    def hash_buffers(record):
        try:return hash_contents(record)
        finally:record['release_cpu_slot'](record['cpu_slot'])

    def poll(engine,wait=False):
        remaining=[]
        for record in engine._idle_probe_pending:
            if wait:record['done'].synchronize()
            if record['done'].query():
                if cfg.get('state_hash_defer_until_close'):engine._idle_deferred_hashes.append(record)
                else:engine._idle_probe_futures.append(engine._idle_probe_pool.submit(hash_buffers,record))
                engine.cache_manager.unpin_transfer(record['rid'],record['lease'])
            else:remaining.append(record)
        engine._idle_probe_pending=remaining

    def copied(runner,batch,prepared):
        if batch.operation==Operation.CODA:
            engine=runner._idle_probe_engine;cache=engine.cache_manager
            hidden=prepared.forwards[0].forward.hidden;offset=0
            for span in batch.spans:
                request=span.request;n=len(request.prompt_token_ids)
                if span.token_start==n-1 and not request.generated_token_ids and observed(request):
                    allocation=cache._get_allocation(request.request_id)
                    lease='idle_exact_prompt_'+request.request_id
                    assert lease not in allocation.transfer_leases
                    cache.pin_transfer(request.request_id,lease)
                    value=hidden[offset].detach().clone()
                    ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(runner.device))
                    record=dict(rid=request.request_id,lease=lease,n=n,seed=request.sampling_params.seed,
                        cut=getattr(request,'arc_received_cut',None),role=engine.stage_config.role,planes=[],gpu_hidden=value)
                    record.update(slot(engine))
                    if cfg.get('async_verify_handoff'):
                        transfer=engine.connector.transfers[request.request_id]
                        record['wire_layout']=list(engine.connector.layout(transfer))
                        record['wire_hidden_sha256']=engine._idle_import_hidden.pop(request.request_id)
                    recurrent=set(getattr(runner.model,'recurrent_kv_layers',range(cache.num_layers)))
                    coda=set(getattr(runner.model,'coda_kv_layers',()))
                    with torch.cuda.stream(engine._idle_probe_stream):
                        engine._idle_probe_stream.wait_event(ready)
                        record['hidden']=host_copy(engine,record,value)
                        value.record_stream(engine._idle_probe_stream)
                        for depth,table in enumerate(allocation.block_tables):
                            blocks=device_blocks(engine,record,table,(n+cache.block_size-1)//cache.block_size)
                            for layer in range(cache.num_layers):
                                if depth and layer not in recurrent:continue
                                valid=n-1 if layer in coda else n
                                assert allocation.written[depth][layer].prefix>=valid
                                for kind,tensor in (('K',cache.key_cache),('V',cache.value_cache)):
                                    cpu=host_copy(engine,record,plane(tensor,blocks,layer).flatten(0,1)[:valid])
                                    record['planes'].append((dict(depth=depth,layer=layer,kind=kind,valid_tokens=valid),cpu))
                        record['done']=torch.cuda.Event();record['done'].record(engine._idle_probe_stream)
                    engine._idle_probe_pending.append(record)
                offset+=span.token_count
        return execute(runner,batch,prepared)

    def stepped(engine,*args,**kwargs):
        result=step(engine,*args,**kwargs);poll(engine);return result

    def closed(engine):
        poll(engine,True)
        for record in engine._idle_deferred_hashes:
            engine._idle_probe_futures.append(engine._idle_probe_pool.submit(hash_buffers,record))
        engine._idle_deferred_hashes.clear()
        for future in engine._idle_probe_futures:future.result()
        engine._idle_probe_pool.shutdown(wait=True)
        write(ROOT/'raw'/cfg['name']/f'async_state_contract_{os.getpid()}.json',dict(
            role=engine.stage_config.role,pending_D2H=len(engine._idle_probe_pending),
            completed_hashes=len(engine._idle_probe_futures),full_prompt_valid_bytes=True,
            measured_only=cfg.get('state_probe_measured_only',False),
            deferred_CPU_hashing=cfg.get('state_hash_defer_until_close',False),
            scope='Extra local read lease only. Native transport ACK and source ownership are unchanged. Every measured valid byte is retained. Warm-up hashing is optional, but model warm-up still executes. All copies and hash futures drain before teardown.'))
        return close(engine)

    def startup_stats(engine):
        if not (ROOT/'raw'/cfg['name']/'measurement_begin.json').exists():
            poll(engine,True)
            for future in engine._idle_probe_futures:future.result()
        return stats(engine)
    StageEngine.__init__,ModelRunner._execute_batch,StageEngine.step,StageEngine.close,StageEngine.stats=initialized,copied,stepped,closed,startup_stats
