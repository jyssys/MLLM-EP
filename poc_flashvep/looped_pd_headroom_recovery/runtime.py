"""Observe native owners. Keep reservation, ACK and numerical rules unchanged."""
import faulthandler
import functools
import json
import os
import pickle
import signal
import subprocess
import time
from pathlib import Path
from .common import ROOT, PYTHON, REPO, configure, write, append, digest

INSTALLED = False


def all_submitted_ids(tickets):
    # A completed GPU event does not retire the request's submitted work.
    # Until native retirement, it is not a fresh compute-ready queue entry.
    return {item.request.request_id for ticket in tickets for item in ticket.batch.requests}


class ProcessAdapter:
    """Expose the Process methods used by the unchanged native StageClient."""
    def __init__(self, process):
        self.process = process
        self.pid = process.pid

    @property
    def exitcode(self):
        return self.process.poll()

    def is_alive(self):
        return self.process.poll() is None

    def join(self, timeout=None):
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()


def install():
    global INSTALLED
    if INSTALLED:
        return
    INSTALLED = True
    old = configure()
    cfg = old.config()
    folder = ROOT / 'raw' / cfg['name']
    hashes=json.loads(os.environ.get('RECOVERY_SOURCE_HASHES','{}'))
    for path,expected in hashes.items():
        assert digest(path)==expected,'Recovery source changed before worker startup: '+path
    write(folder/f'loaded_recovery_source_{os.getpid()}.json',dict(pid=os.getpid(),time=time.time(),hashes=hashes,
        scope='Startup file hashes. The per-run archive also records reused qualified source and the unchanged private runtime diff.'))
    if cfg.get('worker') == 'native_mixed_worker':
        from ..looped_pd_atomic_prefill.native_mixed_worker import install as mixed
        mixed()
        from .ingress import install as ingress
        ingress()
        if cfg.get('observe_graph_captures') or cfg.get('prime_all_graph_shapes'):
            from .graph_warmup import install as graph_warmup
            graph_warmup(cfg)
        if cfg.get('async_evidence'):
            from .durable_evidence import install as durable
            durable(cfg)
        return
    from ..looped_pd_atomic_prefill.runtime import install as qualified
    from ..adaptive_recurrence_cut import state_probe as legacy_probe
    probe_install=legacy_probe.install
    legacy_probe.install=lambda config: None
    try:
        qualified()
    finally:
        legacy_probe.install=probe_install
    from .state_probe import install as exact_probe
    exact_probe(cfg)
    if cfg.get('observe_graph_captures') or cfg.get('prime_all_graph_shapes'):
        from .graph_warmup import install as graph_warmup
        graph_warmup(cfg)
    if cfg.get('observe_lifetime',True):
        from vllm_rlt.pd.engine import PDEngine
        from vllm_rlt.pd.connector import PDStageConnector
        message=PDEngine._message
        acknowledge=PDStageConnector.acknowledge
        forget=PDStageConnector._forget
        def observed_message(self,peer,record):
            kind=record.get('kind')
            pending=self.transfers.get(record.get('tid'))
            rid=pending.request.request_id if pending is not None else None
            selected=kind in ('reserved','prefill_started','prefill_complete','commit','received','activated','released','rejected')
            before=dict(slots=peer.slots,reserved_blocks=peer.reserved_blocks) if selected else None
            result=message(self,peer,record)
            if selected:
                append(folder/'lifetime_events.jsonl',dict(time=time.monotonic(),event='COORDINATOR_'+kind,
                    request_id=rid,tid=record.get('tid'),peer=peer.name,role=peer.role,before=before,
                    after=dict(slots=peer.slots,reserved_blocks=peer.reserved_blocks),
                    P_released=getattr(pending,'p_released',None),D_released=getattr(pending,'d_released',None),
                    scope='Native message handling receipt. A received receipt sends the native ACK. It is not a GPU DMA timer.'))
            return result
        def observed_ack(self,tid):
            result=acknowledge(self,tid)
            transfer=self.transfers.get(tid)
            append(folder/f'lifetime_owner_{os.getpid()}.jsonl',dict(time=time.monotonic(),event='NATIVE_ACK_RECEIVED',
                tid=tid,role=self.engine.stage_config.role,acknowledged=getattr(transfer,'acknowledged',None),
                scope='Receipt of the unchanged native ACK in the source owner.'))
            return result
        def observed_forget(self,transfer):
            before=self.cache.has_allocation(transfer.tid)
            result=forget(self,transfer)
            append(folder/f'lifetime_owner_{os.getpid()}.jsonl',dict(time=time.monotonic(),event='NATIVE_CONNECTOR_FORGET',
                tid=transfer.tid,role=self.engine.stage_config.role,allocation_before=before,
                allocation_after=self.cache.has_allocation(transfer.tid),
                scope='Native connector/staging retirement. Cache and lease rules are unchanged.'))
            return result
        PDEngine._message=observed_message
        PDStageConnector.acknowledge=observed_ack
        PDStageConnector._forget=observed_forget
    if cfg.get('cuda_only_profile'):
        from .cuda_only_profile import install as native_cuda_profile
        native_cuda_profile(cfg)
    if cfg.get('verify_transfer'):
        from ..looped_pd_abc import verify
        from .bytecheck import fingerprint
        verify.fingerprint=fingerprint
    if cfg.get('exact_descriptor_packing'):
        # Generic CPU metadata optimization. Keep the native coalescer,
        # partitioner, CUDA events, NIXL transport, ACKs and KV values.
        from ..looped_pd_abc import boundary
        from .descriptor import valid_segments
        boundary.valid_segments=valid_segments
        write(folder/f'exact_descriptor_packing_{os.getpid()}.json',dict(enabled=True,
            evidence='RUNTIME_CONFIGURATION',
            scope='Exact descriptor construction only. No model, KV allocation, valid frontier or wire protocol changes. Not a loop-specific method.'))
    if cfg.get('observe_descriptor_cost'):
        from ..looped_pd_abc.boundary import BoundaryConnector
        build=BoundaryConnector.queue_chunk
        replay_saved=[False]
        def observed_build(self,transfer,start,end,event,final):
            begin=time.perf_counter()
            cpu=time.process_time()
            before=len(transfer.chunks)
            result=build(self,transfer,start,end,event,final)
            duration=time.perf_counter()-begin
            append(folder/f'descriptor_cost_{os.getpid()}.jsonl',dict(time=time.monotonic(),role=self.engine.stage_config.role,
                request_id=transfer.tid,seed=transfer.request.sampling_params.seed,start=start,end=end,final=final,
                host_seconds=duration,process_CPU_seconds=time.process_time()-cpu,
                fragments_added=len(transfer.chunks)-before,evidence='MEASURED_HOST_DESCRIPTOR_PREPARATION_NOT_DMA'))
            if cfg.get('record_descriptor_replay') and not replay_saved[0] and (folder/'measurement_begin.json').exists():
                replay_saved[0]=True
                fields=('num_layers','depths','block_size','layer_bytes','block_bytes','token_bytes','key_ptr','value_ptr','device','hidden_ptr','hidden_bytes')
                peer=self.transport.peers[transfer.peer]
                write(folder/'descriptor_replay.json',dict(start=max(start,transfer.target_cached_tokens),end=end,final=final,
                    prompt_tokens=len(transfer.request.prompt_token_ids),
                    source_info={k:self.transport.info[k] for k in fields},target_info={k:peer[k] for k in fields},
                    source_tables=self.cache.plane_block_tables(transfer.tid),target_tables=transfer.target_tables,
                    source_slot=transfer.slot,target_slot=transfer.target_slot,
                    recurrent_layers=list(getattr(self.engine.model,'recurrent_kv_layers',range(self.cache.num_layers))),
                    coda_layers=list(getattr(self.engine.model,'coda_kv_layers',())),host_seconds=duration,
                    fragment_bytes=self.config.transfer_chunk_bytes,max_descriptors=self.config.max_transfer_descriptors,
                    scope='Actual native metadata only. CPU replay must not read these recorded GPU pointers or claim measured GPU speedup.'))
            return result
        BoundaryConnector.queue_chunk=observed_build
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.model_runner import ModelRunner
    stack = (folder / f'worker_stacks_{os.getpid()}.log').open('a')
    faulthandler.register(signal.SIGUSR1, file=stack, all_threads=True)
    # Keep the handle alive for asynchronous stack requests from the watchdog.
    StageEngine._recovery_stack_handle = stack
    initialize = StageEngine.__init__
    step = StageEngine._step
    close = StageEngine.close
    clocks = {}
    capture = {'started': False, 'stopped': False, 'until': None, 'submissions': 0}

    def begin_capture(device,reason):
        call_begin=time.monotonic()
        torch.cuda.cudart().cudaProfilerStart()
        capture.update(started=True,until=time.monotonic()+cfg['capture_seconds'])
        lo=time.monotonic()
        torch.cuda.nvtx.mark('RECOVERY_CAPTURE_CLOCK')
        hi=time.monotonic()
        append(folder/'capture_events.jsonl',dict(time=(lo+hi)/2,anchor_lower=lo,anchor_upper=hi,
            profiler_start_call_begin=call_begin,profiler_start_call_seconds=lo-call_begin,
            pid=os.getpid(),event='START',device=str(device),reason=reason,marker='RECOVERY_CAPTURE',wall_ns=time.time_ns()))

    def snapshot(engine, event):
        now = time.monotonic()
        cache = engine.cache_manager
        free, total = torch.cuda.mem_get_info(engine.model_runner.device)
        weights = sum(p.numel() * p.element_size() for p in engine.model_runner.model.parameters()) if hasattr(engine.model_runner, 'model') else None
        active = [dict(request_id=r.request_id, phase=str(r.phase), prompt_tokens=len(r.prompt_token_ids),
                       prefilled=r.num_prefilled_tokens, output_tokens=len(r.generated_token_ids))
                  for r in engine.scheduler.requests.values()]
        blocks_by_phase={}
        declared_written=0
        leases=0
        for request in engine.scheduler.requests.values():
            if not cache.has_allocation(request.request_id):continue
            allocation=cache._get_allocation(request.request_id)
            phase=getattr(request.phase,'value',str(request.phase))
            blocks_by_phase[phase]=blocks_by_phase.get(phase,0)+sum(len(t) for t in allocation.block_tables)
            declared_written+=sum(w.prefix+len(w.pending) for plane in allocation.written for w in plane)
            leases+=len(allocation.transfer_leases)
        active_blocks=sum(n for phase,n in blocks_by_phase.items() if not any(k in phase.lower() for k in ('receiv','wait','transfer','finish')))
        queued={str(phase):list(ids) for phase,ids in engine.scheduler.queues.items()}
        in_flight=all_submitted_ids(getattr(engine,'_inflight',()))
        blocked=set(getattr(engine,'_connector_blocked',()))
        queued_ids={rid for phase,ids in queued.items()
                    if not any(s in phase for s in ('RECEIVING','TRANSFERRING','FINISHED')) for rid in ids}
        ready=queued_ids-in_flight-blocked
        append(folder / f'memory_progress_{os.getpid()}.jsonl', dict(time=now, pid=os.getpid(), role=engine.stage_config.role,
            event=event, device_total_bytes=total, device_free_bytes=free,
            allocator_allocated_bytes=torch.cuda.memory_allocated(engine.model_runner.device),
            allocator_reserved_bytes=torch.cuda.memory_reserved(engine.model_runner.device),
            allocator_peak_bytes=torch.cuda.max_memory_allocated(engine.model_runner.device),
            weight_bytes=weights, KV_pool_bytes=cache.key_cache.numel()*cache.key_cache.element_size()*2,
            KV_total_blocks=cache.num_blocks, KV_used_blocks=cache.num_used_blocks,
            KV_free_blocks=cache.num_blocks-cache.num_used_blocks, KV_block_bytes=cache.bytes_per_block,
            KV_blocks_by_request_phase=blocks_by_phase,KV_active_request_blocks=active_blocks,
            KV_active_request_allocated_bytes=active_blocks*cache.bytes_per_block,
            KV_declared_written_bytes=declared_written*cache.num_kv_heads*cache.head_dim*2*cache.key_cache.element_size(),
            KV_transfer_lease_count=leases,
            KV_phase_scope='Active means a compute-phase request, not a currently executing kernel. Written bytes follow native submitted-write frontiers and stream dependencies. They are not an independent GPU readback proof.',
            requests=active, scheduler_queue_ids=queued, in_flight_ids=sorted(in_flight),
            all_submitted_unretired_ids=sorted(in_flight),
            queued_not_inflight_not_connector_blocked_ids=sorted(ready),connector_blocked=sorted(blocked),
            queue_scope='Stage-local admitted requests only. D destination reservation precedes P ingress. Excludes every submitted, unretired ticket, including completed events, and connector-blocked rows. Unadmitted coordinator requests remain separate.',
            evidence='MEASURED_OWNER_SNAPSHOT', scope='Allocator totals overlap the KV pool and weights. Do not stack them. Unexposed graph/workspace/staging attribution is NOT_MEASURED.'))

    def initialized(self, *args, **kwargs):
        result = initialize(self, *args, **kwargs)
        clocks[id(self)] = 0.
        snapshot(self, 'INITIALIZED')
        from .memory import inventory
        write(folder/f'memory_inventory_initial_{os.getpid()}.json',dict(role=self.stage_config.role,time=time.monotonic(),**inventory(self)))
        if cfg.get('capture_prewarmed') and self.stage_config.role==cfg.get('capture_role','prefill'):
            begin_capture(self.model_runner.device,'BEFORE_WORKER_READY_AND_REQUEST_WARMUP')
        return result

    def stepped(self, *args, **kwargs):
        now = time.monotonic()
        if now-clocks.get(id(self), 0.) >= cfg.get('memory_sample_seconds', .5):
            clocks[id(self)] = now
            snapshot(self, 'PROGRESS')
        if capture['started'] and not capture['stopped'] and not cfg.get('capture_until_close') and now >= capture['until']:
            torch.cuda.nvtx.mark('RECOVERY_CAPTURE_STOP_CLOCK')
            before=time.monotonic()
            torch.cuda.cudart().cudaProfilerStop()
            capture['stopped'] = True
            append(folder / 'capture_events.jsonl', dict(time=time.monotonic(), stop_begin=before,pid=os.getpid(), event='STOP', role=self.stage_config.role))
        return step(self, *args, **kwargs)

    def closed(self):
        if capture['started'] and not capture['stopped']:
            torch.cuda.nvtx.mark('RECOVERY_CAPTURE_STOP_CLOCK')
            before=time.monotonic()
            if not cfg.get('profile_stop_at_exit'):
                torch.cuda.cudart().cudaProfilerStop()
            capture['stopped'] = True
            append(folder / 'capture_events.jsonl', dict(time=time.monotonic(),stop_begin=before,pid=os.getpid(),
                event='METRIC_WINDOW_END_COLLECTION_CONTINUES_TO_APP_EXIT' if cfg.get('profile_stop_at_exit') else 'STOP_CLOSE', role=self.stage_config.role))
        snapshot(self, 'CLOSE')
        from .memory import inventory
        write(folder/f'memory_inventory_close_{os.getpid()}.json',dict(role=self.stage_config.role,time=time.monotonic(),**inventory(self)))
        return close(self)

    StageEngine.__init__ = initialized
    StageEngine._step = stepped
    StageEngine.close = closed
    if cfg.get('capture_seconds'):
        submit = ModelRunner.submit
        def captured_submit(self, scheduled_batch=None, *, batch):
            # Device index follows the native pair mapping. Only the selected
            # process is launched under Nsight, so D remains uninstrumented.
            selected = str(self.device) == ('cuda:0' if cfg.get('capture_role', 'prefill') == 'prefill' else 'cuda:1')
            capture['submissions'] += 1
            measured=folder/'measurement_begin.json'
            eligible=measured.exists() or cfg.get('capacity_mode')
            if selected and eligible and not capture['started'] and capture['submissions'] > cfg.get('capture_after_submissions', 2):
                begin_capture(self.device,'MEASURED_SUBMISSION_HOT_START')
            return submit(self, scheduled_batch, batch=batch)
        ModelRunner.submit = captured_submit
    if cfg.get('P_only_profiler'):
        import multiprocessing as mp
        import vllm_rlt.engine.llm_engine as llm
        original = llm.start_stage_worker
        def profiled(stage_config, device, model_source, options, *, connector_factory=None, max_control_messages=32):
            if stage_config.role != cfg.get('capture_role', 'prefill'):
                return original(stage_config, device, model_source, options,
                    connector_factory=connector_factory, max_control_messages=max_control_messages)
            parent, child = mp.get_context('spawn').Pipe()
            payload = folder / f'profile_stage_{stage_config.role}.pickle'
            # Payload contains dataclasses and native factory parameters only.
            # It contains no model tensors and no replacement model execution.
            payload.write_bytes(pickle.dumps((stage_config, device, model_source, options,
                dict(connector_factory=connector_factory, max_control_messages=max_control_messages))))
            command = ['nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--cuda-graph-trace='+cfg.get('nsys_graph_trace','node'),
                '--cuda-flush-interval='+str(cfg.get('nsys_cuda_flush_ms',0)),
                '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--force-overwrite=false',
                '--output='+str(folder/'p_only_trace'), str(PYTHON), '-m', 'poc_flashvep.looped_pd_headroom_recovery.stage', str(payload), str(child.fileno())]
            log = (folder / 'p_only_profiler.log').open('w')
            process = subprocess.Popen(command, cwd=REPO, env=os.environ.copy(), pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT)
            child.close()
            append(folder/'profile_launch.jsonl', dict(time=time.time(), pid=process.pid, role=stage_config.role, command=command))
            return ProcessAdapter(process), parent
        llm.start_stage_worker = profiled
    if cfg.get('async_evidence'):
        from .durable_evidence import install as durable
        durable(cfg)
