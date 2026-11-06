"""Whole-request local ownership; native transfers remain unchanged."""
import os
import time
from dataclasses import replace
from .common import ROOT, write


def local_placement(request_id, all_local=False, every=4):
    if request_id.startswith('dlocal_warm_'):
        return True
    if not request_id.startswith('r0_q'):
        return False
    return all_local or int(request_id.rsplit('q', 1)[1]) % every == 0


def separate_schedule(scheduler, decode_phases, prefill_phases):
    for phase in decode_phases:
        if scheduler.has_ready(phase):
            return scheduler._take(phase)
    scheduler._admit()
    # Admission can restore an imported native request directly to CODA.
    # Recheck boundaries before returning no work; keep the progress guard.
    for phase in (*decode_phases, *prefill_phases):
        if scheduler.has_ready(phase):
            return scheduler._take(phase)
    return None


def validate_prompt_span(role, method, start, count, prompt_tokens, chunk):
    if count<=1:
        return
    if role=='prefill':
        assert start==0 and count==prompt_tokens,'P full-prompt contract changed'
    elif role=='decode' and method not in ('native','full'):
        assert count<=chunk and start+count<=prompt_tokens,'D-local slice exceeded its token bound'


def install(cfg):
    from vllm_rlt.pd.engine import PDEngine
    from vllm_rlt.engine.worker import StageWorker
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.core.scheduling_policy import SchedulingPolicy
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.request import RequestPhase as Phase
    folder = ROOT / 'raw' / cfg['name']
    method = cfg.get('dlocal_method', 'native')
    records = []
    refusal_clock = {}
    init_pd, add, admit = PDEngine.__init__, PDEngine.add_request, PDEngine._admit
    abort_pd = PDEngine.abort_request
    command, output = StageWorker.command, StageWorker.output
    init_stage, close_stage = StageEngine.__init__, StageEngine.close
    planned = Scheduler._planned_loop_count
    submit = ModelRunner.submit

    def append(event, **fields):
        records.append(dict(time=time.monotonic(), event=event, **fields))
        if cfg.get('dlocal_debug'):
            with (folder / f'dlocal_live_{os.getpid()}.jsonl').open('a') as handle:
                handle.write(__import__('json').dumps(records[-1])+'\n')

    def initialized_pd(self, *args, **kwargs):
        if method not in ('native', 'full'):
            kwargs['decode_scheduler_config'] = replace(kwargs['decode_scheduler_config'],
                mode='mixed', wavefront_chunked_prefill=False,
                prefill_chunk_size=cfg.get('dlocal_chunk', 32),
                max_num_batched_tokens=cfg.get('dlocal_token_budget', 256))
        return init_pd(self, *args, **kwargs)

    def added(self, rid, tokens, params=None, **kwargs):
        result = add(self, rid, tokens, params, **kwargs)
        if method != 'native':
            w = next(w for w in self.transfers.values() if w.request.request_id == rid)
            # Native geometry cohorts must retain their native transfer path:
            # their readiness barrier counts transfers, not local requests.
            selected = local_placement(rid, cfg.get('dlocal_all', False), cfg.get('dlocal_every', 4))
            w.dlocal_selected = selected
            append('PLACEMENT', request_id=rid, local=selected, phase=w.phase,
                P_has_started=False)
        return result

    def admitted(self):
        local = [w for w in self.transfers.values() if getattr(w, 'dlocal_selected', False)
                 and not w.cancelled and w.phase == 'waiting']
        for w in local:
            active_local = sum(getattr(item, 'dlocal_selected', False)
                and item.phase not in ('waiting','dlocal_waiting') and not item.d_released
                for item in self.transfers.values())
            if active_local >= cfg.get('dlocal_max_active', cfg['D_slots']):
                key=(w.tid,'LOCAL_ACTIVE_LIMIT')
                if time.monotonic()-refusal_clock.get(key,0.)>=.25:
                    refusal_clock[key]=time.monotonic()
                    append('LOCAL_ADMISSION_DEFERRED',tid=w.tid,request_id=w.request.request_id,
                        reason=key[1],active_local=active_local,P_KV_required=False)
                continue
            length = len(w.request.prompt_token_ids) + w.request.sampling_params.max_tokens - 1
            eligible = [p for p in self.peers.values() if p.role == 'decode'
                and p.slots < min(cfg['D_slots'], p.client.info['transfer_slots'])
                and p.reserved_blocks + self._blocks(p, length) <=
                    p.client.info['num_blocks'] - (p.client.info['watermark_blocks'] if p.slots else 0)]
            if not eligible:
                key=(w.tid,'D_COORDINATOR_SLOT_OR_KV_CREDIT')
                if time.monotonic()-refusal_clock.get(key,0.)>=.25:
                    refusal_clock[key]=time.monotonic()
                    append('LOCAL_ADMISSION_DEFERRED',tid=w.tid,request_id=w.request.request_id,
                        reason=key[1],P_KV_required=False,D_peers=[dict(slots=p.slots,
                            local_slot_limit=min(cfg['D_slots'],p.client.info['transfer_slots']),
                            reserved_blocks=p.reserved_blocks,total_blocks=p.client.info['num_blocks'],
                            required_blocks=self._blocks(p,length)) for p in self.peers.values() if p.role=='decode'],
                        scope='Coordinator reservation credits and declared local slot cap. Not active GPU compute occupancy.')
                continue
            d = min(eligible, key=lambda p: p.reserved_blocks)
            w.d, w.p, w.p_released = d.name, None, True
            w.d_reserved_blocks = self._blocks(d, length)
            d.reserved_blocks += w.d_reserved_blocks
            d.slots += 1
            w.phase = 'dlocal_prefill'
            w.timings['admitted'] = time.monotonic() - w.created
            self._send(d.name, 'dlocal_add', tid=w.tid,
                tokens=w.request.prompt_token_ids, params=w.request.sampling_params,
                trace_id=w.trace_id)
            append('LOCAL_CREDIT_RESERVED', request_id=w.request.request_id,
                tid=w.tid, blocks=w.d_reserved_blocks, D_reserved=d.reserved_blocks,
                D_total=d.client.info['num_blocks'], P_reserved=0)
        # Hide local waiters only from the native admission scan. Their original
        # ownership remains local, including temporary credit refusal.
        for w in local:
            if w.phase == 'waiting':
                w.phase = 'dlocal_waiting'
        try:
            return admit(self)
        finally:
            for w in local:
                if w.phase == 'dlocal_waiting':
                    w.phase = 'waiting'

    def commanded(self, record):
        if record['kind'] == 'dlocal_abort':
            tid = record['tid']
            if tid in getattr(self, '_dlocal_ids', ()):
                # Cancellation is not a timed fast path. Drain all device
                # readers before releasing local allocation and credit.
                self.engine.model_runner.synchronize()
                if self.engine.has_request(tid):
                    self.engine.abort_request(tid)
                assert not self.engine.cache_manager.has_allocation(tid)
                self._dlocal_ids.remove(tid)
                self.send('released', tid=tid, role='decode')
                append('LOCAL_ABORT_RETIRED_AFTER_DEVICE_DRAIN', tid=tid)
            return
        if record['kind'] != 'dlocal_add':
            return command(self, record)
        assert self.engine.stage_config.role == 'decode'
        tid = record['tid']
        if not hasattr(self, '_dlocal_ids'):
            self._dlocal_ids = set()
        assert tid not in self._dlocal_ids and tid not in self.connector.transfers
        self.engine._add_request(tid, record['tokens'], record['params'],
            trace_id=record.get('trace_id'))
        self._dlocal_ids.add(tid)
        self.send('prefill_started', tid=tid)
        append('LOCAL_ENQUEUED', tid=tid, seed=record['params'].seed,
            prompt_tokens=len(record['tokens']), native_transfer=False)

    def produced(self, value):
        if value.request_id not in getattr(self, '_dlocal_ids', ()):
            return output(self, value)
        self.send('output', request_id=value.request_id, output=value)
        if value.finished:
            assert not self.engine.has_request(value.request_id), 'Local work must retire before credit release'
            self._dlocal_ids.remove(value.request_id)
            self.send('released', tid=value.request_id, role='decode')
            append('LOCAL_RETIRED', tid=value.request_id, tokens=len(value.token_ids),
                exit_depths=list(value.exit_depths), cache_allocated=self.engine.cache_manager.has_allocation(value.request_id))

    def aborted(self, rid):
        request = self.requests.get(rid)
        w = next((item for item in self.transfers.values() if item.request is request), None)
        if w is None or not getattr(w, 'dlocal_selected', False):
            return abort_pd(self, rid)
        from vllm_rlt.request import FinishReason, RequestOutput
        self.requests.pop(rid)
        w.cancelled = True
        self.outputs = type(self.outputs)(o for o in self.outputs if o.request_id != rid)
        if w.d is None:
            del self.transfers[w.tid]
        else:
            self._send(w.d, 'dlocal_abort', tid=w.tid)
        request.phase, request.finish_reason = Phase.FINISHED, FinishReason.ABORT
        append('LOCAL_ABORT_REQUESTED', request_id=rid, tid=w.tid, D_owner=w.d)
        return RequestOutput.from_request(request)

    class PrioritySeparate(SchedulingPolicy):
        PREFILL_PHASES = (Phase.PREFILLING_PRELUDE, Phase.PREFILLING_RECURRENT)

        def schedule(self, scheduler, *, prefer_recurrent=False):
            # Ready decode work wins. Inflight work is not yet ready. A new
            # decode can become ready during a nonpreemptible prefill slice.
            return separate_schedule(scheduler, (Phase.CODA, Phase.PRELUDE, Phase.RECURRENT), self.PREFILL_PHASES)

    def initialized_stage(self, *args, **kwargs):
        result = init_stage(self, *args, **kwargs)
        if self.stage_config.role == 'decode' and method not in ('native', 'full'):
            assert self.scheduler.staged_prefill and not self.scheduler.wavefront_chunked_prefill
            if method in ('chunked', 'paced'):
                self.scheduler.policy = PrioritySeparate(self.scheduler.config)
            self.scheduler._dlocal_quantum = cfg['depth'] if method == 'chunked' else cfg.get('dlocal_quantum', 1)
        if self.stage_config.role == 'decode' and method == 'full':
            self.scheduler.policy.PREFILL_PHASES = (Phase.PREFILL,)
        append('STAGE_CONFIG', role=self.stage_config.role,
            staged=self.scheduler.staged_prefill, mode=self.scheduler.config.mode,
            chunk=self.scheduler.config.prefill_chunk_size)
        self.model_runner._dlocal_stage_role = self.stage_config.role
        if cfg.get('dlocal_debug') and self.stage_config.role == 'decode':
            step = self.step
            snapshot_clock = [0.]
            def stepped():
                result = step()
                if time.monotonic() - snapshot_clock[0] > 1.:
                    snapshot_clock[0] = time.monotonic()
                    append('SCHEDULER_SNAPSHOT', queues={str(k):list(v) for k,v in self.scheduler.queues.items()},
                        policy=type(self.scheduler.policy).__name__,
                        prefill_phases=[str(p) for p in self.scheduler.policy.PREFILL_PHASES],
                        token_budget=self.scheduler.config.max_num_batched_tokens,
                        ready={rid:self._execution_ready(r) for rid,r in self.scheduler.requests.items()},
                        requests={rid:str(r.phase) for rid,r in self.scheduler.requests.items()})
                return result
            self.step = stepped
        return result

    def loop_count(self, items):
        count = planned(self, items)
        if hasattr(self, '_dlocal_quantum') and any(i.phase in
                (Phase.PREFILLING_PRELUDE, Phase.PREFILLING_RECURRENT) for i in items):
            # A partial fused PRELUDE saves only the last row, not the staged
            # chunk. Use the native zero-loop embedding path before yielding.
            # The later recurrent phase gathers and updates the whole chunk.
            if method == 'paced' and any(i.phase == Phase.PREFILLING_PRELUDE for i in items):
                return 0
            # Preserve the scheduler's causal bound. Only yield earlier.
            return min(count, self._dlocal_quantum)
        return count

    def closed(self):
        write(folder / f'dlocal_events_{os.getpid()}.json', dict(events=records,
            scope='Host enqueue/retirement and exact reservation accounting. Enqueue is not actual GPU execution start.'))
        return close_stage(self)

    # The coordinator owns its separate record list; flush at engine close too.
    close_pd = PDEngine.close
    def closed_pd(self):
        try:
            return close_pd(self)
        finally:
            write(folder / f'dlocal_events_{os.getpid()}.json', dict(events=records,
                scope='Whole-request placement before P execution. Native requests retain the original protocol.'))
    PDEngine.__init__, PDEngine.add_request, PDEngine._admit = initialized_pd, added, admitted
    PDEngine.abort_request = aborted
    PDEngine.close = closed_pd
    StageWorker.command, StageWorker.output = commanded, produced
    StageEngine.__init__, StageEngine.close = initialized_stage, closed
    Scheduler._planned_loop_count = loop_count
    def submitted(runner, scheduled_batch=None, *, batch):
        role=getattr(runner,'_dlocal_stage_role',None)
        for span in batch.spans:
            if span.token_count<=1:
                continue
            validate_prompt_span(role,method,span.token_start,span.token_count,
                len(span.request.prompt_token_ids),cfg.get('dlocal_chunk',32))
        return submit(runner,scheduled_batch,batch=batch)
    ModelRunner.submit=submitted
    if method != 'native':
        from ..looped_pd_atomic_prefill import rlt_worker
        served = rlt_worker.serve
        def serve(engine, inputs, args, run, warmup=False, contract=None):
            result = served(engine, inputs, args, run, warmup, contract)
            if warmup:
                from vllm_rlt import SamplingParams
                rid = 'dlocal_warm_primary'
                started = time.monotonic()
                engine.add_request(rid, list(inputs[0]), SamplingParams(max_tokens=2,
                    temperature=0., seed=cfg['seed']-50000, min_loops=cfg['depth'],
                    max_loops=cfg['depth'], exit_threshold=1., ignore_eos=True))
                finished = False
                while not finished or engine.has_unfinished_requests():
                    assert time.monotonic()-started < 240, 'Bounded local graph warm-up'
                    for out in engine.step():
                        assert out.request_id == rid
                        if out.finished:
                            assert len(out.token_ids) == 2 and set(out.exit_depths) == {cfg['depth']}
                            finished = True
                append('LOCAL_WARMUP_COMPLETE', host_seconds=time.monotonic()-started,
                    excluded_from_serving_statistics=True)
            return result
        rlt_worker.serve = serve
