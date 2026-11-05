"""Exact one-handoff staircase candidate. GPU qualification is required."""
import os
import time
from dataclasses import replace


def resolve_prefix(cfg, prompt_tokens, request_id, seed, phase):
    primary_k = cfg['native_frontier'][1]
    if prompt_tokens == cfg['length']:
        return primary_k
    if (prompt_tokens == min(512, cfg['length'])
            and request_id.startswith('recovery_warm_short_D_geometry_')
            and phase.get('state') == 'RUNNING' and phase.get('kind') == 'short_D_geometry'
            and seed in phase.get('seeds', [])):
        return prompt_tokens * primary_k // cfg['length']
    raise ValueError('Only the explicit coordinator-identified short graph warm-up can scale a native frontier')


def install(cfg, base_connector):
    import torch
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import Request, RequestPhase
    from ..adaptive_loop_token_pd_2gpu.connector import PartitionConnector
    from .common import ROOT, write, parent
    c, primary_k = cfg['native_frontier']
    r = cfg['depth']
    assert 0 <= c < r and 0 < primary_k < cfg['length']

    class StaircaseConnector(base_connector):
        def command(self, command):
            if command['kind'] in ('reserve', 'prefill'):
                self._next_frontier_prefix = command['prefix_k']
            return super().command(command)

        def reserve(self, tid, tokens, *args, **kwargs):
            super().reserve(tid, tokens, *args, **kwargs)
            if tid not in self.transfers:
                return
            t = self.transfers[tid]
            t.prefix = self._next_frontier_prefix
            if not 0 < t.prefix < len(tokens):
                raise ValueError('Invalid exact frontier from the coordinator')
            if t.outgoing:
                t.request.native_staircase_prefix = t.prefix
                t.request.native_staircase_c = c

        def prefill_submitted(self, request, *, phase, start, progress, event, stream):
            t = self.transfers[request.request_id]
            if t.compute_done:
                return
            t.last_event = event
            ready = request.prefill_depth_ranges[r - 1][0] >= t.prefix
            if not ready:
                # The full shallow rectangle is complete. Retain its suffix
                # hidden state while P computes the deep prefix rectangle.
                request.arc_export_cut = r
                return
            n = len(request.prompt_token_ids)
            runner = self.engine.model_runner
            slot = runner._ensure_request_slot(request)
            stream = stream or runner.core_stream or torch.cuda.current_stream(self.cache.device)
            with torch.cuda.stream(stream):
                if event is not None:
                    stream.wait_event(event)
                if c:
                    state = runner.prefill_states[slot, :n]
                else:
                    state = request.hidden_state
                self.hidden[t.slot, :state.numel()].copy_(state.flatten())
                if not c and cfg['model'] == 'huginn':
                    generator = runner.rng.peek(request.request_id)
                    initial = getattr(self.engine.model, 'abc_prompt_initial_states', {}).get(id(generator))
                    if initial is None:
                        raise RuntimeError('Missing exact canonical Huginn suffix initial state')
                    begin = self.engine.model.config.hidden_size
                    self.hidden[t.slot, begin:begin + initial[t.prefix:n].numel()].copy_(initial[t.prefix:n].flatten())
                ready_event = torch.cuda.Event()
                ready_event.record(stream)
            generator = runner.rng.peek(request.request_id)
            if generator is not None:
                t.rng_state = generator.get_state().tolist()
            t.last_event = ready_event
            self.queue_snapshot(t, n, ready_event)
            self.engine.suspend_transfer(request)
            t.compute_done = True
            t.transfer_boundary_first = t.transfer_boundary_last = (0, n if c else t.prefix)
            t.transfer_boundary_count = 1
            self.prefill_tokens += n if c else t.prefix

        def layout(self, t):
            n, k = len(t.request.prompt_token_ids), t.prefix
            recurrent = set(getattr(self.engine.model, 'recurrent_kv_layers', range(self.cache.num_layers)))
            coda = set(getattr(self.engine.model, 'coda_kv_layers', ()))
            for depth in range(r):
                for layer in range(self.cache.num_layers):
                    if layer in recurrent:
                        end = n if depth < c else k
                    elif depth:
                        continue
                    elif layer in coda:
                        end = min(k, n - 1)
                    else:
                        end = n if c else k
                    if end:
                        yield depth, layer, end

        def hidden_elements(self, t):
            n, k = len(t.request.prompt_token_ids), t.prefix
            width = self.engine.model.config.hidden_size
            return n * width if c else width + ((n - k) * self.engine.model.config.n_embd
                if cfg['model'] == 'huginn' else 0)

        def accept_cut(self, request, hidden, *, cached_tokens=0):
            if cached_tokens:
                raise ValueError('Staircase migration does not reuse cached prefixes')
            t = self.transfers[request.request_id]
            n, k = len(request.prompt_token_ids), t.prefix
            allocation = self.cache._get_allocation(request.request_id)
            for depth, layer, end in self.layout(t):
                allocation.written[depth][layer].prefix = end
            if cfg.get('verify_transfer'):
                actual = self.fingerprint(t)
                passed = actual == t.expected_fingerprint
                parent.prior.append(ROOT / 'raw' / cfg['name'] / f'KV_checks_{os.getpid()}.jsonl',
                    dict(seed=request.sampling_params.seed, actual=actual, expected=t.expected_fingerprint,
                         passed=passed, shallow_loops=c, prefix=k))
                if not passed:
                    raise RuntimeError('Staircase transport byte identity failed')
            if not c and cfg['model'] == 'huginn':
                runner, model = self.engine.model_runner, self.engine.model
                generator = runner.rng.acquire(request.request_id, request.sampling_params, self.cache.device)
                width, embd = model.config.hidden_size, model.config.n_embd
                initial = hidden.new_zeros((n, embd))
                initial[k:].copy_(hidden[width:width + (n - k) * embd].view(n - k, embd))
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.cache.device))
                for stream in (runner.core_stream, runner.boundary_stream):
                    if stream is not None:
                        stream.wait_event(ready)
                states = getattr(model, 'abc_prompt_initial_states', {})
                model.abc_prompt_initial_states = states
                states[id(generator)] = initial
                initialized = getattr(runner, 'abc_huginn_initialized', set())
                runner.abc_huginn_initialized = initialized
                initialized.add(request.request_id)
            request.num_prefilled_tokens = k
            request.num_preluded_tokens = n if c else k
            request.prefill_window_origin = request.prefill_reclaimable_tokens = 0 if c else k
            request.prefill_depth_ranges[:] = Request.prefill_boundaries(n, r, start=k)
            if c:
                for depth in range(c):
                    request.prefill_depth_ranges[depth][0] = n
                    request.prefill_depth_ranges[depth][1] = n
                request.prefill_depth_ranges[c][1] = n
                width = self.engine.model.config.hidden_size
                request.import_prompt_state = hidden[:n * width].view(n, width)
            request.loops_done = c
            request.import_resume_phase = (RequestPhase.PREFILLING_RECURRENT if c else RequestPhase.PREFILLING_PRELUDE)
            self.engine.scheduler.queues[RequestPhase.WAITING].append(request.request_id)
            return True

    import vllm_rlt.pd.engine as pd
    from . import native_path
    native_path.ACTIVE_CONNECTOR = StaircaseConnector
    pd.PDStageConnector = native_path.NativeFactory
    scheduled = Scheduler._make_scheduled_request
    def span(self, request, phase, token_budget):
        item = scheduled(self, request, phase, token_budget)
        k = getattr(request, 'native_staircase_prefix', None)
        if k is not None and (phase == RequestPhase.PREFILL and c == 0
                or phase == RequestPhase.PREFILLING_RECURRENT):
            count = min(item.token_count, k - item.token_start)
            if count <= 0:
                raise RuntimeError('P attempted work past its exact staircase frontier')
            item = replace(item, token_count=count)
        if phase in (RequestPhase.PREFILLING_PRELUDE, RequestPhase.PREFILLING_RECURRENT):
            # Retained hidden storage uses absolute prompt positions, including
            # imported suffixes. Do not reinterpret it as a chunk-local buffer.
            item = replace(item, prefill_hidden_row_offset=item.token_start)
        return item
    Scheduler._make_scheduled_request = span
    planned = Scheduler._planned_loop_count
    def group(self, items):
        if {item.phase for item in items} == {RequestPhase.PREFILLING_RECURRENT}:
            counts = []
            for item in items:
                start, end = item.token_start, item.token_start + item.token_count
                bounds = item.request.prefill_depth_ranges
                target = getattr(item.request, 'arc_export_cut', r)
                if bounds[item.depth][0] != start or bounds[item.depth][1] < end:
                    raise RuntimeError('Incomplete staircase input frontier')
                if not all(left == start for left, right in bounds[item.depth:target]):
                    raise RuntimeError('Nonrectangular staircase continuation cannot be grouped')
                counts.append(target - item.depth)
            return min(counts)
        return planned(self, items)
    Scheduler._planned_loop_count = group
    progress = Scheduler._apply_prefill_progress
    def advance(self, item, phase, *, loop_count=None, event=None):
        if phase == RequestPhase.PREFILL and hasattr(item.request, 'native_staircase_prefix'):
            return self._apply_split_prefill_progress(item.request, RequestPhase.PREFILLING_PRELUDE,
                item, loop_count=loop_count, event=event)
        return progress(self, item, phase, loop_count=loop_count, event=event)
    Scheduler._apply_prefill_progress = advance
    write(ROOT / 'raw' / cfg['name'] / f'staircase_candidate_{os.getpid()}.json', dict(
        shallow_loops=c, primary_prefix=primary_k,
        evidence='RUNTIME_CANDIDATE_NOT_A_CORRECTNESS_OR_PERFORMANCE_VERDICT'))
