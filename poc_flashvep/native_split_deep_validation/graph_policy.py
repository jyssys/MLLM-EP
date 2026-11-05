"""Reserve primary exact query geometries without changing native graph math."""
import os
import time
from types import MethodType


def preferred_geometry(cfg, role):
    """Return (query rows, maximum sequence query length), not padding buckets."""
    t = cfg['length']
    frontier = cfg.get('native_frontier')
    if role == 'prefill':
        queries = ([frontier[1]] if frontier and frontier[0] == 0 else
                   [t, frontier[1]] if frontier else [t])
        slots, limit = cfg['P_slots'], cfg['P_prefill_graph_rows']
    else:
        queries = ([t - frontier[1]] if frontier else
                   [t] if cfg.get('native_partial_cut') is not None else [])
        slots, limit = cfg['D_slots'], cfg['D_prefill_graph_rows']
    batches = [1, 2, 4, 8] if cfg['deep_graph_budget'] <= 4 else list(range(1, 9))
    return {(q * b, q) for q in queries for b in batches if b <= slots and q * b <= limit}


def install(cfg):
    from vllm_rlt.engine.stage_engine import StageEngine
    from vllm_rlt.worker.cuda_graph import PreparedGraph
    from ..looped_pd_atomic_prefill import rlt_worker
    from .common import ROOT, write, prior
    append = prior.parent.prior.append
    folder = ROOT / 'raw' / cfg['name']
    marker = folder / 'measurement_begin.json'
    retire_geometry_hook = cfg.get('deep_retire_geometry_hook_after_warmup', False)
    initialize, close = StageEngine.__init__, StageEngine.close

    def initialized(engine, *args, **kwargs):
        result = initialize(engine, *args, **kwargs)
        graphs = getattr(engine, '_autonomous_prefill_graphs', None)
        if graphs is None:
            return result
        role = engine.stage_config.role
        allowed = preferred_geometry(cfg, role)
        counts = {phase: dict(selects=0, existing_hits=0, cold_entries=0, fallbacks=0,
                              policy_bypass=0, captures=0, prepare_host_s=0.)
                  for phase in ('startup', 'measured')}
        geometries = {}
        measured = [False]
        select, prepare = graphs.select, graphs.prepare_inputs
        def phase():
            if not measured[0]:
                measured[0] = marker.exists()
            return 'measured' if measured[0] else 'startup'

        def selected(owner, hidden, metadata):
            stage = phase()
            row = counts[stage]
            q = len(metadata.rows)
            geometry = (q, metadata.max_seqlen_q)
            description = (q, len(metadata.context_lengths), metadata.max_seqlen_q,
                           metadata.cu_seqlens_q is not None, getattr(metadata, 'write_mask', None) is not None)
            key = repr(description)
            entry = geometries.setdefault(key, dict(key=list(description), startup=0, measured=0,
                                                    existing_hits=0, fallbacks=0, cold_entries=0))
            entry[stage] += 1
            row['selects'] += 1
            if cfg['deep_graph_policy'] == 'primary_reserved' and geometry not in allowed:
                owner.fallbacks += 1
                row['fallbacks'] += 1
                row['policy_bypass'] += 1
                entry['fallbacks'] += 1
                return PreparedGraph(metadata)
            previous = len(owner.entries)
            call = select(hidden, metadata)
            if call.entry is None:
                row['fallbacks'] += 1
                entry['fallbacks'] += 1
            elif len(owner.entries) > previous:
                row['cold_entries'] += 1
                entry['cold_entries'] += 1
            else:
                row['existing_hits'] += 1
                entry['existing_hits'] += 1
            return call

        def prepared(owner, call, hidden):
            stage = phase()
            captures = owner.captures
            begin = time.perf_counter()
            result = prepare(call, hidden)
            counts[stage]['prepare_host_s'] += time.perf_counter() - begin
            counts[stage]['captures'] += owner.captures - captures
            return result

        graphs.select = MethodType(selected, graphs)
        graphs.prepare_inputs = MethodType(prepared, graphs)
        engine._deep_graph_audit = dict(role=role, allowed_geometries=sorted(allowed),
            policy=cfg['deep_graph_policy'], budget=cfg['deep_graph_budget'], counts=counts,
            geometries=geometries, initial_allocated_bytes=__import__('torch').cuda.memory_allocated())
        return result

    def closed(engine):
        value = getattr(engine, '_deep_graph_audit', None)
        if value is not None:
            graphs = engine._autonomous_prefill_graphs
            entries = [dict(key=list(key), captured=entry.graph is not None,
                            input_hidden_bytes=entry.hidden.numel() * entry.hidden.element_size())
                       for key, entry in graphs.entries.items()]
            write(folder / f'deep_graph_cache_{os.getpid()}.json', dict(value, entries=entries,
                scope='Exact native packed-prefill cache. Startup and measured counters are separate. Host preparation time is not exclusive kernel time.'))
        if cfg.get('deep_complete_D_geometry_warmup'):
            write(folder / f'deep_warmup_hook_{os.getpid()}.json', dict(role=engine.stage_config.role,
                retire_terminal_hook=retire_geometry_hook,
                terminal_state_cached=getattr(engine.connector, '_deep_D_geometry_warmup_done', False),
                scope='Terminal warm-up state caching only. It does not change native transfer readiness or model computation. The legacy observer remains unless the configuration explicitly enables retirement.'))
        return close(engine)
    StageEngine.__init__, StageEngine.close = initialized, closed

    if not cfg.get('deep_serial_primary_warmup'):
        return
    if cfg.get('deep_complete_D_geometry_warmup'):
        from vllm_rlt.pd.connector import PDStageConnector
        can_schedule = PDStageConnector.can_schedule
        deep_marker = folder / 'deep_D_geometry_phase.json'
        def schedule(owner, request):
            if retire_geometry_hook and getattr(owner, '_deep_D_geometry_warmup_done', False):
                return can_schedule(owner, request)
            if owner.engine.stage_config.role == 'decode' and deep_marker.exists():
                value = __import__('json').loads(deep_marker.read_text())
                if retire_geometry_hook and value['state'] == 'ALL_COMPLETE':
                    owner._deep_D_geometry_warmup_done = True
                    return can_schedule(owner, request)
                if value['state'] == 'RUNNING' and request.sampling_params.seed in value['seeds']:
                    key = tuple(value['seeds'])
                    if getattr(owner, '_deep_released_geometry', None) != key:
                        transfers = [item for item in owner.transfers.values()
                                     if item.request.sampling_params.seed in value['seeds'] and not item.cancelled]
                        if len(transfers) != value['count'] or not all(item.ready for item in transfers):
                            return False
                        owner._deep_released_geometry = key
            return can_schedule(owner, request)
        PDStageConnector.can_schedule = schedule
    serve = rlt_worker.serve
    def warm_primary(engine, inputs):
        from vllm_rlt import SamplingParams
        from ..looped_pd_headroom_recovery.graph_warmup import worker_stats
        seed = cfg['seed'] - 20000
        rid = 'deep_primary_serial_warmup'
        phase_marker = folder / 'native_graph_warmup_phase.json'
        write(phase_marker, dict(state='RUNNING', kind='deep_primary_serial', seeds=[seed], count=1))
        before = worker_stats(engine)
        begin = time.monotonic()
        engine.add_request(rid, list(inputs[0]), SamplingParams(max_tokens=1, temperature=0., seed=seed,
            min_loops=cfg['depth'], max_loops=cfg['depth'], exit_threshold=1., ignore_eos=True))
        finished = False
        while not finished:
            if time.monotonic() - begin > 180:
                raise TimeoutError('Deterministic primary graph warm-up')
            for out in engine.step():
                if out.finished:
                    assert out.request_id == rid and len(out.token_ids) == 1
                    assert set(out.exit_depths) == {cfg['depth']}
                    finished = True
        while getattr(engine, 'transfers', {}):
            if time.monotonic() - begin > 180:
                raise TimeoutError('Deterministic primary ACK drain')
            assert not engine.step()
        after = worker_stats(engine)
        work = sum(v['issued_loop_rows'] - before[k]['issued_loop_rows'] for k, v in after.items())
        assert work == len(inputs[0]) * cfg['depth']
        append(folder / 'deep_primary_warmup.jsonl', dict(host_seconds=time.monotonic() - begin,
            input_tokens=len(inputs[0]), full_recurrent_rows=work, passed=True,
            scope='One exact startup request before ordinary warm-up. Excluded from serving statistics.'))
        write(phase_marker, dict(state='COMPLETE', kind='deep_primary_serial'))

    def warm_D_geometries(engine, inputs):
        from vllm_rlt import SamplingParams
        from ..looped_pd_headroom_recovery.graph_warmup import worker_stats
        contracts = [__import__('json').loads(path.read_text())
                     for path in folder.glob('native_contract_initial_*.json')]
        actual = next(value for value in contracts if value['role'] == 'decode')
        blocks = ((cfg['length'] + 15) // 16) * cfg['depth']
        maximum = min(cfg['D_slots'], actual['KV_blocks'] // blocks)
        allowed = preferred_geometry(cfg, 'decode')
        queries = {query for rows, query in allowed}
        # Native and zero-offload have no D prompt continuation.
        partial = cfg.get('native_frontier') or (cfg.get('native_partial_cut', cfg['depth']) < cfg['depth'])
        if not partial:
            if retire_geometry_hook:
                write(deep_marker, dict(state='ALL_COMPLETE', batches=[], queries=[]))
            return
        batches = sorted({rows // query for rows, query in allowed if rows // query <= maximum})
        for batch in batches:
            seeds = [cfg['seed'] - 30000 - batch * 100 + index for index in range(batch)]
            write(deep_marker, dict(state='RUNNING', seeds=seeds, count=batch,
                scope='Startup-only barrier after exact native import. Not a measured scheduler policy.'))
            before = worker_stats(engine)
            started = time.monotonic()
            active = {}
            for index, seed in enumerate(seeds):
                rid = f'deep_D_geometry_{batch}_{index}'
                active[rid] = index
                engine.add_request(rid, list(inputs[index % len(inputs)]), SamplingParams(
                    max_tokens=1, temperature=0., seed=seed, min_loops=cfg['depth'],
                    max_loops=cfg['depth'], exit_threshold=1., ignore_eos=True))
            while active:
                if time.monotonic() - started > 180:
                    raise TimeoutError('Exact D geometry warm-up or admission barrier')
                for result in engine.step():
                    if result.finished:
                        index = active.pop(result.request_id)
                        assert len(result.token_ids) == 1 and set(result.exit_depths) == {cfg['depth']}
                        append(folder / 'deep_D_geometry_outputs.jsonl', dict(batch=batch, seed=seeds[index],
                            output_ids=list(result.token_ids), exit_depths=list(result.exit_depths), warmup=True))
            while getattr(engine, 'transfers', {}):
                if time.monotonic() - started > 180:
                    raise TimeoutError('Exact D geometry ACK drain')
                assert not engine.step()
            after = worker_stats(engine)
            work = sum(v['issued_loop_rows'] - before[k]['issued_loop_rows'] for k, v in after.items())
            assert work == batch * cfg['length'] * cfg['depth']
            append(folder / 'deep_D_geometry_warmup.jsonl', dict(batch=batch, host_seconds=time.monotonic() - started,
                full_recurrent_rows=work, native_KV_credit_limit=maximum, passed=True,
                scope='Exact full-prompt startup work. Excluded from measured latency, throughput, and migration.'))
            write(deep_marker, dict(state='COMPLETE', count=batch))
        write(deep_marker, dict(state='ALL_COMPLETE', batches=batches, queries=sorted(queries)))

    def served(engine, inputs, args, run, warmup=False, contract=None):
        if warmup:
            warm_primary(engine, inputs)
        result = serve(engine, inputs, args, run, warmup, contract)
        if warmup and cfg.get('deep_complete_D_geometry_warmup'):
            warm_D_geometries(engine, inputs)
        return result
    rlt_worker.serve = served
