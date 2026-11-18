"""Reuse frozen client metrics and audit actual completed work."""
import collections
import json
from .common import ROOT,BASE,bind_legacy,write,digest


def output_gate(name):
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    path=ROOT/cfg.get('historical_output_reference','raw/HISTORICAL_original_outputs/requests.jsonl')
    reference={r['prompt_id']:r for r in map(json.loads,path.read_text().splitlines())}
    rows=[json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
    failures=[]
    for row in rows:
        old=reference.get(row['prompt_id'])
        if old is None or old['input_ids']!=row['input_ids']:
            failures.append(dict(sample_id=row['sample_id'],reason='INCOMPARABLE_PROMPT'))
        elif row['output_ids']!=old['output_ids'][:row['output_tokens']]:
            failures.append(dict(sample_id=row['sample_id'],reason='GREEDY_OUTPUT_MISMATCH'))
    value=dict(name=name,passed=len(rows)==cfg['count'] and not failures,
        count=len(rows),failures=failures,reference=str(path),evidence='CORRECTNESS_ONLY',
        scope='Preserved original BF16 reference outputs only. No historical latency reuse.')
    write(ROOT/'analysis'/(name+'_output_gate.json'),value)
    return value


def metrics(name):
    bind_legacy()
    from ..dense_ar_layer_token_pd import analysis as prior
    prior.ROOT,prior.BASE,prior.bind_previous=ROOT,BASE,bind_legacy
    from ..dense_ar_layer_token_pd import transport_audit
    transport_audit.ROOT=ROOT
    from ..duetserve_agg_decode import analyze as agg
    agg.output_gate=output_gate
    value=prior.metrics(name)
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    common_SLO=ROOT/'configs/common_output_sweep_SLO.json'
    if common_SLO.exists():
        from ..dense_ar_slo_layer_pd.analyze import request_SLOs
        doc=json.loads(common_SLO.read_text())
        limits={key:doc[key] for key in ('TTFT_ms','ITL_ms','TPOT_ms')}
        thresholds={str(cap):{regime:limits for regime in ('Balanced','TTFT_priority','Decode_priority')}
            for cap in (16,32,64,128,200)}
        rows=[json.loads(line) for line in (ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()]
        intervals=[[1000*(b['time']-a['time']) for a,b in zip(r['token_events'],r['token_events'][1:])] for r in rows]
        value['SLO_regimes']['Common_Balanced_OutputSweep']=request_SLOs(
            cfg,rows,intervals,thresholds,value['observation_seconds'])['Balanced']
        value['secondary_common_SLO_SHA256']=digest(common_SLO)
    value.update(fixed_layers=cfg.get('static_layers'),common_D_admission_cap=cfg['D_admission_batch'],
        source_macro_limit=cfg['P_group_size'],both_workers_B8192_S1024=True,
        matched_primary_requests=cfg['count']%9==0,
        source_publication=BASE,comparison_scope='Fixed partition, common admission predicate; natural GPU geometry can differ')
    if cfg['arm']=='PD_G1' and not cfg.get('correctness_only'):
        if cfg.get('token_axis') or cfg['policy']=='STATIC':
            expected=cfg['count']*8000*cfg['static_layers']
            work=value['work_accounting']
            assert work['completed_migrated_requests']==cfg['count'], 'Unmatched fallback'
            assert work['D_GEMM_token_layer_rows']==expected, 'Incomplete or unmatched rows'
            assert work['P_GEMM_token_layer_rows']==cfg['count']*8000*36-expected
            value['exact_GEMM_row_gate']=True
    value['unified_worker_gate']=json.loads((ROOT/'raw'/name/'unified_worker_gate.json').read_text())['passed']
    if cfg['arm']=='PD_G1':
        folder=ROOT/'raw'/name
        if cfg.get('common_prospective_KV_reservation'):
            ledger=json.loads((folder/'reservation_lifetime_decode.json').read_text())
            reservations=[r for r in ledger['history'] if r['action']=='RESERVE']
            granted=collections.Counter(rid for row in reservations for rid in row['requests'])
            retired=collections.Counter(r['request'] for r in ledger['history'] if r['action']=='FINISH')
            expected_ids=collections.Counter({'q'+str(i):1 for i in range(cfg['count'])})
            expected_cut=36-cfg['static_layers'] if cfg['policy']=='STATIC' else 36
            value['common_reservation_gate']=dict(passed=ledger['passed'] and granted==expected_ids and
                retired==expected_ids and all(r['cut']==expected_cut for r in reservations),
                granted_requests=sum(granted.values()),retired_requests=sum(retired.values()),
                generations=len(reservations),expected_cut=expected_cut,
                evidence='TRACE_ANALYSIS',
                scope='The same existing prospective KV ledger is used by Native, Layer and Token. Every measured request has one grant, one committed allocation and one retirement. Worker validation checks its generation and cut. This verifies normal completion, not every failure/cancellation scenario.')
            assert value['common_reservation_gate']['passed'],value['common_reservation_gate']
        observed={}
        role_shapes={}
        for role in ('prefill','decode'):
            counts=json.loads((folder/f'measured_geometry_{role}.json').read_text())['counts']
            shapes=[(json.loads(key),n) for key,n in counts.items()]
            role_shapes[role]=shapes
            observed[role]=dict(
                prompt_phase_query_rows=sum(s['query_tokens']*n for s,n in shapes if s['prefilling']),
                decode_phase_query_rows=sum(s['query_tokens']*n for s,n in shapes if not s['prefilling']),
                total_query_rows=sum(s['query_tokens']*n for s,n in shapes))
        suffixes=cfg.get('token_suffix_by_request',[cfg.get('token_suffix',0)]*cfg['count'])
        expected_P=sum(8000-s for s in suffixes) if cfg.get('token_axis') else cfg['count']*8000
        expected_D=(sum(suffixes) if cfg.get('token_axis') else cfg['count']*8000 if cfg['policy']=='STATIC' else 0)
        expected_D+=sum(c-1 for c in cfg['request_caps'])
        checks=dict(P_queries=observed['prefill']['total_query_rows']==expected_P,
            D_queries=observed['decode']['total_query_rows']==expected_D,
            no_P_decode_queries=observed['prefill']['decode_phase_query_rows']==0)
        value['actual_query_coverage_gate']=dict(passed=all(checks.values()),checks=checks,
            observed=observed,expected_P=expected_P,expected_D=expected_D,
            evidence='MEASURED',measurement_class='INPUT_GEOMETRY_WITH_STRUCTURAL_LAYER_ACCOUNTING',
            scope='Actual unpadded query counts in measured GPU inputs. Prompt and generated Decode queries are separate. Prior state gates establish actual layer execution. This is not exclusive kernel timing.')
        assert all(checks.values()),value['actual_query_coverage_gate']
        # Aggregate rows cannot detect a missing span replaced by a duplicate.
        # Audit each original prompt position at the scheduler boundary too.
        spans=collections.defaultdict(list)
        for path in folder.glob('native_prefill_*.json'):
            for span in json.loads(path.read_text())['prompt_spans']:
                rid=span['request_id'].split('-')[0]
                if rid.startswith('q') and rid[1:].isdigit():
                    spans[rid].append(span)
        failures=[]
        details=[]
        for index in range(cfg['count']):
            rid='q'+str(index)
            cursor=0
            length=8000-suffixes[index] if cfg.get('token_axis') else 8000
            for span in sorted(spans[rid],key=lambda r:r['time']):
                if span['start']!=cursor or span['count']<=0:
                    failures.append(dict(request_id=rid,cursor=cursor,span=span))
                cursor=span['start']+span['count']
            if cursor!=length:
                failures.append(dict(request_id=rid,final_cursor=cursor,expected=length))
            details.append(dict(request_id=rid,P_prompt_queries=cursor,
                D_prompt_queries=suffixes[index] if cfg.get('token_axis') else 8000 if cfg['policy']=='STATIC' else 0,
                suffix_tokens=suffixes[index] if cfg.get('token_axis') else None,
                P_transformer_layers=36-cfg['static_layers'] if cfg['policy']=='STATIC' else 36,
                D_prompt_transformer_layers=cfg['static_layers'] if cfg['policy']=='STATIC' else 36 if cfg.get('token_axis') else 0,
                source_fragments=[dict(start=s['start'],count=s['count']) for s in spans[rid]]))
        value['per_request_P_query_coverage']=dict(passed=not failures,failures=failures,
            requests=len(spans),chunks=sum(map(len,spans.values())),
            rows=details,
            evidence='TRACE_ANALYSIS',scope='Original prompt positions at the actual source scheduler. Every prompt query must occur exactly once. Warm-up IDs are excluded.')
        assert not failures,value['per_request_P_query_coverage']
        P_layers=36-cfg['static_layers'] if cfg['policy']=='STATIC' else 36
        D_suffix_layers=cfg['static_layers'] if cfg['policy']=='STATIC' else 36
        value['observed_batch_structural_projection_calls']=dict(
            P_model_inputs=sum(n for s,n in role_shapes['prefill']),
            P_major_projection_calls=4*P_layers*sum(n for s,n in role_shapes['prefill']),
            D_major_projection_calls=4*sum(n*(D_suffix_layers if s['prefilling'] else 36) for s,n in role_shapes['decode']),
            evidence='STRUCTURAL_ACCOUNTING',
            scope='Four Transformer projections per executed layer, counted from actual prepared batches. This is not a CUDA kernel-launch count. It excludes LM head, sampling, normalization and attention. Chunk fragmentation can change these calls despite matched aggregate token-layer rows.')
        if value.get('coordinator_queue_observations'):
            value['coordinator_queue_observations']['scope']='All unified PD arms hold external requests in the coordinator. P_HOST_COMPUTE_PENDING is an external wait state. The common outstanding-window, KV and reservation gates can block admission. Queue growth alone is not evidence of P compute saturation.'
    assert value['output_gate']['passed'] and value['unified_worker_gate']
    value['analysis_schema_version']=3
    write(ROOT/'analysis'/(name+'_metrics.json'),value)
    return value
