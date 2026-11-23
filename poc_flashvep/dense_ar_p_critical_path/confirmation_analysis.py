"""Report all paired starts,including quality and decode-gate failures."""
from collections import defaultdict
from pathlib import Path
import json
import time
import numpy as np
from .common import ROOT,write,digest,bind
from .critical_analysis import credits,csv_save

FIELDS=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms','max_ITL_ms',
    'generated_output_tokens_per_second','completed_requests_per_second','SLO_qualified_output_tokens_per_second',
    'TTFT_SLO_attainment','every_ITL_SLO_attainment','decode_SLO_attainment','Joint_SLO_attainment','output_tokens')
P_STAGES=('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms','P_prepare_to_ready_ms')


def analyze(plan_name='c3_credit_retirement_confirmation_r1.json',output_stem='c3_credit_confirmation'):
    bind()
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from ..dense_ar_output_aware_v2.c3_analysis import actual_quality,means_ci
    from ..dense_ar_output_aware_v2.statistics import ratio,paired_difference,safety
    from ..dense_ar_output_aware_v2.work_contract import executed_prompt_work
    from ..dense_ar_longbench400_pd.qualification import audit
    from ..dense_ar_longbench400_pd.service_accounting import analyze as service
    plan_path=ROOT/'jobs'/plan_name;plan=json.loads(plan_path.read_text())
    terminal_unrun={}
    for stage_name in [plan_name,*plan.get('execution_stage_plans',[])]:
        for lane in range(4):
            complete=ROOT/'state'/(Path(stage_name).stem+f'_lane{lane}_complete.json')
            if complete.exists():
                for row in json.loads(complete.read_text()).get('results',[]):
                    if row['status']=='NOT_RUN':terminal_unrun[row['name']]=row.get('reason','TERMINAL_LANE_NOT_RUN')
    marker_names={'output192_D_cap_intervention_v1.json':'output192_D_cap_not_run.json',
        'coordinator_hygiene_c3_confirmation_warm_regime_v8.json':'coordinator_hygiene_warm_regime_not_run_v8.json',
        'coordinator_hygiene_c3_confirmation_warm_regime_v9.json':'coordinator_hygiene_warm_regime_not_run_v9.json',
        'coordinator_hygiene_c3_confirmation_warm_regime_v10.json':'coordinator_hygiene_warm_regime_not_run_v10.json'}
    marker=ROOT/'state'/marker_names.get(plan_name,'nonexistent_optional_marker.json')
    stopped=json.loads(marker.read_text()) if marker.exists() else None
    starts=[];by_mode=defaultdict(dict);statuses=[]
    for block in plan['blocks']:
        values={}
        for label,name in block['arms'].items():
            folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
            if not receipt.exists():
                reason=terminal_unrun.get(name) or (stopped.get('reason') if stopped and
                    block['mode'] in stopped.get('modes',[block['mode']]) else None)
                statuses.append(dict(mode=block['mode'],start=block['start'],label=label,name=name,
                    status='NOT_RUN_BY_RECORDED_DECISION' if reason else 'NOT_RUN_OR_RUNNING',reason=reason));continue
            if not json.loads(receipt.read_text())['complete']:
                statuses.append(dict(mode=block['mode'],start=block['start'],label=label,name=name,status='FAILED'));continue
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());m=summarize(name);q=actual_quality(name)
            runtime={role:json.loads((folder/f'actual_runtime_{role}.json').read_text()) for role in ('prefill','decode')}
            assert all(r['scheduler_B']==8192 and r['scheduler_S']==1024 and r['chunked_prefill'] and
                r['model_layers']==36 and r['parameter_dtypes']==['torch.bfloat16'] for r in runtime.values())
            for role in ('prefill','decode'):assert json.loads((folder/f'KV_lifetime_{role}.json').read_text())['passed']
            state=None
            if cfg.get('axis_kind'):
                state=audit(name);assert state['passed'] and state['decode_full36_observation_complete']
            path=service(name);credit=credits(folder);data=json.loads((ROOT/'data'/cfg['dataset']).read_text())
            work=executed_prompt_work(cfg,folder,data)
            mean_P=sum(path['means'][k] for k in P_STAGES)
            runtime_policy=dict(passed=True,status='NO_ADDITIONAL_CACHE_REGIME_REQUIRED')
            if name.startswith('PC_WarmC3'):
                from .warm_serving_followup import timed_regime_check
                runtime_policy=timed_regime_check(name,label,block['start'],cfg)
            row=dict(mode=block['mode'],start=block['start'],seed=block['seed'],QPS=block['qps'],label=label,name=name,
                GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'],source_SHA256=cfg['source_capsule_SHA256'],
                model_revision='b968826d9c46dd6066d109eabc6255188de91218',B=8192,S=1024,KV_GiB=42,KV_block=128,
                actual_geometry=m['actual_geometry'],source_credit=cfg.get('continuous_source_credit_limit',0),
                D_running_admission_cap=cfg.get('D_running_admission_cap'),axis_kind=cfg.get('axis_kind','Native'),
                axis_cut=cfg.get('axis_cut'),hybrid_suffix_fraction=cfg.get('hybrid_suffix_fraction'),
                safe_retirement_enabled=cfg.get('source_credit_release_on_retirement',False),
                P_compile=cfg.get('P_compile_prefix',False),D_compile=cfg.get('D_compile_partial',False),
                quality_passed=q['passed'],quality_CI95_percentage_points=q['CI95_percentage_points'],
                runtime_policy=runtime_policy,runtime_policy_passed=runtime_policy['passed'],
                quality_scope=q.get('scope'),output_contract='FIXED_LENGTH_DIAGNOSTIC_REPLAY' if cfg['ignore_eos'] else 'NATURAL_EOS',
                D_request_registration_deferred=cfg.get('defer_D_request_registration',False),
                mean_P_arrival_to_ready_ms=mean_P,mean_host_path_ms=path['means'],source_window=credit,
                actual_executed_work=work,state_scope='Fresh full400 composition plus clean-run query/layer coverage and lifetime. No full-KV hashing during timing.',
                actual_runtime=runtime,scope='Finite cohort. No sustainable-capacity or exclusive-GPU-time claim.',
                evidence='DIAGNOSTIC_RUNTIME_POLICY_FAILURE' if not runtime_policy['passed'] else
                    'MEASURED_ONLINE' if q['passed'] else 'DIAGNOSTIC_QUALITY_FAILURE',**{k:m[k] for k in FIELDS})
            starts.append(row);values[label]=dict(metric=m,config=cfg,path=path,credit=credit,row=row)
            statuses.append(dict(mode=block['mode'],start=block['start'],label=label,name=name,
                status='INVALID_RUNTIME' if not runtime_policy['passed'] else 'COMPLETE' if q['passed'] else 'INVALID_QUALITY'))
        by_mode[block['mode']][block['start']]=values
    comparisons=[];cells=[]
    pairs=[('OptimizedLayer','OriginalLayer'),('OptimizedHybrid','OriginalHybrid'),
        ('OriginalLayer','StockToken'),('OptimizedLayer','StockToken'),('OriginalHybrid','StockToken'),
        ('OptimizedHybrid','StockToken'),('OptimizedHybrid','OptimizedLayer')]
    pairs += [(label,ref) for label in ('OriginalLayer','OptimizedLayer','OriginalHybrid','OptimizedHybrid')
        for ref in ('NativeStrong','NativeCommon2')]
    pairs += [tuple(value) for value in plan.get('extra_pairs',[])]
    for mode,blocks in by_mode.items():
        for label in plan['arms']:
            matched=[value[label] for start,value in sorted(blocks.items()) if label in value]
            if not matched:continue
            m=[v['metric'] for v in matched];all_quality=all(v['row']['quality_passed'] for v in matched)
            refs={key:[value[key]['metric'] for start,value in sorted(blocks.items()) if label in value and key in value]
                for key in ('NativeStrong','NativeCommon2')}
            gate=safety(m,{'NativeStrong':refs['NativeStrong'],'NativeCommon':refs['NativeCommon2']}) if all(len(v)==len(m) for v in refs.values()) else dict(qualified=False,reason='MISSING_PAIRED_REFERENCES')
            additional={}
            extra_references=plan.get('additional_decode_references',['NativeCommon16'] if 'NativeCommon16' in plan['arms'] else [])
            for reference in extra_references:
                values=[value[reference]['metric'] for start,value in sorted(blocks.items()) if label in value and reference in value]
                additional[reference]=safety(m,{'NativeStrong':refs['NativeStrong'],'NativeCommon':values}) if len(values)==len(m) and len(refs['NativeStrong'])==len(m) else dict(qualified=False,reason='MISSING_PAIRED_REFERENCES')
            cells.append(dict(mode=mode,label=label,independent_starts=len(m),quality_passed_all_starts=all_quality,
                runtime_policy_passed_all_starts=all(v['row']['runtime_policy_passed'] for v in matched),
                fields={k:means_ci([v[k] for v in m]) for k in FIELDS},
                mean_P_arrival_to_ready_ms=means_ci([v['row']['mean_P_arrival_to_ready_ms'] for v in matched]),decode_gate=gate,
                additional_decode_gates=additional,
                evidence='DIAGNOSTIC_RUNTIME_POLICY_FAILURE' if not all(v['row']['runtime_policy_passed'] for v in matched) else
                    'MEASURED_ONLINE' if all_quality else 'DIAGNOSTIC_QUALITY_FAILURE'))
        for left,right in pairs:
            matched=[(value[left],value[right]) for start,value in sorted(blocks.items()) if left in value and right in value]
            if not matched:continue
            same_output=[]
            for a,b in matched:
                for field in ('dataset_SHA256','arrival_offsets','gpus','qps','seed','output_cap','ignore_eos'):
                    assert a['config'][field]==b['config'][field],(left,right,field)
                def output_table(v):
                    return {r['prompt_id']:r['output_tokens'] for r in map(json.loads,
                        (ROOT/'raw'/v['row']['name']/'requests.jsonl').read_text().splitlines())}
                ta,tb=output_table(a),output_table(b);assert set(ta)==set(tb)
                same_output.append(ta==tb)
                if a['config']['ignore_eos']:assert ta==tb
            delta={key:paired_difference([a['path']['means'][key] for a,b in matched],
                [b['path']['means'][key] for a,b in matched]) for key in matched[0][0]['path']['means']}
            comparisons.append(dict(mode=mode,left=left,right=right,paired_starts=len(matched),
                quality_passed_all_paired_starts=all(a['row']['quality_passed'] and b['row']['quality_passed'] for a,b in matched),
                runtime_policy_passed_all_paired_starts=all(a['row']['runtime_policy_passed'] and b['row']['runtime_policy_passed'] for a,b in matched),
                per_start_request_output_work_exact=same_output,
                ratios={key:ratio([a['metric'][key] for a,b in matched],[b['metric'][key] for a,b in matched]) for key in FIELDS},
                P_critical_path_ratio=ratio([a['row']['mean_P_arrival_to_ready_ms'] for a,b in matched],
                    [b['row']['mean_P_arrival_to_ready_ms'] for a,b in matched]),
                P_critical_path_delta_ms=paired_difference([a['row']['mean_P_arrival_to_ready_ms'] for a,b in matched],
                    [b['row']['mean_P_arrival_to_ready_ms'] for a,b in matched]),mean_host_delta_ms=delta,
                evidence='MEASURED_ONLINE_PAIRED_START_COMPARISON' if all(a['row']['runtime_policy_passed'] and b['row']['runtime_policy_passed'] for a,b in matched)
                    else 'DIAGNOSTIC_RUNTIME_POLICY_FAILURE',
                scope='Same physical pair and request/arrival contract. Natural output work can differ. '
                    'A shorter coordinator queue alone is not a shorter P critical path. Host deltas are additive means,not p95 GPU attribution.'))
    result=dict(time=time.time(),plan_SHA256=digest(plan_path),starts=starts,cells=cells,comparisons=comparisons,execution_status=statuses,
        scope=plan.get('analysis_scope','No failed or unrun start is removed from the registry. A candidate needs all three valid quality-passing starts and both Native decode references. '
            'Performance CIs use independent server starts only. Pointwise empirical paired95% bootstrap. '
            'Best-vs-best Layer2/StockToken8 is not matched GEMM/Attention work or a common-policy axis-only intervention. '
            'Only Original/Optimized within one axis is the one-factor safe-retirement intervention. No dynamic safe-window claim.'))
    write(ROOT/'analysis'/(output_stem+'_analysis.json'),result)
    csv_save(ROOT/'figures'/(output_stem+'_per_start.csv'),starts)
    csv_save(ROOT/'figures'/(output_stem+'_cells.csv'),cells)
    csv_save(ROOT/'figures'/(output_stem+'_paired_comparisons.csv'),comparisons)
    csv_save(ROOT/'figures'/(output_stem+'_execution_status.csv'),statuses)
    print('C3_CONFIRMATION_ANALYSIS',len(starts),len(comparisons),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default='c3_credit_retirement_confirmation_r1.json')
    parser.add_argument('--stem',default='c3_credit_confirmation');args=parser.parse_args()
    analyze(args.plan,args.stem)
