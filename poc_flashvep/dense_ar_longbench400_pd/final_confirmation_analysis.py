"""Judge serving gains and decode protection with paired independent starts."""
import json
import time
import numpy as np
from .common import ROOT,write
from .metrics import summarize
from .confirmation_analysis import ratio
from .final_continuous_selection import policy


FIELDS=('TTFT_p50_ms','TTFT_p95_ms','TTFT_p99_ms','TTFT_mean_ms',
    'mean_TBT_ms','ITL_p99_ms','max_ITL_ms','TPOT_p50_ms','TPOT_p95_ms','TPOT_p99_ms','E2E_p95_ms',
    'generated_output_tokens_per_second','completed_requests_per_second',
    'Joint_SLO_attainment','SLO_qualified_output_tokens_per_second','decode_SLO_attainment',
    'actual_arrival_QPS','actual_dispatch_QPS','output_tokens')


def complete_metric(value,name):
    # The inherited metric writer publishes its base record before this
    # campaign adds provenance and geometry. A concurrent reader can observe
    # that valid but intermediate schema. Recompute from immutable raw events.
    required={'runtime_source_archive_SHA256','dataset_SHA256','actual_geometry'}
    return value if required<=value.keys() else summarize(name)


def load(name,source):
    folder=ROOT/'raw'/name
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists():
        return None,None,'NOT_RUN'
    result=json.loads(receipt.read_text())
    if not result['complete']:
        return None,None,'FAILED'
    assert result['source_SHA256']==source
    cfg=json.loads((folder/'configuration.json').read_text())
    metric_path=ROOT/'analysis'/(name+'_metrics.json')
    m=complete_metric(json.loads(metric_path.read_text()),name) if metric_path.exists() else summarize(name)
    assert m['runtime_source_archive_SHA256']==source and m['dataset_SHA256']==cfg['dataset_SHA256']
    gate=ROOT/'analysis'/(name+'_quality_gate.json')
    if not gate.exists():
        return None,cfg,'PENDING_TASK_QUALITY'
    q=json.loads(gate.read_text())
    if not q['passed']:
        return m,cfg,'FAILED_TASK_QUALITY'
    assert m['n']==cfg['count'] and not cfg['ignore_eos'] and not cfg['controlled_pipeline']
    for role in ('prefill','decode'):
        path=folder/f'KV_lifetime_{role}.json'
        if path.exists():
            assert json.loads(path.read_text())['passed']
    return m,cfg,'MEASURED_ONLINE'


def main():
    path=ROOT/'configs/final_continuous_execution_plan_v1.json'
    plan=json.loads(path.read_text())
    groups={}
    for cell in plan['cells']:
        groups.setdefault((cell['workload'],cell['QPS']),[]).append(cell)
    rows=[]
    for (workload,qps),cells in sorted(groups.items()):
        cells=sorted(cells,key=lambda r:r['start'])
        arms=sorted(cells[0]['jobs'])
        assert all(sorted(c['jobs'])==arms for c in cells)
        row=dict(workload=workload,QPS=qps,required_starts=len(cells),starts=[],
            status='NOT_RUN',arm_results={},comparisons={})
        values={a:[] for a in arms}
        configurations={a:[] for a in arms}
        all_valid=True
        for cell in cells:
            block={}
            for arm,name in cell['jobs'].items():
                m,cfg,status=load(name,plan['source_capsule_SHA256'])
                block[arm]=dict(name=name,status=status)
                if status=='MEASURED_ONLINE':
                    values[arm].append(m)
                    configurations[arm].append(cfg)
                else:
                    all_valid=False
            if all(v['status']=='MEASURED_ONLINE' for v in block.values()):
                base=configurations['NativeStrong'][-1]
                for arm in arms:
                    cfg=configurations[arm][-1]
                    assert all(cfg[k]==base[k] for k in ('dataset_SHA256','arrival_offsets','seed','gpus',
                        'output_cap','ignore_eos','token_budget','max_num_seqs'))
            row['starts'].append(dict(seed=cell['seed'],GPU_pair=cell['GPU_pair'],arms=block))
        for arm in arms:
            v=values[arm]
            row['arm_results'][arm]=dict(valid_starts=len(v),means={f:float(np.mean([r[f] for r in v]))
                for f in FIELDS} if v else {},
                actual_geometry=[r['actual_geometry'] for r in v],
                per_start_output_tokens=[r['output_tokens'] for r in v],
                runtime_contract=[dict(execution_family=c['execution_family'],
                    Source_credit=c.get('continuous_source_credit_limit',0),
                    policy=list(policy(c)),axis=c.get('axis_kind','Native'),cut=c.get('axis_cut',0),
                    suffix_fraction=c.get('hybrid_suffix_fraction'),fractional_token_work=c.get('fractional_token_work'),
                    hidden_slots=[c['hidden_source_slots'],c['hidden_destination_slots']],
                    B=c['token_budget'],S=c['max_num_seqs'],block=c['KV_block_tokens'])
                    for c in configurations[arm]])
        if any(values.values()):
            row['status']='PARTIAL_ONLINE_RESULTS'
        if all_valid:
            row['status']='MEASURED_ONLINE_CONFIRMED' if len(cells)>=3 else 'MEASURED_ONLINE_SINGLE_START'
        # A failed optional control must remain visible. It must not prevent an
        # otherwise complete, paired arm/reference block from being evaluated.
        if any(len(v)==len(cells) for v in values.values()):
            for arm in arms:
                if not arm.startswith('Best') and arm!='TokenCap2Control':
                    continue
                if len(values[arm])!=len(cells):
                    continue
                cfg=configurations[arm][0]
                common=[a for a in arms if a.startswith('NativeCommon') and configurations[a]
                    and policy(configurations[a][0])==policy(cfg)]
                assert len(common)<=1,(arm,common)
                refs=['NativeStrong']+common
                if 'NativeTuned' in arms:
                    refs.append('NativeTuned')
                if arm in ('BestLayer','BestHybrid') and 'BestToken' in arms:
                    refs.append('BestToken')
                if arm in ('BestLayer','BestHybrid') and 'TokenCap2Control' in arms:
                    refs.append('TokenCap2Control')
                if arm=='TokenCap2Control':
                    refs.append('BestToken')
                if arm=='BestHybrid' and 'BestLayer' in arms:
                    refs.append('BestLayer')
                comparisons={}
                for ref in refs:
                    if len(values[ref])!=len(cells):
                        continue
                    for a,b in zip(configurations[arm],configurations[ref]):
                        assert all(a[k]==b[k] for k in ('dataset_SHA256','arrival_offsets','seed','gpus',
                            'output_cap','ignore_eos','token_budget','max_num_seqs'))
                    tests={}
                    for field in FIELDS:
                        a=[r[field] for r in values[arm]]
                        b=[r[field] for r in values[ref]]
                        tests[field]=ratio(a,b) if len(cells)>=3 else dict(
                            estimate=a[0]/b[0] if b[0]>0 else None,CI95=None,starts=1,
                            qualifier='Single start. No confidence-interval claim.')
                    decode=None if len(cells)<3 else all(tests[f] is not None and tests[f]['CI95'][1]<=1.05
                        for f in ('ITL_p99_ms','TPOT_p95_ms'))
                    comparisons[ref]=dict(ratios=tests,decode_5percent_CI_pass=decode,
                        absolute_decode_attainment_95percent=all(r['decode_SLO_attainment']>=.95 for r in values[arm]),
                        TTFT_p95_CI_win=None if len(cells)<3 else tests['TTFT_p95_ms']['CI95'][1]<1,
                        goodput_CI_win=None if len(cells)<3 or tests['SLO_qualified_output_tokens_per_second'] is None
                            else tests['SLO_qualified_output_tokens_per_second']['CI95'][0]>1)
                row['comparisons'][arm]=comparisons
                native_refs=['NativeStrong']+common
                complete_native_controls=len(common)==1 and all(r in comparisons for r in native_refs)
                confirmed=len(cells)>=3 and complete_native_controls
                safe=confirmed and all(comparisons[r]['decode_5percent_CI_pass']
                    and comparisons[r]['absolute_decode_attainment_95percent'] for r in native_refs)
                row['arm_results'][arm]['serving_qualification']=dict(
                    both_native_controls_complete=complete_native_controls,
                    independent_start_safety_confirmed=bool(safe),
                    TTFT_CI_win_over_both_native_controls=bool(safe and all(
                        comparisons[r]['TTFT_p95_CI_win'] for r in native_refs)),
                    goodput_CI_win_over_both_native_controls=bool(safe and all(
                        comparisons[r]['goodput_CI_win'] for r in native_refs)),
                    scope='False can mean failure or insufficient evidence. Read the paired tests and start statuses. '
                        'These flags do not assert a win over the best pure axis or a dynamic-scheduler gain.')
        row['missing_or_invalid_arms']=[a for a in arms if len(values[a])!=len(cells)]
        rows.append(row)
    write(ROOT/'analysis/final_continuous_strong_confirmation_v1.json',dict(time=time.time(),cells=rows,
        source_capsule_SHA256=plan['source_capsule_SHA256'],evidence='MEASURED_ONLINE',
        scope='Frozen discovery choices. Paired independent server-start/seed blocks. '
            'All Heldout200 confirmations use GPUs0/1. Main400 three-start blocks span0/1,2/3,4/5. '
            'Pointwise unadjusted empirical CIs with27 resamples for3 starts. '
            'Actual natural output counts may differ under the declared BI0 task-quality contract. '
            'Report those counts,not the C1 reference lengths as identical generated work. '
            'Both NativeStrong and policy-matched Native protect ITL and TPOT. '
            'Failed,quality-invalid,pending and missing starts do not become qualified points. '
            'Finite-cohort throughput is not sustainable capacity. A static gain is not an active-decode safe-window or dynamic-selector gain.'))
    print([(r['workload'],r['QPS'],r['status']) for r in rows],flush=True)


if __name__=='__main__':
    main()
