"""Recompute continuous-serving tables without hiding quality or safety failures."""
import argparse
import collections
import itertools
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .statistics import ratio,safety
from .c0_analysis import FIELDS,csv_save,load_metric

def means_ci(values):
    a=np.asarray(values,float);ci=None
    if len(a)>=3:
        indices=np.asarray(list(itertools.product(range(len(a)),repeat=len(a)))) if len(a)<=5 else np.random.default_rng(2026112201).integers(0,len(a),(20000,len(a)))
        ci=np.percentile(a[indices].mean(axis=1),[2.5,97.5]).tolist()
    return dict(mean=float(a.mean()),CI95=ci,starts=len(a))

def actual_quality(name):
    from .quality_gate import compare
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    if cfg.get('ignore_eos'):
        proof=json.loads((ROOT/'state'/cfg['qualification_gate_v2']).read_text())
        return dict(passed=proof['quality_passed'],CI95_percentage_points=None,exact_greedy_requests=None,
            scope='Inherit the validated natural-generation composition. Do not score forced post-EOS output as a new natural answer.')
    path=ROOT/'analysis'/(name+'_quality_gate.json')
    return json.loads(path.read_text()) if path.exists() else compare(name)

def raw_rows(name):return list(map(json.loads,(ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()))

def analyze(plan_name='c3_main_serving_v2.json'):
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.qualification import audit as coverage_audit
    path=ROOT/'jobs'/plan_name;plan=json.loads(path.read_text());starts=[];groups=collections.defaultdict(list);complete_blocks=[]
    for block in plan['blocks']:
        metrics={};statuses={}
        for label,name in block['arms'].items():
            m,status=load_metric(name);statuses[label]=status
            if m is None:continue
            q=actual_quality(name);cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());rows=raw_rows(name)
            from .derived_metrics import output_contract
            m=dict(m,output_contract=output_contract(cfg))
            runtime={role:json.loads(p.read_text()) for role in ('prefill','decode','replica0','replica1')
                if (p:=ROOT/'raw'/name/f'actual_runtime_{role}.json').exists()}
            assert runtime and all(r['scheduler_B']==8192 and r['scheduler_S']==1024 and r['chunked_prefill'] and
                r['model_layers']==36 and r['parameter_dtypes']==['torch.bfloat16'] for r in runtime.values())
            coverage=None
            partition=None
            from .work_contract import executed_prompt_work
            work=executed_prompt_work(cfg,ROOT/'raw'/name,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
            if cfg.get('axis_kind'):
                from .work_contract import actual_partitions
                partition=actual_partitions(cfg,ROOT/'raw'/name,json.loads((ROOT/'data'/cfg['dataset']).read_text()))
                cpath=ROOT/'analysis'/(name+'_coverage.json')
                coverage=json.loads(cpath.read_text()) if cpath.exists() else coverage_audit(name)
                assert coverage['passed'] and coverage['decode_full36_observation_complete']
            value=dict(split=block['split'],QPS=block['qps'],start=block['start'],seed=block['seed'],label=label,name=name,
                GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'],B=8192,S=1024,actual_geometry=m['actual_geometry'],
                source_SHA256=m['runtime_source_archive_SHA256'],source_credit=m['Source_credit'],KV_block=cfg['KV_block_tokens'],
                D_running_admission_cap=cfg.get('D_running_admission_cap'),P_compile_prefix=cfg.get('P_compile_prefix',False),
                D_compile_partial=cfg.get('D_compile_partial',False),fuse_short_hybrid=cfg.get('fuse_short_hybrid',False),
                hidden_source_slots=cfg.get('hidden_source_slots'),hidden_destination_slots=cfg.get('hidden_destination_slots'),
                token_stock_model=cfg.get('token_stock_model',False),quality_passed=q['passed'],
                fractional_token_work=cfg.get('fractional_token_work'),aggregate_work_audit=cfg.get('aggregate_work_audit'),
                quality_CI95_percentage_points=q['CI95_percentage_points'],quality_exact_greedy_requests=q['exact_greedy_requests'],
                request_count=m['n'],output_contract=m['output_contract'],
                client_ITL_interval_count=m['ITL_interval_count'],
                requests_with_decode_intervals=sum(r['output_tokens']>=2 for r in rows),
                requests_without_decode_intervals=sum(r['output_tokens']<2 for r in rows),
                first_delivery_multi_token_requests=m['first_event_multi_token_requests'],
                token_delivery_scope=m['token_delivery_scope'],
                timing_run_query_layer_coverage_passed=coverage['passed'] if coverage else None,
                actual_static_partition_audit=partition,
                executed_valid_prompt_work_and_chunk_audit=work,
                timing_run_state_scope='Full-state composition proof plus clean-run coverage. No full KV hash capture in timing runs.',
                natural_cap_hits=None if cfg.get('ignore_eos') else sum(r['cap_hit'] for r in rows),output_length_mean=float(np.mean([r['output_tokens'] for r in rows])),
                output_length_p95=float(np.percentile([r['output_tokens'] for r in rows],95)),
                output_length_max=max(r['output_tokens'] for r in rows),actual_runtime=runtime,
                evidence='MEASURED_ONLINE' if q['passed'] else 'DIAGNOSTIC_QUALITY_FAILURE',sustainable_capacity=False,**{k:m[k] for k in FIELDS})
            starts.append(value);metrics[label]=m
        groups[block['split'],block['qps']].append((block,metrics,statuses))
        complete_blocks.append(dict(split=block['split'],qps=block['qps'],start=block['start'],statuses=statuses))
    cells=[];comparisons=[]
    for (split,qps),blocks in sorted(groups.items()):
        labels=sorted({label for block,_,_ in blocks for label in block['arms']})
        for label in labels:
            matched=[(b,m) for b,m,_ in blocks if label in m]
            if not matched:continue
            metrics=[m[label] for _,m in matched]
            row=dict(split=split,QPS=qps,label=label,independent_starts=len(metrics),requests_per_start=metrics[0]['n'],
                output_contract=metrics[0]['output_contract'],
                quality_passed_all_starts=all(r['quality_passed'] for r in starts if r['split']==split and r['QPS']==qps and r['label']==label),
                evidence='MEASURED_ONLINE',fields={field:means_ci([m[field] for m in metrics]) for field in FIELDS})
            ref_blocks=[]
            for b,m in matched:
                common=b['NativeCommon_for'].get(label)
                if common and 'NativeStrong' in m and common in m:ref_blocks.append((b,m,common))
            if len(ref_blocks)==len(metrics) and ref_blocks:
                references={'NativeStrong':[m['NativeStrong'] for _,m,_ in ref_blocks],
                    'NativeCommon':[m[common] for _,m,common in ref_blocks]}
                row['decode_gate']=safety(metrics,references)
                row['common_reference_labels']=[common for _,_,common in ref_blocks]
            else:row['decode_gate']=dict(qualified=False,reason='MISSING_PAIRED_NATIVE_REFERENCES')
            if not row['quality_passed_all_starts']:row['evidence']='DIAGNOSTIC_QUALITY_FAILURE'
            cells.append(row)
        for left,right in [('Layer','StockToken'),('Hybrid','StockToken'),('Hybrid','Layer'),
                ('Layer','NativeStrong'),('StockToken','NativeStrong'),('Hybrid','NativeStrong'),('Deflection','NativeStrong')]:
            matched=[(b,m) for b,m,_ in blocks if left in m and right in m]
            if not matched:continue
            for b,m in matched:
                ca=json.loads((ROOT/'configs'/(b['arms'][left]+'.json')).read_text())
                cb=json.loads((ROOT/'configs'/(b['arms'][right]+'.json')).read_text())
                for field in ('dataset_SHA256','arrival_offsets','gpus','qps','seed','output_cap','ignore_eos'):
                    assert ca[field]==cb[field],(left,right,field)
            work=[abs(m[left]['output_tokens']-m[right]['output_tokens'])/min(m[left]['output_tokens'],m[right]['output_tokens']) for _,m in matched]
            comparisons.append(dict(split=split,QPS=qps,left=left,right=right,paired_starts=len(matched),
                output_work_unmatched=any(v>.05 for v in work),per_start_output_work_difference_fraction=work,
                ratios={field:ratio([m[left][field] for _,m in matched],[m[right][field] for _,m in matched]) for field in FIELDS}))
    csv_save(ROOT/'figures'/f'{path.stem}_per_start.csv',starts)
    flat=[]
    for cell in cells:
        row={k:v for k,v in cell.items() if k!='fields'}
        for field,value in cell['fields'].items():
            row[field]=value['mean'];row[field+'_CI95']=value['CI95']
        flat.append(row)
    csv_save(ROOT/'figures'/f'{path.stem}_summary.csv',flat)
    csv_save(ROOT/'figures'/f'{path.stem}_paired_comparisons.csv',comparisons)
    result=dict(time=time.time(),plan_SHA256=digest(path),cells=cells,comparisons=comparisons,execution_status=complete_blocks,
        scope='Client-visible continuous-serving metrics. Independent starts,not requests,define performance uncertainty. '
            'Intervals are paired,pointwise and unadjusted. N1 has no CI. Primary quality uses the frozen original compiled C1 reference. '
            'Both NativeStrong and exact common-D-policy Native are mandatory for a decode-safety claim. '
            'Decode failures remain visible. Natural EOS can change actual output work. Finite-cohort throughput is not sustainable capacity.')
    write(ROOT/'analysis'/f'{path.stem}_analysis.json',result);return result

def figures(result,prefix='c3_main_serving_v2',include_labels=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_labels=('Agg','NativeStrong','Layer','StockToken','Hybrid','Deflection')
    labels=all_labels if include_labels is None else include_labels
    colors=dict(zip(all_labels,('#dd6b00','#009e73','#0072b2','#cc79a7','#d55e00','#666666')))
    specs=[('TTFT_p95_ms','TTFT p95 (ms)'),('mean_TBT_ms','Mean client-visible TBT (ms)'),
        ('ITL_p99_ms','Client-visible ITL p99 (ms)'),('TPOT_p95_ms','Request TPOT p95 (ms/token)'),
        ('generated_output_tokens_per_second','Generated output tokens/s'),
        ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s'),
        ('TTFT_SLO_attainment','TTFT SLO attainment'),('decode_SLO_attainment','Decode SLO attainment'),
        ('Joint_SLO_attainment','Joint SLO attainment')]
    for split in sorted({r['split'] for r in result['cells']}):
        for field,ylabel in specs:
            fig,ax=plt.subplots(figsize=(8.2,5.3));plotted=[]
            for label in labels:
                values=sorted([r for r in result['cells'] if r['split']==split and r['label']==label],key=lambda r:r['QPS'])
                if not values:continue
                x=[r['QPS'] for r in values];y=[r['fields'][field]['mean'] if r['quality_passed_all_starts'] else np.nan for r in values]
                ax.plot(x,y,'o-',label=label,color=colors[label],linewidth=1.6,markersize=4)
                for r in values:
                    value=r['fields'][field];ci=value['CI95'];point=value['mean']
                    if not r['quality_passed_all_starts']:
                        ax.plot(r['QPS'],point,'s',color='#aaaaaa',fillstyle='none',markersize=6)
                    elif ci:ax.errorbar(r['QPS'],point,yerr=np.array([[point-ci[0]],[ci[1]-point]]),color=colors[label],capsize=3)
                    if label not in ('Agg','NativeStrong') and not r['decode_gate']['qualified']:
                        ax.plot(r['QPS'],point,'x',color='black',markersize=8)
                    plotted.append(dict(label=label,QPS=r['QPS'],value=point,CI95=ci,starts=value['starts'],requests_per_start=r['requests_per_start'],
                        output_contract=r['output_contract'],
                        decode_qualified=r['decode_gate']['qualified'],quality_passed=r['quality_passed_all_starts']))
            if field=='ITL_p99_ms':ax.axhline(34,color='#999999',linestyle=':',label='Absolute ITL limit34ms')
            if field=='TPOT_p95_ms':ax.axhline(33,color='#999999',linestyle=':',label='Absolute TPOT limit33ms')
            if field=='TTFT_p95_ms':ax.axhline(5000,color='#999999',linestyle=':',label='TTFT limit5000ms')
            ax.set(xlabel='Offered QPS (requests/s)',ylabel=ylabel,title=f'LongBench400-v2 | BI0 C3 | {split}')
            if field.endswith('SLO_attainment'):
                from matplotlib.ticker import PercentFormatter
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
            fig.text(.02,.015,'MEASURED_ONLINE. Marks are raw measured QPS points. Bars: start-level95%CI,only N≥3.\n'
                'Black ×: decode gate unqualified. Hollow gray squares: quality failure. Request count and output contract: CSV.\n'
                'Insufficient starts do not pass the decode gate. Finite-cohort throughput is not sustainable capacity.',fontsize=8)
            fig.tight_layout(rect=(0,.08,1,1))
            target=ROOT/'figures'/f'{prefix}_{split}_{field}'
            for ext in ('png','pdf'):fig.savefig(target.with_suffix('.'+ext),dpi=170)
            csv_save(target.with_suffix('.csv'),plotted);plt.close(fig)

def motivation(result):
    """Use the same measured Main trace,not a new or historical workload."""
    figures(result,'c3_motivation_agg_native_v2',include_labels=('Agg','NativeStrong'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=(('TTFT_p95_ms','TTFT p95 (ms)'),('ITL_p99_ms','Client-visible ITL p99 (ms)'),
        ('TPOT_p95_ms','Request TPOT p95 (ms/token)'),('generated_output_tokens_per_second','Generated output tokens/s'))
    for split in sorted({r['split'] for r in result['cells']}):
        fig,axes=plt.subplots(2,2,figsize=(12,8));records=[]
        for ax,(field,ylabel) in zip(axes.flat,specs):
            for label,color in (('Agg','#dd6b00'),('NativeStrong','#009e73')):
                rows=sorted([r for r in result['cells'] if r['split']==split and r['label']==label],key=lambda r:r['QPS'])
                ax.plot([r['QPS'] for r in rows],
                    [r['fields'][field]['mean'] if r['quality_passed_all_starts'] else np.nan for r in rows],
                    'o-',label=label,color=color,markersize=4)
                for row in rows:
                    value=row['fields'][field];point=value['mean'];ci=value['CI95']
                    if ci and row['quality_passed_all_starts']:
                        ax.errorbar(row['QPS'],point,yerr=[[point-ci[0]],[ci[1]-point]],color=color,capsize=3)
                    records.append(dict(split=split,metric=field,label=label,QPS=row['QPS'],value=point,CI95=ci,
                        starts=value['starts'],requests_per_start=row['requests_per_start'],quality_passed=row['quality_passed_all_starts']))
            ax.set(xlabel='Offered QPS (requests/s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle(f'Qwen3-8B | 2 × H100 | LongBench400-v2 | BI0 C3 | {split}')
        fig.text(.02,.015,'New variable-length benchmark using DuetServe-style comparisons. This is not a reproduction of its Figure2 values.\n'
            'Raw QPS points only. Bars: start-level95%CI when N≥3. Natural EOS output totals remain in the tables.\n'
            'Output-only finite-cohort throughput is not sustainable capacity.',fontsize=8)
        fig.tight_layout(rect=(0,.09,1,.94));stem=f'c3_motivation_agg_native_four_panel_{split}_v2'
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
        csv_save(ROOT/'figures'/f'{stem}.csv',records);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='c3_main_serving_v2.json');args=p.parse_args()
    result=analyze(args.plan);figures(result,args.plan.removesuffix('.json'))
    if args.plan=='c3_main_serving_v2.json':motivation(result)
    print('C3_TABLES_AND_FIGURES',len(result['cells']),flush=True)
