"""Keep C0 point estimates, uncertainty and decode safety in separate outputs."""
import collections
import csv
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .statistics import ratio,point_winner,confidence,ratio_of_ratios

FIELDS=('TTFT_p50_ms','TTFT_p95_ms','TTFT_p99_ms','TTFT_mean_ms','E2E_p95_ms','mean_TBT_ms','ITL_p99_ms',
    'max_ITL_ms','TPOT_p95_ms','generated_output_tokens_per_second','completed_requests_per_second',
    'SLO_qualified_output_tokens_per_second','TTFT_SLO_attainment','every_ITL_SLO_attainment',
    'TPOT_SLO_attainment','decode_SLO_attainment','Joint_SLO_attainment','actual_arrival_QPS',
    'measurement_seconds','output_tokens')
INPUTS=('3K–6K','6K–10K','10K–20K','20K–36K')
OUTPUTS=(8,16,64,128,192)
CUTS=(2,4,8,12)

def csv_save(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def load_metric(name):
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if not receipt.exists():return None,'NOT_RUN'
    if not json.loads(receipt.read_text())['complete']:return None,'FAILED'
    path=ROOT/'analysis'/(name+'_metrics.json')
    if not path.exists():return None,'AWAITING_METRIC_RECOMPUTATION'
    return json.loads(path.read_text()),'COMPLETE'

def c0_safety(arm,native):
    tests={field:ratio([r[field] for r in arm],[r[field] for r in native]) for field in ('ITL_p99_ms','TPOT_p95_ms')}
    relative=len(arm)>=3 and all(t['CI95'] is not None and t['CI95'][1]<=1.05 for t in tests.values())
    absolute=all(r['decode_SLO_attainment']>=.95 for r in arm)
    return dict(tests=tests,relative_5percent_pass=relative,absolute_decode_attainment_95percent_pass=absolute,
        controlled_decode_gate_pass=bool(relative and absolute),
        scope='C0 matched-geometry Native reference. This is not the C3 NativeStrong plus common-policy Native gate.',
        per_start_all_ITL_request_attainment=[r['every_ITL_SLO_attainment'] for r in arm],
        per_start_max_ITL_ms=[r['max_ITL_ms'] for r in arm])

def check_block(block,metrics):
    from .bridge import bind
    bind()
    from ..dense_ar_longbench400_pd.qualification import audit as coverage_audit
    configs={arm:json.loads((ROOT/'configs'/(block[arm]+'.json')).read_text()) for arm in ('Native','Layer','Token')}
    for field in ('dataset_SHA256','arrival_offsets','gpus','qps','seed','output_cap','ignore_eos'):
        assert len({json.dumps(c[field],sort_keys=True) for c in configs.values()})==1,(block,field)
    assert configs['Token']['aggregate_work_audit']['exact_match']
    data=json.loads((ROOT/'data'/block['dataset']).read_text());expected=sum(r['input_tokens'] for r in data)*block['cut']
    assert sum(configs['Token']['suffix_by_request'].values())*36==expected
    for arm,m in metrics.items():
        assert m['output_tokens']==len(data)*block['output']
        assert m['B']==8192 and m['S']==1024 and m['Source_credit']==4
        assert m['runtime_source_archive_SHA256']==configs[arm]['source_capsule_SHA256']
        for role,shape in m['actual_geometry'].items():
            assert shape['maximum_requests']==1,(block,arm,role,shape)
        if arm in ('Layer','Token'):
            path=ROOT/'analysis'/(block[arm]+'_coverage.json')
            coverage=json.loads(path.read_text()) if path.exists() else coverage_audit(block[arm])
            assert coverage['passed'] and coverage['decode_full36_observation_complete']
    return dict(exact_D_GEMM_rows=expected,exact_P_GEMM_rows=sum(r['input_tokens'] for r in data)*36-expected,
        dataset_SHA256=configs['Token']['dataset_SHA256'],actual_batch='1/1/1',GPU_pair=block['pair'],
        runtime_source_archive_SHA256=configs['Token']['source_capsule_SHA256'],aggregate_rows_exact=True,
        runtime_query_layer_coverage_passed=True,
        timing_run_import_parity='Inherited full-state qualification. Full bytes are not hashed in clean timing runs.')

def structural(block):
    from ..dense_ar_longbench400_pd.geometry import Partition
    data=json.loads((ROOT/'data'/block['dataset']).read_text())
    cfg=json.loads((ROOT/'configs'/(block['Token']+'.json')).read_text())
    result={}
    for axis in ('Layer','Token'):
        partitions=[Partition(r['input_tokens'],block['cut'] if axis=='Layer' else 0,
            0 if axis=='Layer' else cfg['suffix_by_request'][f'q{i}']).record() for i,r in enumerate(data)]
        result[axis]={k:sum(r[k] for r in partitions) for k in
            ('P_rows','D_rows','P_attention_pairs','D_attention_pairs','valid_KV_bytes','hidden_residual_bytes')}
    assert result['Layer']['D_rows']==result['Token']['D_rows']
    return result

def analyze():
    plan=json.loads((ROOT/'jobs/c0_output_matrix_v2.json').read_text());grouped=collections.defaultdict(list)
    starts=[];progress=collections.Counter()
    for block in plan['blocks']:
        data={arm:load_metric(block[arm]) for arm in ('Native','Layer','Token')}
        state='FAILED' if any(s=='FAILED' for _,s in data.values()) else 'COMPLETE' if all(m for m,_ in data.values()) else 'NOT_RUN'
        progress[state]+=1
        if state!='COMPLETE':
            grouped[block['split'],block['input_bucket'],block['output'],block['cut']].append((block,None,state));continue
        values={arm:m for arm,(m,_) in data.items()};geometry=check_block(block,values)
        grouped[block['split'],block['input_bucket'],block['output'],block['cut']].append((block,values,state))
        for arm,m in values.items():
            starts.append(dict(split=block['split'],input_bucket=block['input_bucket'],output=block['output'],cut=block['cut'],
                start=block['start'],seed=block['seed'],name=block[arm],arm=arm,offered_QPS=block['qps'],
                evidence='MEASURED_ONLINE',**geometry,**{k:m[k] for k in FIELDS},actual_geometry=m['actual_geometry']))
    cells=[]
    for key,blocks in sorted(grouped.items()):
        split,bucket,output,cut=key;complete=[x for x in blocks if x[1] is not None]
        row=dict(split=split,input_bucket=bucket,input_label=INPUTS[bucket],output=output,cut=cut,
            requested_starts=len(blocks),completed_paired_starts=len(complete),offered_QPS=plan['QPS_by_input'][str(bucket)],
            evidence='MEASURED_ONLINE' if complete else 'NOT_RUN',point_winner='NOT_RUN',statistically_confirmed='NOT_RUN',
            status='FAILED' if any(x[2]=='FAILED' for x in blocks) else 'COMPLETE' if len(complete)==len(blocks) else 'PARTIAL_OR_NOT_RUN')
        if complete:
            arms={arm:[x[1][arm] for x in complete] for arm in ('Native','Layer','Token')}
            test=ratio([m['TTFT_p95_ms'] for m in arms['Layer']],[m['TTFT_p95_ms'] for m in arms['Token']])
            row.update(TTFT_ratio=test['estimate'],TTFT_CI95=test['CI95'],point_winner=point_winner(test),statistically_confirmed=confidence(test),
                Layer_decode=c0_safety(arms['Layer'],arms['Native']),Token_decode=c0_safety(arms['Token'],arms['Native']))
            row['structural_accounting']=structural(complete[0][0])
            row['both_arms_controlled_decode_gate_pass']=all(row[k]['controlled_decode_gate_pass'] for k in ('Layer_decode','Token_decode'))
            for field in FIELDS:
                for arm in arms:row[f'{arm}_{field}']=float(np.mean([m[field] for m in arms[arm]]))
                row[f'Layer_over_Token_{field}']=ratio([m[field] for m in arms['Layer']],[m[field] for m in arms['Token']])
            for arm in ('Layer','Token'):
                row[f'{arm}_over_C0Native_TTFT_p95']=ratio([m['TTFT_p95_ms'] for m in arms[arm]],[m['TTFT_p95_ms'] for m in arms['Native']])
        cells.append(row)
    csv_save(ROOT/'figures/c0_output_map_per_start_v2.csv',starts)
    csv_save(ROOT/'figures/c0_output_map_cells_v2.csv',cells)
    result=dict(time=time.time(),plan_SHA256=digest(ROOT/'jobs/c0_output_matrix_v2.json'),cells=cells,progress=dict(progress),
        scope='C0 controlled output mechanism measurements. The Mode0 Token arm is not the strong compiled C3 Stock Token baseline. '
            'Point estimates use the ratio of start means. Confidence intervals use paired server starts,not requests. '
            'Intervals are pointwise and unadjusted. Discovery N1 has no CI. Safety does not change the point-winner color. '
            'The controlled Native gate does not replace the two-reference C3 gate. Finite-cohort throughput is not sustainable capacity.')
    write(ROOT/'analysis/c0_output_map_v2.json',result);return result

def output_effect(result):
    plan=json.loads((ROOT/'jobs/c0_output_matrix_v2.json').read_text());rows=[]
    for bucket in range(4):
        for cut in CUTS:
            blocks={(b['output'],b['start']):b for b in plan['blocks']
                if b['split']=='heldout' and b['input_bucket']==bucket and b['cut']==cut}
            endpoints={};complete=True
            for output in (8,192):
                for start in range(3):
                    b=blocks[output,start];values={arm:load_metric(b[arm])[0] for arm in ('Layer','Token')}
                    if not all(values.values()):complete=False;continue
                    endpoints[output,start]=values
            row=dict(input_bucket=bucket,cut=cut,contrast='Output192 versus8',evidence='NOT_RUN')
            if complete:
                a=np.array([endpoints[192,i]['Layer']['TTFT_p95_ms'] for i in range(3)])
                b=np.array([endpoints[192,i]['Token']['TTFT_p95_ms'] for i in range(3)])
                c=np.array([endpoints[8,i]['Layer']['TTFT_p95_ms'] for i in range(3)])
                d=np.array([endpoints[8,i]['Token']['TTFT_p95_ms'] for i in range(3)])
                # A paired output intervention preserves the same prompt IDs and
                # arrival seed at each endpoint. Its effect is not a natural-EOS predictor.
                row.update(evidence='MEASURED_ONLINE',ratio_of_axis_ratios=ratio_of_ratios(a,b,c,d),
                    per_start_Layer_over_Token_O8=(c/d).tolist(),per_start_Layer_over_Token_O192=(a/b).tolist())
            rows.append(row)
    counts={}
    for output in OUTPUTS:
        cells=[r for r in result['cells'] if r['split']=='heldout' and r['output']==output and r['completed_paired_starts']==3]
        counts[output]=dict(completed_cells=len(cells),point_winners=dict(collections.Counter(r['point_winner'] for r in cells)),
            statistical_confidence=dict(collections.Counter(r['statistically_confirmed'] for r in cells)),
            both_arms_decode_qualified=sum(r['both_arms_controlled_decode_gate_pass'] for r in cells))
    write(ROOT/'analysis/c0_controlled_output_effect_v2.json',dict(time=time.time(),rows=rows,counts=counts,
        scope='Paired endpoint intervention on unchanged prompts,Input,Cut,QPS and arrival seeds. '
            'Output control also changes D residency and capacity-relative load. The total effect does not identify one mediator. '
            'Axis-ratio endpoint contrast is descriptive. Confidence intervals are pointwise,not multiplicity-adjusted. '
            'No future natural EOS is available to an online policy.'))
    csv_save(ROOT/'figures/c0_controlled_output_effect_v2.csv',rows)

def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,TwoSlopeNorm
    from matplotlib.patches import Rectangle
    for split in ('discovery','heldout'):
        rows={(r['cut'],r['input_bucket'],r['output']):r for r in result['cells'] if r['split']==split}
        for kind in ('point','confidence','safety'):
            fig,axes=plt.subplots(1,4,figsize=(18,5.8),sharey=True)
            for ax,cut in zip(axes,CUTS):
                values=np.full((4,5),np.nan)
                for i in range(4):
                    for j,o in enumerate(OUTPUTS):
                        row=rows[cut,i,o]
                        if 'TTFT_ratio' not in row:continue
                        values[i,j]=row['TTFT_ratio'] if kind=='point' else (
                            -1 if row['statistically_confirmed']=='LAYER_STATISTICALLY_CONFIRMED' else
                            1 if row['statistically_confirmed']=='TOKEN_STATISTICALLY_CONFIRMED' else 0) if kind=='confidence' else (
                            1 if row['both_arms_controlled_decode_gate_pass'] else 0)
                if kind=='point':ax.imshow(values,cmap='coolwarm',norm=TwoSlopeNorm(vmin=.7,vcenter=1.,vmax=1.3),aspect='auto')
                elif kind=='confidence':ax.imshow(values,cmap=ListedColormap(['#86bdd9','#eeeeee','#f0a688']),vmin=-1,vmax=1,aspect='auto')
                else:ax.imshow(values,cmap=ListedColormap(['#f0a688','#98ceaa']),vmin=0,vmax=1,aspect='auto')
                for i in range(4):
                    for j,o in enumerate(OUTPUTS):
                        r=rows[cut,i,o]
                        if 'TTFT_ratio' not in r:label=r['status'] if r['status']=='FAILED' else 'NOT_RUN'
                        elif kind=='point':label=f"{'L' if r['TTFT_ratio']<1 else 'T' if r['TTFT_ratio']>1 else '='}:{r['TTFT_ratio']:.4f}\nN={r['completed_paired_starts']}"
                        elif kind=='confidence':label='L confirmed' if values[i,j]<0 else 'T confirmed' if values[i,j]>0 else 'No CI' if r['TTFT_CI95'] is None else 'CI crosses1'
                        else:label='PASS' if values[i,j]==1 else 'FAIL / unqualified'
                        ax.text(j,i,label,ha='center',va='center',fontsize=8)
                        if kind=='point' and 'TTFT_ratio' in r and not r['both_arms_controlled_decode_gate_pass']:
                            ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='//',edgecolor='#666666',linewidth=.4))
                ax.set(title=f'Last{cut}',xticks=range(5),xticklabels=OUTPUTS,xlabel='Controlled output tokens',yticks=range(4),yticklabels=INPUTS)
            fig.suptitle(f'LongBench400-v2 | C0 actual batch1/1/1 | {split} | {kind}')
            fig.text(.02,.02,'MEASURED_ONLINE. L/T TTFT p95 ratio uses start means. Point color ignores CI. Hatching: controlled decode gate unqualified.\n'
                'N1 discovery has no CI. Heldout targetN3. C0 Mode0 Token is not strong C3 Stock Token. Full ratios,CI and scope: paired CSV.',fontsize=9)
            fig.tight_layout(rect=(0,.10,1,.94))
            for extension in ('png','pdf'):fig.savefig(ROOT/'figures'/f'c0_{split}_{kind}_map_v2.{extension}',dpi=170)
            plt.close(fig)

if __name__=='__main__':
    result=analyze();figures(result);output_effect(result);print(result['progress'],flush=True)
