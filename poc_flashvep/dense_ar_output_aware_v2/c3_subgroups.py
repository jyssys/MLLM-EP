"""Describe frozen input/output strata inside the same mixed C3 trace."""
import argparse
import collections
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .statistics import ratio,confidence,safety
from .c0_analysis import csv_save,INPUTS

def subgroup(rows):
    from ..dense_ar_bi0_shortanswer_pd.metrics import delivery_intervals
    itls=[v for r in rows for v in delivery_intervals(r['token_events'],r['output_tokens'])]
    tpots=[r['TPOT_ms'] for r in rows if r['TPOT_ms'] is not None]
    if not itls or not tpots:return None
    safe=[];all_itl=[]
    for r in rows:
        intervals=delivery_intervals(r['token_events'],r['output_tokens']);ok=all(v<=34 for v in intervals)
        all_itl.append(ok);safe.append(ok and (r['TPOT_ms'] is None or r['TPOT_ms']<=33))
    return dict(TTFT_p95_ms=float(np.percentile([r['TTFT_ms'] for r in rows],95)),
        ITL_p99_ms=float(np.percentile(itls,99)),TPOT_p95_ms=float(np.percentile(tpots,95)),
        decode_SLO_attainment=float(np.mean(safe)),every_ITL_SLO_attainment=float(np.mean(all_itl)),max_ITL_ms=max(itls),
        output_tokens=sum(r['output_tokens'] for r in rows),requests=len(rows))

def analyze():
    plan=json.loads((ROOT/'jobs/c3_main_serving_v2.json').read_text());data=json.loads((ROOT/'data/longbench_balanced400_v2.json').read_text())
    keys={(i,j):{r['prompt_id'] for r in data if r['input_bucket']==i and r['natural_output_bucket']==j} for i in range(4) for j in range(4)}
    groups=collections.defaultdict(list)
    for block in plan['blocks']:
        allrows={};ready=True
        for label,name in block['arms'].items():
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists() or not json.loads(receipt.read_text())['complete']:ready=False;break
            allrows[label]=list(map(json.loads,(ROOT/'raw'/name/'requests.jsonl').read_text().splitlines()))
        if not ready:continue
        for key,ids in keys.items():
            metrics={label:subgroup([r for r in rows if r['prompt_id'] in ids]) for label,rows in allrows.items()}
            if not all(metrics.values()):continue
            groups[block['qps'],*key].append((block,metrics))
    cells=[]
    for (qps,i,j),blocks in sorted(groups.items()):
        arms={label:[m[label] for _,m in blocks] for label in ('Layer','StockToken','Hybrid')}
        means={label:float(np.mean([m['TTFT_p95_ms'] for m in values])) for label,values in arms.items()}
        point=min(means,key=lambda name:(means[name],name));tests={};gates={}
        for label,values in arms.items():
            tests[label]={other:ratio([m['TTFT_p95_ms'] for m in values],[m['TTFT_p95_ms'] for m in arms[other]]) for other in arms if other!=label}
            refs={'NativeStrong':[m['NativeStrong'] for _,m in blocks],
                'NativeCommon':[m[b['NativeCommon_for'][label]] for b,m in blocks]}
            gates[label]=safety(values,refs)
        confirmed=[label for label in arms if all(t['CI95'] and t['CI95'][1]<1 for t in tests[label].values())]
        work={label:[m['output_tokens'] for m in values] for label,values in arms.items()}
        cells.append(dict(QPS=qps,input_bucket=i,selection_reference_output_bin=j,requests_per_start=len(keys[i,j]),
            paired_starts=len(blocks),point_winner=point,statistically_confirmed_winner=confirmed[0] if len(confirmed)==1 else 'TIE_OR_INCONCLUSIVE',
            start_mean_TTFT_p95_ms=means,TTFT_ratios=tests,decode_gates=gates,per_start_actual_output_tokens=work,
            actual_output_work_unmatched=any(max(work[label][k] for label in arms)/min(work[label][k] for label in arms)>1.05 for k in range(len(blocks))),
            evidence='MEASURED_ONLINE_SUBGROUP',
            scope='Conditional subgroup inside Mixed400. Output bin uses frozen Mode0 C1 selection length,not the method realized EOS. '
                'This is not an isolated output intervention or an iso-work partition comparison. Global configuration quality gates remain separate.'))
    result=dict(time=time.time(),plan_SHA256=digest(ROOT/'jobs/c3_main_serving_v2.json'),cells=cells,
        scope='Use start means for point winners. Keep paired pointwise95%CI and two-reference decode gates separate. '
            'Natural EOS work can differ. The actual output totals remain visible. Do not infer a causal output effect from this mixed-trace stratification.')
    write(ROOT/'analysis/c3_frozen_stratum_point_and_confidence_v2.json',result)
    csv_save(ROOT/'figures/c3_frozen_stratum_point_and_confidence_v2.csv',cells);figures(result)
    print('C3_FROZEN_STRATUM_TABLE',len(cells),flush=True)

def figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    palette=ListedColormap(['#7ab1d0','#e7a36f','#b7a0d4','#eeeeee'])
    for qps in sorted({r['QPS'] for r in result['cells']}):
        rows={(r['input_bucket'],r['selection_reference_output_bin']):r for r in result['cells'] if r['QPS']==qps}
        fig,axes=plt.subplots(1,3,figsize=(14,5));labels=('Layer','StockToken','Hybrid')
        for ax,kind in zip(axes,('Point winner','Confidence winner','Point-winner decode gate')):
            a=np.full((4,4),3.)
            for key,row in rows.items():
                winner=row['point_winner'] if kind!='Confidence winner' else row['statistically_confirmed_winner']
                a[key]=labels.index(winner) if winner in labels else 3
                if kind=='Point-winner decode gate':a[key]=0 if row['decode_gates'][row['point_winner']]['qualified'] else 1
            ax.imshow(a,cmap=palette,aspect='auto',vmin=0,vmax=3)
            for i in range(4):
                for j in range(4):
                    row=rows.get((i,j));text='NOT_RUN'
                    if row:
                        text=row['point_winner'] if kind=='Point winner' else row['statistically_confirmed_winner'] if kind=='Confidence winner' else 'PASS' if a[i,j]==0 else 'Unqualified'
                        text+=f'\nN={row["paired_starts"]},n={row["requests_per_start"]}'
                        if row['actual_output_work_unmatched']:ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch='//',edgecolor='#777777',linewidth=.4))
                    ax.text(j,i,text,ha='center',va='center',fontsize=7)
            ax.set(title=kind,xticks=range(4),xticklabels=('1–10','11–50','51–100','101–200'),yticks=range(4),yticklabels=INPUTS,
                xlabel='Frozen selection-reference output bin')
        fig.suptitle(f'C3 mixed400 subgroups | QPS{qps} | best-tuned fixed configurations')
        fig.text(.02,.015,'MEASURED_ONLINE_SUBGROUP. Conditional strata,not an output intervention. Hatching: actual output work differs>5%.\n'
            'Point colors use start means. Confidence is pointwise,unadjusted. Decode safety remains a separate two-Native gate. See CSV for every ratio/CI.',fontsize=8)
        fig.tight_layout(rect=(0,.1,1,.93))
        for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'c3_frozen_stratum_winner_q{qps}_v2.{ext}',dpi=170)
        plt.close(fig)

if __name__=='__main__':analyze()
