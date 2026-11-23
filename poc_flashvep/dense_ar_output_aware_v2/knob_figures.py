"""Plot the completed fixed-output interventions without relabeling natural serving."""
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .statistics import ratio,safety,paired_difference
from .c3_analysis import means_ci
from .c0_analysis import csv_save


def main():
    plan=json.loads((ROOT/'jobs/c3_knob_independent_confirmation_v2.json').read_text())
    original=json.loads((ROOT/'analysis/c3_knob_independent_confirmation_v2.json').read_text())
    values={b['start']:{label:json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
            for label,name in b['arms'].items()} for b in plan['blocks']}
    labels=list(plan['blocks'][0]['arms']);absolute=[]
    fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second',
            'generated_output_tokens_per_second','max_ITL_ms','every_ITL_SLO_attainment')
    for label in labels:
        for field in fields:
            tests=means_ci([values[k][label][field] for k in range(3)])
            absolute.append(dict(arm=label,metric=field,start_mean=tests['mean'],CI95=tests['CI95'],independent_starts=3,
                evidence='DIAGNOSTIC_CONTROLLED_REPLAY'))
    contrasts=[dict(left=r['left'],right=r['right'],TTFT_ratio=r['ratios']['TTFT_p95_ms'],
        left_decode_passed=r['decode_gates'][r['left']]['qualified'],right_decode_passed=r['decode_gates'][r['right']]['qualified'],
        prespecified=True,evidence='DIAGNOSTIC_CONTROLLED_REPLAY') for r in original['rows']]
    if 'DFused' in labels and 'DEager' in labels:
        aa=[values[k]['DFused'] for k in range(3)];bb=[values[k]['DEager'] for k in range(3)]
        assert all(a['output_tokens']==b['output_tokens'] for a,b in zip(aa,bb))
        refs={key:[values[k][label] for k in range(3)] for key,label in
            (('NativeStrong','NativeStrong'),('NativeCommon','NativeCommon2'))}
        result=dict(left='DFused',right='DEager',paired_starts=3,posthoc_compound_comparison=True,
            TTFT_ratio=ratio([m['TTFT_p95_ms'] for m in aa],[m['TTFT_p95_ms'] for m in bb]),
            goodput_ratio=ratio([m['SLO_qualified_output_tokens_per_second'] for m in aa],[m['SLO_qualified_output_tokens_per_second'] for m in bb]),
            left_decode=safety(aa,refs),right_decode=safety(bb,refs),evidence='DIAGNOSTIC_CONTROLLED_REPLAY',
            scope='Posthoc compound contrast on the same three existing starts. Change both D compilation and short-hybrid fusion. '
                'Compute its direct paired interval. Do not multiply CI endpoints or count these starts again. '
                'This is not a natural-EOS serving result or a one-flag causal attribution.')
        write(ROOT/'analysis/c3_knob_posthoc_compound_contrast_v2.json',dict(time=time.time(),**result))
        contrasts.append(dict(left='DFused',right='DEager',TTFT_ratio=result['TTFT_ratio'],
            left_decode_passed=result['left_decode']['qualified'],right_decode_passed=result['right_decode']['qualified'],
            prespecified=False,evidence='DIAGNOSTIC_CONTROLLED_REPLAY'))
    write(ROOT/'analysis/c3_knob_absolute_and_contrast_tables_v2.json',dict(time=time.time(),absolute=absolute,contrasts=contrasts,
        source_plan_SHA256=digest(ROOT/'jobs/c3_knob_independent_confirmation_v2.json'),
        scope='Same original192 requests and exact per-request C1 output lengths. Three independent starts per fixed intervention. '
            'All intervals are start-level,pointwise and unadjusted. These are fixed-length replay diagnostics,not natural serving gains.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5.5))
    for index,row in enumerate(contrasts):
        value=row['TTFT_ratio'];point=value['estimate'];ci=value['CI95']
        ax.errorbar(point,index,xerr=[[point-ci[0]],[ci[1]-point]],fmt='o',capsize=4,
            color='#0072b2' if row['prespecified'] else '#999999')
    ax.axvline(1,color='#777777',linestyle=':')
    ax.set(yticks=range(len(contrasts)),yticklabels=[r['left']+'/'+r['right']+(' *' if not r['prespecified'] else '') for r in contrasts],
        xlabel='TTFT p95 ratio of start means',title='BI0 C3 | equal-output replay | paired N3')
    ax.grid(axis='x',alpha=.2)
    fig.text(.02,.015,'DIAGNOSTIC_CONTROLLED_REPLAY. Bars: paired start-level95%CI. All comparisons use the same existing three starts.\n'
        '* Posthoc two-flag bundle. A ratio below1 is not a natural-EOS speedup or proof of superiority over Stock Token.',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));stem='c3_knob_TTFT_ratio_controls_v2'
    for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
    csv_save(ROOT/'figures'/f'{stem}.csv',contrasts);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for ax,field,ylabel in zip(axes,('TTFT_p95_ms','SLO_qualified_output_tokens_per_second'),('TTFT p95 (ms)','SLO-qualified output tokens/s')):
        rows=[next(r for r in absolute if r['arm']==label and r['metric']==field) for label in labels]
        y=[r['start_mean'] for r in rows]
        ax.bar(range(len(labels)),y,color='#4477aa')
        ax.errorbar(range(len(labels)),y,yerr=np.array([[v-r['CI95'][0] for r,v in zip(rows,y)],
            [r['CI95'][1]-v for r,v in zip(rows,y)]]),fmt='none',color='black',capsize=3)
        ax.set(ylabel=ylabel);ax.grid(axis='y',alpha=.2)
    axes[-1].set(xticks=range(len(labels)),xticklabels=labels)
    axes[-1].tick_params(axis='x',labelrotation=35)
    fig.suptitle('Fixed-length C1 replay | absolute denominators | 192 original requests/start | N3')
    fig.text(.02,.015,'DIAGNOSTIC_CONTROLLED_REPLAY. Future reference EOS is experimental output control,not an online feature.\n'
        'Bars: start means. Error bars: start-level95%CI. Throughput is finite-cohort,not sustainable capacity.',fontsize=8)
    fig.tight_layout(rect=(0,.1,1,.95));stem='c3_knob_absolute_TTFT_goodput_v2'
    for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
    csv_save(ROOT/'figures'/f'{stem}.csv',absolute);plt.close(fig)
    from .c0_output_path import GROUPS
    host_rows=[];fig,ax=plt.subplots(figsize=(12,6));positive=np.zeros(len(original['rows']));negative=positive.copy()
    for index,(group,fields) in enumerate(GROUPS):
        points=[]
        for row in original['rows']:
            values=[sum(start[field] for field in fields) for start in row['per_start_additive_mean_host_delta_ms']]
            value=paired_difference(values,[0.]*3);points.append(value['estimate'])
            host_rows.append(dict(left=row['left'],right=row['right'],host_group=group,
                mean_delta_ms=value['estimate'],CI95=value['CI95'],independent_starts=3,evidence='TRACE_ANALYSIS',
                output_contract='DIAGNOSTIC_FIXED_LENGTH_REPLAY'))
        y=np.asarray(points);base=np.where(y>=0,positive,negative)
        ax.bar(range(len(points)),y,bottom=base,label=group)
        positive+=np.maximum(y,0);negative+=np.minimum(y,0)
    for row in original['rows']:
        parts=[r['mean_delta_ms'] for r in host_rows if r['left']==row['left'] and r['right']==row['right']]
        expected=np.mean([v['TTFT_ms'] for v in row['per_start_additive_mean_host_delta_ms']])
        assert abs(sum(parts)-expected)<1e-3
    ax.axhline(0,color='black',linewidth=.7)
    ax.set(xticks=range(len(original['rows'])),xticklabels=[r['left']+'/'+r['right'] for r in original['rows']],
        ylabel='Changed minus control mean host-stage latency (ms)',title='Fixed-output runtime interventions | paired N3')
    ax.legend(fontsize=7,ncol=2);ax.grid(axis='y',alpha=.2)
    fig.text(.02,.015,'TRACE_ANALYSIS. Stages sum to mean client TTFT,not TTFT p95. Prepare is a host boundary,not a GPU kernel start.\n'
        'The knob intervention identifies a total runtime effect. Its overlapping queue,batch and service mediators are not separately identified.',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,1));stem='c3_knob_additive_host_path_v2'
    for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{stem}.{ext}',dpi=170)
    csv_save(ROOT/'figures'/f'{stem}.csv',host_rows);plt.close(fig)
    print('KNOB_REPLAY_ABSOLUTE_AND_RATIO_FIGURES',flush=True)


if __name__=='__main__':main()
