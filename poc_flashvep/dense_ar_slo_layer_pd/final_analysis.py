"""Summarize frozen independent starts and draw evidence-labeled figures."""
import itertools,statistics,csv,collections
from .common import *
from .analyze import records,percentile
from .statistics import paired_ratio

METRICS=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','raw_completed_output_tokens_s',
    'completed_output_requests_s','maximum_ITL_ms')


def mean_CI(values):
    n=len(values)
    if not values:return dict(samples=0,mean=None,CI95=None)
    draws=[sum(values[i] for i in indices)/n for indices in itertools.product(range(n),repeat=n)]
    return dict(samples=n,mean=statistics.mean(values),CI95=[percentile(draws,.025),percentile(draws,.975)])


def load():
    values=[]
    for path in sorted((ROOT/'analysis').glob('heldout_*_sustainable_v2.json')):
        value=json.loads(path.read_text());cfg=json.loads((ROOT/'configs'/(value['name']+'.json')).read_text())
        value['method']=cfg['selected_method_label'];value['seed']=cfg['arrival_seed'];value['rate']=cfg['arrival_rate']
        value['configuration']=cfg;values.append(value)
    return values


def statistics_report(values):
    table={};capacity=[v for v in values if v['name'].startswith('heldout_capacity_')]
    for method in ('Native','Colocated','Static','Dynamic'):
        selected=sorted([v for v in capacity if v['method']==method],key=lambda v:v['seed'])
        result=dict(names=[v['name'] for v in selected],independent_starts=len(selected),
            seeds=[v['seed'] for v in selected],rates=[v['rate'] for v in selected],
            metrics={key:mean_CI([v[key] for v in selected]) for key in METRICS},
            maximum_observed_ITL_ms=max((v['maximum_ITL_ms'] for v in selected),default=None),SLO_regimes={})
        for regime in ('TTFT_priority','Balanced','Decode_priority'):
            passes=[v['SLO_regimes'][regime]['qualified'] for v in selected]
            result['SLO_regimes'][regime]=dict(all_three_qualified=len(passes)==3 and all(passes),
                passed_starts=sum(passes),starts=len(passes),
                all_three_qualified_99pct_sensitivity=len(passes)==3 and all(
                    v['SLO_regimes'][regime]['qualified'] and
                    v['SLO_regimes'][regime]['arrival_cohort_joint_attainment']>=.99 and
                    v['SLO_regimes'][regime]['completion_window_joint_attainment']>=.99 for v in selected),
                qualified_tokens_s=mean_CI([v['SLO_regimes'][regime]['SLO_qualified_output_tokens_s'] for v in selected]),
                arrival_joint=mean_CI([v['SLO_regimes'][regime]['arrival_cohort_joint_attainment'] for v in selected]),
                completion_joint=mean_CI([v['SLO_regimes'][regime]['completion_window_joint_attainment'] for v in selected]))
        table[method]=result
    comparisons={}
    for baseline in ('Native','Colocated','Static'):
        a={v['seed']:v for v in capacity if v['method']=='Dynamic'}
        b={v['seed']:v for v in capacity if v['method']==baseline};seeds=sorted(set(a)&set(b))
        if len(seeds)<2:continue
        comparisons[baseline]={regime:paired_ratio(
            [a[s]['SLO_regimes'][regime]['SLO_qualified_output_tokens_s'] for s in seeds],
            [b[s]['SLO_regimes'][regime]['SLO_qualified_output_tokens_s'] for s in seeds])
            for regime in ('TTFT_priority','Balanced','Decode_priority')
            if sum(b[s]['SLO_regimes'][regime]['SLO_qualified_output_tokens_s'] for s in seeds)>0}
    dynamic={v['seed']:v for v in capacity if v['method']=='Dynamic'};native={}
    for value in values:
        if value['method']=='Native' and value['seed'] in dynamic and value['rate']==dynamic[value['seed']]['rate']:
            native[value['seed']]=value
    seeds=sorted(set(native)&set(dynamic));relative={}
    if len(seeds)>=2:
        for key in ('ITL_p99_ms','TPOT_p95_ms','TTFT_p95_ms'):
            relative[key]=paired_ratio([dynamic[s][key] for s in seeds],[native[s][key] for s in seeds])
        relative['Decode_5pct_noninferiority_passed']=all(
            relative[key]['paired_bootstrap_95_percentile_CI'][1]<=1.05 for key in ('ITL_p99_ms','TPOT_p95_ms'))
        relative['paired_seeds']=seeds
        relative['Native_scope']='Same arrival QPS and seeds as the selected Dynamic point. A failed Native capacity screen is retained, not called sustainable.'
    result=dict(time=time.time(),methods=table,Dynamic_capacity_point_goodput_ratios=comparisons,
        same_rate_Native_relative=relative,evidence='MEASURED',
        scope='Independent starts are the statistical unit. Enumerate the empirical paired bootstrap. Three starts give a coarse interval. A capacity-point ratio compares selected operating points, not equal arrival QPS. Absolute SLO eligibility and same-rate Decode noninferiority are separate. Failed starts remain in all summaries. No exact global maximum or infinite-horizon guarantee is claimed.')
    write(ROOT/'analysis/frozen_independent_start_statistics.json',result)
    return result


def interval_details(values):
    limits=json.loads((ROOT/'configs/frozen_SLO_regimes.json').read_text())['output_caps'];results=[]
    for value in values:
        rows=[r for r in records(ROOT/'raw'/value['name']/'requests.jsonl')
            if value['window_begin']<=r['arrival']<value['window_end']]
        cfg=value['configuration'];first=[];steady=[];pairs=[];per_regime={}
        for row in rows:
            events=row['token_events'];intervals=[]
            for i,(a,b) in enumerate(zip(events,events[1:])):
                delta=b['count']-a['count'];parent=1000*(b['time']-a['time'])/delta
                intervals.append(parent);(first if i==0 else steady).append(parent)
                if 'worker_time' in a and 'worker_time' in b:
                    worker=1000*(b['worker_time']-a['worker_time'])/delta
                    pairs.append(dict(sample_id=row['sample_id'],token_from=a['count'],token_to=b['count'],
                        parent_ms=parent,worker_ms=worker,receive_lag_change_ms=parent-worker))
            for regime,slo in limits[str(cfg['request_caps'][row['sample_id']])].items():
                item=per_regime.setdefault(regime,dict(requests=0,TTFT_pass=0,Decode_pass=0,ITL_threshold_exceedances=0))
                item['requests']+=1;item['TTFT_pass']+=row['ttft_ms']<=slo['TTFT_ms']
                item['Decode_pass']+=row['tpot_ms']<=slo['TPOT_ms'] and max(intervals,default=0)<=slo['ITL_ms']
                item['ITL_threshold_exceedances']+=sum(x>slo['ITL_ms'] for x in intervals)
        def distribution(points):return dict(count=len(points),p50_ms=percentile(points,.5),
            p95_ms=percentile(points,.95),p99_ms=percentile(points,.99),maximum_ms=max(points,default=None))
        for item in per_regime.values():
            item['TTFT_attainment']=item['TTFT_pass']/item['requests']
            item['Decode_attainment']=item['Decode_pass']/item['requests']
        results.append(dict(name=value['name'],first_ITL=distribution(first),steady_ITL=distribution(steady),
            maximum_parent_interval_identity=max(pairs,key=lambda p:p['parent_ms']) if pairs else None,
            worker_emission_ITL_p99_ms=percentile([p['worker_ms'] for p in pairs],.99),regimes=per_regime))
    write(ROOT/'analysis/frozen_heldout_interval_details.json',dict(runs=results,evidence='MEASURED',
        scope='Use the measurement-arrival cohort. The first ITL remains in the main SLO test. Worker emission intervals still include Host and GPU scheduling. A parent interval equals the worker interval plus receive-lag change. This identity does not identify an exclusive GPU cost or the cause of delivery delay.'))


def plots(values,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none'})
    names=('Native','Colocated','Static','Dynamic');colors=('#666666','#0072b2','#009e73','#d55e00')
    capacity=[v for v in values if v['name'].startswith('heldout_capacity_')]
    for index,key,ylabel,scale in ((1,'TTFT_p95_ms','TTFT p95 (s)',.001),
        (2,'TPOT_p95_ms','Request TPOT p95 (ms)',1.),
        (3,'raw_completed_output_tokens_s','Raw completed output tokens/s',1.)):
        fig,ax=plt.subplots(figsize=(7,4))
        for method,color in zip(names,colors):
            selected=[v for v in capacity if v['method']==method]
            if not selected:continue
            value=summary['methods'][method]['metrics'][key];middle=value['mean'];lo,hi=value['CI95']
            rate=selected[0]['rate']
            ax.scatter([v['rate'] for v in selected],[v[key]*scale for v in selected],c=color,s=18,alpha=.4)
            ax.errorbar([rate],[middle*scale],yerr=[[max(0.,middle-lo)*scale],[max(0.,hi-middle)*scale]],
                color=color,fmt='o',capsize=4,label=method)
        ax.set_xlabel('Offered requests/s');ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend()
        ax.set_title('Frozen Output16 operating points: independent-start 95% empirical CI')
        fig.tight_layout();fig.savefig(ROOT/'figures'/f'final_{index}_{key}.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,regime in zip(axes,('Balanced','Decode_priority')):
        for method,color in zip(names,colors):
            selected=[v for v in capacity if v['method']==method]
            if not selected:continue
            row=summary['methods'][method]['SLO_regimes'][regime];value=row['qualified_tokens_s']
            mid=value['mean'];lo,hi=value['CI95'];rate=selected[0]['rate']
            ax.errorbar([rate],[mid],yerr=[[max(0.,mid-lo)],[max(0.,hi-mid)]],fmt='o',capsize=4,
                color=color,mfc=color if row['all_three_qualified'] else 'white',label=method)
        ax.set_title(regime);ax.set_xlabel('Offered requests/s');ax.grid(alpha=.25)
    axes[0].set_ylabel('SLO-qualified completed output tokens/s');axes[0].legend()
    fig.suptitle('Filled: all three starts pass the operational screen. Hollow: not confirmed.')
    fig.tight_layout();fig.savefig(ROOT/'figures/final_4_SLO_qualified_throughput.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,regime in zip(axes,('Balanced','Decode_priority')):
        for i,(method,color) in enumerate(zip(names,colors)):
            row=summary['methods'][method]['SLO_regimes'][regime];value=row['qualified_tokens_s']
            if value['mean'] is None:continue
            mid=value['mean'];lo,hi=value['CI95']
            ax.bar(i,mid,color=color,alpha=.85 if row['all_three_qualified'] else .25,
                hatch=None if row['all_three_qualified'] else '//')
            ax.errorbar(i,mid,yerr=[[max(0.,mid-lo)],[max(0.,hi-mid)]],color='black',capsize=3)
            ax.text(i,mid+1,f"{row['passed_starts']}/{row['starts']} pass",ha='center',fontsize=8)
        ax.set_xticks(range(4),names);ax.set_title(regime);ax.set_ylabel('Qualified output tokens/s');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Largest selected tested points, not exact global maxima. Hatched points are ineligible.')
    fig.tight_layout();fig.savefig(ROOT/'figures/final_5_confirmed_capacity_points.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));histograms=[]
    for value in sorted([v for v in capacity if v['method']=='Dynamic'],key=lambda v:v['seed']):
        counts=collections.Counter(r.get('completed_D_prefill_layers',0) for r in records(ROOT/'raw'/value['name']/'requests.jsonl')
            if value['window_begin']<=r['arrival']<value['window_end'])
        histograms.append(dict(name=value['name'],counts=dict(counts)))
    for i,row in enumerate(histograms):
        left=0.;total=sum(row['counts'].values())
        for cut,color in zip((0,2,4,8,12),('#cccccc','#56b4e9','#009e73','#e69f00','#d55e00')):
            part=row['counts'].get(cut,0)/total;ax.barh(i,part,left=left,color=color,label=f'Last{cut}' if i==0 else None);left+=part
    ax.set_yticks(range(len(histograms)),[v['name'].split('_s')[1].split('_')[0] for v in histograms])
    ax.set_xlabel('Actual completed request fraction in the measurement-arrival cohort');ax.set_xlim(0,1);ax.legend(ncol=5,fontsize=8)
    ax.set_title('Frozen online Dynamic decisions, not forced-cut qualification')
    fig.tight_layout();fig.savefig(ROOT/'figures/final_6_actual_Dynamic_cuts.svg');plt.close(fig)
    write(ROOT/'analysis/final_figure_provenance.json',dict(time=time.time(),runs=[v['name'] for v in values],
        Dynamic_histograms=histograms,evidence='MEASURED',physical_kernel_activity='NOT_MEASURED',
        scope='No connection between different output lengths or tuned settings. Error bars resample independent starts. These plots do not measure exclusive GPU activity. Failed SLO and queue screens remain visible.'))


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'Large post-processing must not contaminate primary timing'
    values=load();summary=statistics_report(values);plots(values,summary);interval_details(values)
    inventory=json.loads((ROOT/'analysis/campaign_result_inventory.json').read_text())
    actual={r['name']:r for r in inventory['runs']}
    fields=['name','method','seed','rate','realized_window_arrival_QPS',*METRICS,
        'Decode_priority_qualified_tokens_s','Decode_priority_joint_attainment',
        'Decode_priority_screen_passed','actual_migrated_requests','actual_migrated_layer_work',
        'measurement_arrival_request_count','measurement_arrival_migrated_requests','measurement_arrival_migrated_layer_work']
    table=[]
    for value in values:
        row={k:value[k] for k in ['name','method','seed','rate',*METRICS]}
        regime=value['SLO_regimes']['Decode_priority']
        row.update(realized_window_arrival_QPS=value['stability']['arrival_rate'],
            Decode_priority_qualified_tokens_s=regime['SLO_qualified_output_tokens_s'],
            Decode_priority_joint_attainment=regime['arrival_cohort_joint_attainment'],
            Decode_priority_screen_passed=regime['qualified'],
            actual_migrated_requests=actual[value['name']]['actual_migrated_requests'],
            actual_migrated_layer_work=actual[value['name']]['actual_migrated_layer_work'],
            **{key:actual[value['name']][key] for key in ('measurement_arrival_request_count',
                'measurement_arrival_migrated_requests','measurement_arrival_migrated_layer_work')})
        table.append(row)
    with (ROOT/'analysis/frozen_confirmation_main_table.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows(table)
    print('FROZEN_START_STATISTICS',len(values),flush=True)


if __name__=='__main__':main()
