"""Plot qualified static cells. Keep replay and cost diagnostics separate."""
import collections
import json
import shutil
import numpy as np
from .common import ROOT,REPO,write
from .figures import finish,interval,draw_ci_points


def read(name):
    path=ROOT/'analysis'/name
    return json.loads(path.read_text()) if path.exists() else None


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'pdf.fonttype':42})
    manifest=[];excluded=[];groups=collections.defaultdict(list)
    for filename in ('BI0_shared_KV128_confirmation_v2.json','BI0_static_QPS_extension_v1.json','BI0_controlled_low_QPS_followup_v1.json'):
        value=read(filename)
        if not value:continue
        for row in value['results']:
            rate=row.get('QPS',row['metrics']['offered_QPS'])
            groups[(row['label'],rate)].append(row)
    for key,rows in list(groups.items()):
        if not all(r['metrics']['quality_gate']['CI_passed'] for r in rows):
            excluded.append(dict(cell=list(key),names=[r['name'] for r in rows],reason='FULL200_TASK_QUALITY_CI_FAILED'))
            del groups[key]
    colors={'Agg':'#d95f02','ContinuousNative':'#009e73','ControlledNative':'#555555','Layer':'#0072b2','Token':'#cc79a7'}
    labels={'Agg':'Agg DP2 C3','ContinuousNative':'Native PD C3','ControlledNative':'Native PD C0 batch1','Layer':'Static Layer C0 batch1','Token':'Static Token C0 batch1'}
    def display_name(label,cut):
        if label=='Layer' and cut:return labels[label]+f' Last{cut}'
        if label=='Token' and cut:return labels[label]+f' suffix~L*{cut}/36'
        return labels[label]
    fields=(('TTFT_p95_ms','TTFT p95 (ms)','TTFT_p95'),('ITL_p99_ms','Client-visible ITL p99 (ms)','ITL_p99'),
        ('TPOT_p95_ms','Request TPOT p95 (ms/token)','TPOT_p95'),
        ('generated_output_tokens_per_second','Generated output tokens/s','output_throughput'),
        ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','SLO_goodput'),
        ('Joint_SLO_attainment','Joint SLO attainment','SLO_attainment'))
    if groups:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        panels=(('Joint_SLO_attainment','Joint SLO attainment (%)',100.),
            ('TTFT_mean_ms','Mean TTFT (ms)',1.),('mean_TBT_ms','Mean TBT (ms)',1.),
            ('generated_output_tokens_per_second','Generated output tokens/s',1.))
        for ax,(field,ylabel,scale) in zip(axes.flat,panels):
            for label in labels:
                cells=sorted((q,rows) for (name,q),rows in groups.items() if name==label)
                if not cells:continue
                x=[];y=[];lo=[];hi=[]
                for q,rows in cells:
                    values=[scale*r['metrics'][field] for r in rows];bounds=interval(values)
                    x.append(q);y.append(float(np.mean(values)))
                    lo.append(bounds[0] if bounds else np.nan);hi.append(bounds[1] if bounds else np.nan)
                cut=cells[0][1][0]['cut'];legend=display_name(label,cut)
                ax.plot(x,y,'o-',color=colors[label],label=legend)
                ax.fill_between(x,lo,hi,color=colors[label],alpha=.13)
                draw_ci_points(ax,x,lo,hi,colors[label])
            ax.set_xlabel('Offered QPS (requests/s)');ax.set_ylabel(ylabel)
            ax.set_xticks(sorted({q for _,q in groups}))
            ax.set_title(ylabel.split(' (')[0])
            if field=='Joint_SLO_attainment':ax.set_ylim(0,105)
        handles,legend=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.015),fontsize=8)
        counts={q:sorted({len(rows) for (_,rate),rows in groups.items() if rate==q}) for q in sorted({rate for _,rate in groups})}
        fig.suptitle('BI0 ShortAnswer200 | 2 x H100 | B8192/S1024 | shared KV128 | Natural EOS')
        fig.text(.01,-.055,f'Independent starts per measured method/QPS: {counts}. Start-level95% CI only when n>=3.',fontsize=8)
        fig.text(.01,-.078,'C3 continuous external baselines and C0 actual-batch1 PD controls are distinct. Finite cohorts. No Dynamic result.',fontsize=8)
        fig.text(.01,-.101,'Frozen SLO: TTFT <=5000ms, every client ITL <=34ms, request TPOT <=33ms.',fontsize=8)
        fig.tight_layout(rect=(0,.055,1,.96))
        finish(fig,'bi0_shared128_four_metrics_confirmed',manifest,'MEASURED_ONLINE; distinct C3/C0 execution families',list(groups));plt.close(fig)
        for suffix in ('png','pdf'):
            shutil.copyfile(ROOT/'figures'/f'bi0_shared128_four_metrics_confirmed.{suffix}',
                REPO/'poc_flashvep'/f'bi0_shortanswer200_shared128_four_metrics_confirmed.{suffix}')
    for external in (False,True):
        allowed=('ControlledNative','Layer','Token') if not external else tuple(labels)
        if not any(label in allowed for label,_ in groups):continue
        for field,ylabel,stem in fields:
            fig,ax=plt.subplots(figsize=(8,4.8))
            for label in allowed:
                cells=sorted((q,rows) for (l,q),rows in groups.items() if l==label)
                if not cells:continue
                x=[];y=[];lo=[];hi=[]
                for q,rows in cells:
                    values=[r['metrics'][field] for r in rows];bounds=interval(values);point=float(np.mean(values))
                    x.append(q);y.append(point);lo.append(bounds[0] if bounds else np.nan);hi.append(bounds[1] if bounds else np.nan)
                    ax.annotate(f'n={len(rows)}',(q,point),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
                cut=next(iter(cells))[1][0]['cut']
                title=display_name(label,cut)
                ax.plot(x,y,marker='o',linestyle='--' if label in ('Agg','ContinuousNative') else '-',color=colors[label],label=title)
                ax.fill_between(x,lo,hi,color=colors[label],alpha=.14)
                draw_ci_points(ax,x,lo,hi,colors[label])
            ax.set_xlabel('Offered QPS (requests/s)');ax.set_ylabel(ylabel);ax.legend(fontsize=8)
            ax.set_xticks(sorted({q for label,q in groups if label in allowed}))
            if field=='Joint_SLO_attainment':ax.set_ylim(0,1.05)
            ax.set_title('BI0 Natural200 | KV128 | B8192/S1024 | '+('different execution families' if external else 'PD actual batches1/1/1'))
            fig.text(.01,-.02,'Discovery-frozen cuts. Start-level95% CI only when n>=3. Finite cohorts, not sustainable capacity.',fontsize=7)
            fig.text(.01,-.045,'C3 continuous external baselines are not actual-batch matched. No Dynamic result is implied.' if external else 'Source credit4. Quiescent-only D admission. Not active-decode gap insertion.',fontsize=7)
            if field in ('Joint_SLO_attainment','SLO_qualified_output_tokens_per_second'):
                fig.text(.01,-.07,'Frozen SLO: TTFT <=5000ms, every client ITL <=34ms, request TPOT <=33ms.',fontsize=7)
            finish(fig,f'bi0_shared128_{"external" if external else "controlled"}_{stem}_vs_qps',manifest,
                'MEASURED_ONLINE; quality-qualified whole cells only',list(groups));plt.close(fig)
            for suffix in ('png','pdf'):
                source=ROOT/'figures'/f'bi0_shared128_{"external" if external else "controlled"}_{stem}_vs_qps.{suffix}'
                shutil.copyfile(source,REPO/'poc_flashvep'/f'bi0_shortanswer200_shared128_{"external" if external else "controlled"}_{stem}_vs_qps.{suffix}')
    discovery=read('BI0_controlled_fixed_axis_discovery_v3.json')
    from .pipeline_queues import main as pipeline_analysis
    queue_lookup={r['name']:r for r in pipeline_analysis()['results']}
    if any(label in ('ControlledNative','Layer','Token') for label,_ in groups):
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,key,title in zip(axes,('P_pipeline_waiting','D_pipeline_waiting','D_pipeline_active'),
                ('P coordinator waiting','D prefill/admission waiting','D active request')):
            for label in ('ControlledNative','Layer','Token'):
                cells=sorted((q,rows) for (l,q),rows in groups.items() if l==label)
                points=[]
                for q,rows in cells:
                    values=[queue_lookup[r['name']]['queues'][key]['mean'] for r in rows]
                    bounds=interval(values);points.append((q,float(np.mean(values)),bounds,len(rows)))
                if not points:continue
                ax.plot([p[0] for p in points],[p[1] for p in points],'o-',label=label,color=colors[label])
                for q,value,_,count in points:ax.annotate(f'n={count}',(q,value),fontsize=7,xytext=(0,5),textcoords='offset points')
                ax.fill_between([p[0] for p in points],[p[2][0] if p[2] else np.nan for p in points],
                    [p[2][1] if p[2] else np.nan for p in points],color=colors[label],alpha=.14)
                draw_ci_points(ax,[p[0] for p in points],[p[2][0] if p[2] else np.nan for p in points],
                    [p[2][1] if p[2] else np.nan for p in points],colors[label])
            ax.set_xlabel('Offered QPS');ax.set_ylabel('Time-weighted requests');ax.set_title(title);ax.legend(fontsize=7)
            ax.set_xticks(sorted({q for label,q in groups if label in ('ControlledNative','Layer','Token')}))
        fig.suptitle('Actual batches1/1/1 | host coordinator states | not physical GPU activity')
        fig.text(.01,-.015,'D admission waiting differs from worker decode-ready waiting. Quiescent-only admission does not create a predicted active-decode window.',fontsize=7)
        fig.tight_layout();finish(fig,'bi0_controlled_host_queues_vs_qps',manifest,'TRACE_ANALYSIS of measured host states',list(queue_lookup));plt.close(fig)
    if discovery:
        cuts=(1,2,4,6,8,12);lookup={(r['axis'],r['cut']):r for r in discovery['results']}
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for axis in ('Layer','Token'):
            selected=[lookup.get((axis,c)) for c in cuts]
            for plot,(field,ylabel) in zip(axes,(('TTFT_p95_ms','TTFT p95 (ms)'),('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s'))):
                valid=[r for r in selected if r and r['quality']['CI_passed']]
                bad=[r for r in selected if r and not r['quality']['CI_passed']]
                plot.plot([r['cut'] for r in valid],[r['metrics'][field] for r in valid],'o-',label=axis,color=colors[axis])
                if bad:plot.scatter([r['cut'] for r in bad],[r['metrics'][field] for r in bad],marker='x',color='gray',label=axis+' quality-unqualified')
                plot.set_xlabel('Cut m; Token suffix approximately L*m/36');plot.set_ylabel(ylabel);plot.set_xticks(cuts);plot.legend(fontsize=8)
        fig.suptitle('BI0 Natural200 | QPS2.35 | Actual batches1/1/1 | One discovery start/cell')
        fig.text(.01,-.02,'No start-level CI at n=1. Natural output work can differ. Gray markers are not qualified performance.',fontsize=7)
        finish(fig,'bi0_fixed_cut_discovery',manifest,'MEASURED_ONLINE discovery only',['analysis/BI0_controlled_fixed_axis_discovery_v3.json']);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for plot,(field,label) in zip(axes,(('TTFT_p95_ms','Layer / Token TTFT p95'),('SLO_qualified_output_tokens_per_second','Layer / Token SLO-goodput'))):
            x=[];y=[]
            for cut in cuts:
                a=lookup.get(('Layer',cut));b=lookup.get(('Token',cut))
                if not a or not b or not a['quality']['CI_passed'] or not b['quality']['CI_passed']:continue
                if b['metrics'][field]<=0:continue
                x.append(cut);y.append(a['metrics'][field]/b['metrics'][field])
            plot.plot(x,y,'o-',color='#0072b2');plot.axhline(1,color='black',linestyle='--',linewidth=1)
            plot.set_xticks(cuts);plot.set_xlabel('Offloaded layers m');plot.set_ylabel(label)
        fig.suptitle('Matched-request-batch axis ratios | one discovery start/cell | not confirmed winners')
        finish(fig,'bi0_axis_cut_ratios_discovery',manifest,'MEASURED_ONLINE discovery only',['analysis/BI0_controlled_fixed_axis_discovery_v3.json']);plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(12,8),sharey=True)
        for row_index,cut in enumerate((4,8,12)):
            for col_index,axis in enumerate(('Layer','Token')):
                ax=axes[row_index,col_index];cell=lookup.get((axis,cut))
                if not cell:ax.text(.5,.5,'NOT_RUN',ha='center',va='center',transform=ax.transAxes);continue
                folder=ROOT/'raw'/cell['name'];origin=json.loads((folder/'window.json').read_text())['start']
                sampled=[json.loads(line) for line in (folder/'coordinator_queues.jsonl').read_text().splitlines()]
                for key,label,color in (('P_pipeline_waiting','P waiting','#d95f02'),('D_pipeline_waiting','D admission waiting','#009e73')):
                    ax.step([r['time']-origin for r in sampled],[r[key] for r in sampled],where='post',label=label,color=color)
                ax.set_title(f'{axis}{cut}: separate server run');ax.set_ylabel('Requests');ax.set_xlabel('Time since this run start (s)');ax.legend(fontsize=7)
        fig.suptitle('Representative host queue timelines | QPS2.35 | n=1 per panel | not GPU timelines')
        fig.tight_layout();finish(fig,'bi0_fixed_cut_host_queue_timelines',manifest,'TRACE_ANALYSIS; each panel is a separate run',
            ['analysis/BI0_controlled_fixed_axis_discovery_v3.json']);plt.close(fig)
    costs=read('BI0_continuation_serial_costs.json')
    if costs:
        fig,axes=plt.subplots(2,2,figsize=(11,7));counts={}
        for ax,cut in zip(axes.flat,(1,4,8,12)):
            for axis in ('Layer','Token'):
                values=[]
                for row in costs['results']:
                    if row['axis']!=axis or row['cut']!=cut:continue
                    trace=read(row['name']+'_axis_timeline.json')
                    values.extend(r['ready_to_first_worker_output_ms'] for r in trace['rows'])
                if not values:continue
                values=np.sort(values);counts[f'{axis}{cut}']=len(values)
                ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=f'{axis}, n={len(values)}',color=colors[axis])
            ax.set_xlabel('Observed Ready-to-first-worker service (ms)');ax.set_ylabel('Observed sample fraction');ax.set_title(f'Cut {cut}');ax.legend(fontsize=8)
        fig.suptitle('Isolated serial candidate latency CDF | NOT executable-window success')
        fig.text(.01,-.01,'Real prompts, host service enclosures. No queue from other requests. Small samples are not WCET or a calibrated safety probability.',fontsize=7)
        fig.tight_layout();finish(fig,'bi0_serial_candidate_latency_CDF',manifest,'ISOLATED_SERIAL_SERVICE_MEASUREMENT; not safe-fit rate',counts);plt.close(fig)
    oracle_source='BI0_conditional_axis_oracle_v2.json' if (ROOT/'analysis/BI0_conditional_axis_oracle_v2.json').exists() else 'BI0_conditional_axis_oracle_v1.json'
    oracle=read(oracle_source)
    if oracle:
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for ax,rate in zip(axes,(1.2,1.65,2.35)):
            rows=[r for r in oracle['cells'] if r['QPS']==rate];positions=np.arange(len(rows));width=.2
            for index,label in enumerate(('Fixed','Layer','Token','All')):
                values=[r['discovery_frozen_fixed']['metrics']['modeled_TTFT_qualified_tokens_per_second'] if label=='Fixed' else
                    r['oracles'][label]['metrics']['modeled_TTFT_qualified_tokens_per_second'] for r in rows]
                ax.bar(positions+(index-1.5)*width,values,width,label=label)
            ax.set_xticks(positions,[str(r['seed'])[-2:] for r in rows]);ax.set_xlabel('Arrival seed suffix;61 discovery,62/63 held-out')
            ax.set_ylabel('TTFT-qualified token proxy/s');ax.set_title(f'QPS{rate}');ax.legend(fontsize=7)
        fig.suptitle('OFFLINE_ORACLE | state-updating FIFO/source-credit4 | not full joint-SLO goodput')
        fig.text(.01,-.015,'Oracle uses future arrivals/EOS. Beam24 is not a global bound. No ITL-tail or task-quality model. Not measured online gain.',fontsize=7)
        fig.tight_layout();finish(fig,'bi0_conditional_oracle_proxy_headroom',manifest,'OFFLINE_ORACLE',['analysis/'+oracle_source]);plt.close(fig)
    write(ROOT/'figures/serving_figure_manifest.json',dict(figures=manifest,excluded=excluded,
        missing='Missing cells and Dynamic controllers are not fabricated. Costs and oracle proxies remain separate from online results.'))
    statuses=[]
    for number,stem in enumerate(('TTFT_p95','ITL_p99','TPOT_p95','output_throughput','SLO_goodput'),1):
        path=ROOT/'figures'/f'bi0_shared128_external_{stem}_vs_qps.png'
        statuses.append(dict(requested_figure=number,status='MEASURED_ONLINE_STATIC_ONLY' if path.exists() else 'NOT_RUN',
            path=str(path.relative_to(REPO)) if path.exists() else None,
            limitation='Dynamic is not measured. C3 external baselines and C0 batch-matched controls have different execution families.'))
    statuses.extend([
        dict(requested_figure=6,status='PARTIAL_MEASUREMENT',limitation='Clean P-only busy/queue timeline exists. Qualified simultaneous P/D kernel busy is not measured.'),
        dict(requested_figure=7,status='NOT_MEASURED',limitation='Host/current-stream quiescence is plotted separately. It is not a future decode-safe window distribution.'),
        dict(requested_figure=8,status='NOT_RUN',limitation='No qualified future safe-window budget exists. A serial candidate latency CDF is not a window fit probability.'),
        dict(requested_figure=9,status='DISCOVERY_STATIC_COMPARISON',limitation='Measured fixed-cut ratios are plotted. No replicated multi-input/output TTFT-gain heatmap exists.'),
        dict(requested_figure=10,status='NOT_RUN',limitation='No online dynamic controller was qualified. Static and offline choices are not dynamic online decisions.')])
    write(ROOT/'figures/requested_figure_evidence_status.json',dict(figures=statuses,
        scope='Missing evidence remains explicit. No placeholder is presented as a measured curve.'))
    return manifest


if __name__=='__main__':main()
