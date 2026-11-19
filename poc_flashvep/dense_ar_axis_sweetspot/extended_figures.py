"""Plot measured host mechanisms separately from isolated and replay results."""
import json
import math
import statistics
from .common import ROOT,REPO,write,digest
from .replay import Cost
from .axis_oracle import demand_scores,cycle_boundary
from .figures import STYLE


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .resampling import mean_interval
    stats=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())
    path=ROOT/'analysis/figure_manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else dict(figures=[])
    def save(fig,stem,evidence,description,sources):
        files=[]
        for ext in ('.png','.pdf'):
            target=ROOT/'figures'/(stem+ext)
            fig.savefig(target,dpi=175,bbox_inches='tight')
            files.append(dict(path=str(target.relative_to(REPO)),SHA256=digest(target)))
        plt.close(fig)
        manifest['figures']=[r for r in manifest['figures'] if r['stem']!=stem]
        manifest['figures'].append(dict(stem=stem,evidence=evidence,description=description,
            files=files,sources=sources))

    output_cells=sorted([r for r in stats['comparisons'] if r['stage']=='output_residency_counterfactual'
        and r['reference']=='Token' and r['candidate']=='Layer' and r['layers']==4 and r['QPS']==6
        and r['batch']==2 and r['starts']>=3],key=lambda r:r['output'])
    if {r['output'] for r in output_cells}=={16,64}:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,key,label in ((axes[0],'TTFT','Layer / Token client TTFT p95'),
                (axes[1],'Goodput','Layer / Token qualified output token/s')):
            for r in output_cells:
                v=r['metrics']['TTFT_p95_ms'] if key=='TTFT' else r['common_Balanced_goodput']
                ci=v['ratio_CI95']
                ax.errorbar([r['output']],[v['ratio']],yerr=[[v['ratio']-ci[0]],[ci[1]-v['ratio']]],
                    fmt='o',linestyle='none',capsize=4,color='#0072b2')
                ax.annotate('n='+str(r['starts']),(r['output'],v['ratio']),xytext=(6,0),textcoords='offset points',fontsize=8)
            ax.axhline(1,color='gray',ls='--');ax.set_xticks((16,64));ax.set_xlabel('Fixed generated Output cap (tokens)')
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].set_title('Lower favors Layer');axes[1].set_title('Higher favors Layer')
        fig.suptitle('MEASURED_ONLINE | Last4 / exact matched Token work | Input8000 / QPS6 / actual1/1/2 / N72')
        fig.text(.01,.01,'Three paired independent starts. Frozen common SLO. Context cap also derives8016→8064. Finite overloaded cohorts, not sustainable capacity. Intermediate Output caps are NOT_RUN.',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,.93))
        save(fig,'controlled_output_cap_crossover','MEASURED_ONLINE',
            'The same matched partition and actual batch change rank across Output caps. No exact intermediate crossover threshold or pure-GPU causal effect is inferred.',output_cells)

    operator_path=ROOT/'analysis/actual_serving_API_operator_analysis.json'
    if operator_path.exists():
        op=json.loads(operator_path.read_text())
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        rows=op['matched_projection_models']
        axes[0].plot([r['layers'] for r in rows],[r['Token_over_Layer_D_projection_model'] for r in rows],marker='o')
        axes[0].set_ylabel('Token / Layer projection model (higher favors Layer)')
        axes[0].set_title('STRUCTURAL_ACCOUNTING\nWeighted isolated serving-API projection medians')
        used=[]
        for b in (1,2,4):
            selected=[]
            for m in (2,4,8,12):
                choices=[r for r in stats['comparisons'] if r['reference']=='Token' and r['candidate']=='Layer' and
                    r['layers']==m and r['output']==16 and r['QPS']==6 and r['batch']==b and r['window']==32]
                preferred=[r for r in choices if r['starts']>=3]
                if not choices:continue
                r=(preferred or choices)[0];selected.append(r);used.append(r)
            if selected:
                axes[1].plot([r['layers'] for r in selected],[r['metrics']['TTFT_p95_ms']['ratio'] for r in selected],
                    marker='o',label='Decode'+str(b))
                for r in selected:
                    v=r['metrics']['TTFT_p95_ms'];ci=v['ratio_CI95']
                    if ci:axes[1].errorbar([r['layers']],[v['ratio']],yerr=[[max(0,v['ratio']-ci[0])],[max(0,ci[1]-v['ratio'])]],capsize=3)
        axes[1].set_title('MEASURED_ONLINE\nO16/QPS6, actual P1/D-suffix1')
        axes[1].set_ylabel('Layer / Token client TTFT p95 (lower favors Layer)');axes[1].legend(fontsize=8)
        for ax in axes:
            ax.axhline(1,color='gray',ls='--');ax.set_xticks((2,4,8,12));ax.set_xlabel('Cut / exact matched Token work');ax.grid(alpha=.2)
        fig.text(.01,.01,'Left is neither whole Forward nor TTFT. Right uses finite cohorts and paired starts. CI absent at n=1. Seed sets can differ across Decode protocols.',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1))
        save(fig,'isolated_projection_vs_online_TTFT','ISOLATED_KERNEL_MEASUREMENT_AND_MEASURED_ONLINE',
            'Keep the weighted projection model and real client latency in separate panels.',
            [op['name']]+[dict(stage=r['stage'],layers=r['layers'],batch=r['batch'],starts=r['starts'],seeds=r['seeds']) for r in used])

    causal_path=ROOT/'analysis/service_queue_causal_decomposition.json'
    if causal_path.exists():
        ca=json.loads(causal_path.read_text())
        amplification=ca.get('historical_queue_amplification_proxy',[])
        if amplification:
            fig,ax=plt.subplots(figsize=(7,5))
            colors={4:'#d55e00',8:'#0072b2',12:'#009e73'}
            for m in (4,8,12):
                rows=[r for r in amplification if r['layers']==m]
                ax.scatter([r['cycle_delta_times_tail_rank_ms'] for r in rows],
                    [r['measured_Layer_minus_Token_TTFT_p95_ms'] for r in rows],
                    color=colors[m],label='Last'+str(m)+' / n='+str(len(rows)))
            vals=[r[k] for r in amplification for k in ('cycle_delta_times_tail_rank_ms','measured_Layer_minus_Token_TTFT_p95_ms')]
            low,high=min(vals)-200,max(vals)+200
            ax.plot([low,high],[low,high],ls='--',color='gray');ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6)
            ax.set_xlabel('Cycle difference × 0.95(N−1) approximation (ms)')
            ax.set_ylabel('Measured Layer−Token TTFT p95 difference (ms)');ax.grid(alpha=.2);ax.legend()
            ax.set_title('HISTORICAL TRACE_ANALYSIS | N144 / O16 / q6 / actual1/1/2')
            fig.text(.01,.01,'Costs come from the same completed trace. This is a post-run explanation, not held-out prediction, exclusive GPU time or a causal proof.',fontsize=8)
            fig.tight_layout(rect=(0,.07,1,1))
            save(fig,'historical_queue_amplification_proxy','TRACE_ANALYSIS',
                'Compare the deterministic overloaded-pipeline proxy with measured tail differences, using all three cuts and starts.',amplification)
        rows=ca.get('source_window_independent_start_summary',[])
        if rows:
            fig,axes=plt.subplots(1,3,figsize=(14,4));xs=list(range(len(rows)))
            for ax,field,label in ((axes[0],'Ready_D_mean_ms','Ready→D Δ (ms/request)'),
                    (axes[1],'P_external_wait_mean_ms','Arrival→P submit Δ (ms/request)')):
                vals=[r['mean_delta_ms'][field]['mean'] for r in rows]
                ax.bar(xs,vals,color=[STYLE[r['method']][0] for r in rows])
                for x,v,r in zip(xs,vals,rows):
                    ci=r['mean_delta_ms'][field]['CI95']
                    if ci:ax.errorbar([x],[v],yerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],color='black',capsize=3)
                ax.axhline(0,color='gray',lw=1);ax.set_ylabel(label)
            for x,r in zip(xs,rows):
                value=r['outcome_ratios']['TTFT_p95_ratio_W4_over_W32']
                axes[2].plot([x],[value['mean']],'o',color=STYLE[r['method']][0])
                ci=value['CI95']
                if ci:axes[2].errorbar([x],[value['mean']],yerr=[[max(0,value['mean']-ci[0])],[max(0,ci[1]-value['mean'])]],color='black',capsize=3)
            axes[2].axhline(1,color='gray',ls='--');axes[2].set_ylabel('Client TTFT p95 W4/W32')
            for ax in axes:
                ax.set_xticks(xs,[r['method']+str(r['layers'])+'\nn='+str(r['starts']) for r in rows],fontsize=8)
                ax.grid(axis='y',alpha=.2)
            fig.suptitle('MEASURED_ONLINE | Source outstanding Window32→4 | O16/QPS6 | actual1/1/2')
            fig.text(.01,.01,'Paired independent starts, not requests. W4−W32 on left/middle. No statistical equivalence claim at one start. Host service drift must be inspected separately.',fontsize=8)
            fig.tight_layout(rect=(0,.06,1,.94))
            save(fig,'source_window_wait_relocation','MEASURED_ONLINE',
                'A source-window intervention can relocate waiting without an equal TTFT gain.',rows)

    model_path=ROOT/'configs/frozen_action_service_costs_v2.json'
    if model_path.exists():
        models=json.loads(model_path.read_text())['model']['models']
        cost={(r['method'],r['layers'],r['output'],r['batch']):Cost(**r['cost_seconds']) for r in models}
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        sources=[]
        for ax,m in zip(axes,(4,8,12)):
            table={a:cost[a,m,16,2] for a in ('Layer','Token')}
            boundary=cycle_boundary(table,2)
            demand=[i*.004 for i in range(101)]
            for a,c in table.items():
                p=c.P+c.P_gap;u=c.preparation+c.D_gap/2
                ax.plot([1000*d for d in demand],[1000*max(p,u+d) for d in demand],
                    color=STYLE[a][0],label=a)
            threshold=boundary.get('common_Decode_demand_threshold_seconds')
            if threshold is not None and 0<threshold<.4:ax.axvline(1000*threshold,color='gray',ls='--')
            for b,marker in ((1,'o'),(2,'s'),(4,'^')):
                d=statistics.median(c.decode for c in table.values())/b
                ax.scatter([1000*d],[1000*max(c.P+c.P_gap,c.preparation+c.D_gap/2+d) for c in [table['Layer']]],
                    color='black',marker=marker,s=20,label='d reference Decode'+str(b))
            ax.set_title('Last'+str(m));ax.set_xlabel('Common Decode cohort demand/request (ms)')
            ax.set_ylabel('Modeled saturated cycle (ms/request)');ax.grid(alpha=.2)
            sources.append(dict(layers=m,boundary=boundary))
        axes[0].legend(fontsize=7)
        fig.suptitle('STRUCTURAL_ACCOUNTING | max(p_axis, u_axis+d) | frozen Decode2 action costs')
        fig.text(.01,.01,'These are model curves, not measured TTFT. Markers only change the common Decode-demand reference. Actual Batch4 costs are extrapolated and validated separately.',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,.94))
        save(fig,'minimal_service_demand_boundary','STRUCTURAL_ACCOUNTING',
            'A one-demand model can have a crossing or a dominance region, depending on calibrated action costs.',sources)

    feedback_path=ROOT/'analysis/reservation_feedback_index.json'
    if feedback_path.exists():
        from .resampling import mean_interval
        feedback=json.loads(feedback_path.read_text())['runs']
        selected=[r for r in feedback if r['stage']=='decode_batch_counterfactual' and r['method'] in ('Layer','Token')]
        if selected:
            starts={r['name']:r for r in stats['starts']}
            fig,axes=plt.subplots(1,3,figsize=(14,4));used=[]
            for a in ('Layer','Token'):
                color,marker=STYLE[a]
                for key,ls,label in (('P','-','effective P'),('D','--','D guard/request')):
                    xs=[];means=[]
                    for b in (2,4):
                        rows=[r for r in selected if r['method']==a and r['batch']==b and r['mean_backlogged_nonwindow_P_gap_ms'] is not None]
                        if not rows:continue
                        vals=[starts[r['name']]['P_submit_to_ready_mean_ms']+r['mean_backlogged_nonwindow_P_gap_ms'] if key=='P'
                            else starts[r['name']]['D_host_slot_per_request_ms'] for r in rows]
                        v=statistics.mean(vals);ci=mean_interval(vals);xs.append(b);means.append(v)
                        if ci:axes[0].errorbar([b],[v],yerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],color=color,capsize=3)
                        used.extend(r['name'] for r in rows)
                    axes[0].plot(xs,means,color=color,marker=marker,ls=ls,label=a+' '+label)
            axes[0].set_ylabel('Host demand (ms/request)');axes[0].legend(fontsize=7)
            for b in (2,4):
                rows=[r for r in stats['comparisons'] if r['stage']=='decode_batch_counterfactual' and
                    r['reference']=='Token' and r['candidate']=='Layer' and r['batch']==b]
                if not rows:continue
                r=rows[0];v=r['metrics']['TTFT_p95_ms'];ci=v['ratio_CI95']
                axes[1].plot([b],[v['ratio']],'o',color='black')
                if ci:axes[1].errorbar([b],[v['ratio']],yerr=[[max(0,v['ratio']-ci[0])],[max(0,ci[1]-v['ratio'])]],color='black',capsize=3)
                axes[1].annotate('n='+str(r['starts']),[b,v['ratio']],xytext=(5,8),textcoords='offset points',fontsize=8)
            axes[1].axhline(1,color='gray',ls='--');axes[1].set_ylabel('Layer / Token TTFT p95')
            labels=[];ack=[];rest=[]
            for b in (2,4):
                for a in ('Layer','Token'):
                    rows=[r for r in selected if r['method']==a and r['batch']==b and r['mean_backlogged_nonwindow_P_gap_ms'] is not None]
                    labels.append(a+' / b'+str(b))
                    ack.append(statistics.mean(r['mean_backlogged_nonwindow_gap_parts_ms']['reservation_RTT_ms'] for r in rows))
                    rest.append(statistics.mean(r['mean_backlogged_nonwindow_P_gap_ms'] for r in rows)-ack[-1])
            axes[2].bar(range(4),ack,label='Mandatory reservation RTT')
            axes[2].bar(range(4),rest,bottom=ack,label='Other host gap')
            axes[2].set_xticks(range(4),labels,rotation=25,fontsize=8);axes[2].set_ylabel('Backlogged P gap (ms)');axes[2].legend(fontsize=7)
            for ax in axes[:2]:ax.set_xlabel('Actual Decode batch');ax.set_xticks((2,4));ax.grid(alpha=.2)
            fig.suptitle('MEASURED_ONLINE + TRACE_ANALYSIS | Last8 exact-row pair | Input8000/Output16/QPS6 | prep1')
            fig.text(.01,.01,'Source demand includes the next reservation ACK gap, not exclusive GPU time. CI resamples paired starts. Batch4 rank is unresolved when CI crosses1. No sustainable-capacity claim.',fontsize=8)
            fig.tight_layout(rect=(0,.07,1,.93))
            save(fig,'decode_batch_control_and_ACK_feedback','MEASURED_ONLINE_AND_TRACE_ANALYSIS',
                'Changing actual Decode batch alters the critical stage. Keep the unresolved reversal and mandatory ACK delay visible.',sorted(set(used)))

    for version in (2,3,4):
        feature_path=ROOT/('analysis/recorded_observable_knob_validation'+(f'_v{version}' if version>=3 else '')+'.json')
        if not feature_path.exists():continue
        doc=json.loads(feature_path.read_text());rows=doc['heldout_cell_evaluations']
        if rows:
            fig,ax=plt.subplots(figsize=(12,max(4,.29*len(rows)+1.3)))
            for y,r in enumerate(rows):
                for x,choice in ((0,r['confidence_label']),(1,r['majority_fixed_axis_forecast']['one']),
                        (2,r['majority_fixed_axis_forecast']['two'])):
                    color=STYLE[choice][0] if choice in STYLE else '#cccccc'
                    ax.scatter([x],[y],s=110,marker='s',color=color)
                    ax.text(x+.1,y,choice,va='center',fontsize=8)
            ax.set_yticks(range(len(rows)),[f"{r['stage']} / L{r['layers']} / O{r['output']} / q{r['QPS']:g} / b{r['batch']} / n{r['starts']}" for r in rows],fontsize=7)
            ax.set_xticks((0,1,2),('Measured95% CI label','One-pressure majority','Two-scalar majority'))
            ax.set_xlim(-.3,2.8);ax.invert_yaxis();ax.grid(axis='y',alpha=.15)
            ax.set_title(f'TRACE_ANALYSIS | Cost v{version} | Frozen rule shadow votes on pre-P host observations')
            fig.text(.01,.01,'Each row is a paired cell, not one token. Gray means unresolved CI. Votes do not change the measured fixed-axis run. This is not measured dynamic speedup.',fontsize=8)
            fig.tight_layout(rect=(0,.05,1,1))
            save(fig,'heldout_one_two_knob_classification'+(f'_v{version}' if version>=3 else ''),'TRACE_ANALYSIS',
                'Compare confident cell labels and frozen shadow-vote forecasts. Unresolved cells remain gray.',rows)

    prediction_models={}
    for version in (2,3,4):
        file=ROOT/('analysis/frozen_model_validation_and_oracle'+(f'_v{version}' if version>=3 else '')+'.json')
        if file.exists():prediction_models[version]=json.loads(file.read_text())['results']
    if 3 in prediction_models or 4 in prediction_models:
        fig,ax=plt.subplots(figsize=(9,4));used=[]
        for offset,(version,rows) in enumerate(sorted(prediction_models.items())):
            xs=[];vals=[]
            for q in (3.,6.):
                selected=[r for r in rows if r['stage']=='batch4_knob_confirmation' and r['QPS']==q]
                samples=[]
                for seed in sorted({r['seed'] for r in selected}):
                    r=next(r for r in selected if r['seed']==seed)
                    samples.append(statistics.mean(v['errors']['TTFT_p95_relative']*100 for v in r['fixed_trace_validation'].values()))
                if not samples:continue
                x=q+(offset-1)*.18;v=statistics.mean(samples);ci=mean_interval(samples)
                ax.bar([x],[v],width=.16,label='Cost v'+str(version) if not xs else None)
                if ci:ax.errorbar([x],[v],yerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],color='black',capsize=3)
                xs.append(x);vals.append(v);used.append(dict(version=version,QPS=q,seeds=sorted({r['seed'] for r in selected})))
        ax.axhline(10,color='gray',ls='--',label='Per-action timing gate10%')
        ax.set_xticks((3,6));ax.set_xlabel('Offered QPS');ax.set_ylabel('Mean paired-axis TTFT p95 relative error (%)')
        ax.set_title('OFFLINE_REPLAY validation | New Batch4 seeds1970/1971/1972');ax.legend(fontsize=8)
        fig.text(.01,.01,'The same measured traces test all frozen costs. Average errors across axes within each seed. CI uses paired seeds. The displayed mean does not replace each-action validation.',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,1))
        save(fig,'fresh_cost_model_prediction_errors','OFFLINE_REPLAY',
            'Separate unseen-batch bias, batch calibration and uncensored control-gap calibration. No test refitting.',used)

    control_path=ROOT/'analysis/control_boundary_replay_validation.json'
    if control_path.exists():
        rows=json.loads(control_path.read_text())['results']
        if rows:
            fig,axes=plt.subplots(1,3,figsize=(14,4))
            for a in ('Layer','Token'):
                selected=[r for r in rows if r['method']==a]
                for ax,key,label in ((axes[0],'TTFT_p95_relative','Client TTFT p95 relative error (%)'),
                        (axes[1],'Ready_D_mean_absolute_ms','Ready→D mean absolute error (ms)')):
                    vals=[r['errors'][key]*(100 if key.endswith('relative') else 1) for r in selected]
                    ax.scatter([r['QPS']+(-.07 if a=='Layer' else .07) for r in selected],vals,
                        color=STYLE[a][0],marker=STYLE[a][1],label=a)
                    ax.set_ylabel(label)
                axes[2].scatter([r['QPS']+(-.07 if a=='Layer' else .07) for r in selected],
                    [r['ACK_error']['MAE_ms'] for r in selected],color=STYLE[a][0],marker=STYLE[a][1],label=a)
            axes[0].axhline(10,color='gray',ls='--',label='TTFT gate10%')
            axes[2].set_ylabel('Reservation RTT prediction MAE (ms)')
            for ax in axes:
                ax.set_xticks((3,6));ax.set_xlabel('Offered QPS');ax.grid(alpha=.2);ax.legend(fontsize=7)
            passed=sum(r['timing_passed'] and r['SLO_passed'] for r in rows)
            fig.suptitle('OFFLINE_REPLAY validation | Explicit ACK/host-step model | Fixed-axis gates '+str(passed)+'/'+str(len(rows)))
            fig.text(.01,.01,'Each point is an independent server start. The same six calibration traces freeze this model before new measured arrivals. Preparation-step and async-detail assumptions can fail. No test refit.',fontsize=8)
            fig.tight_layout(rect=(0,.07,1,.93))
            save(fig,'explicit_control_boundary_replay_errors','OFFLINE_REPLAY',
                'Validate the mandatory ACK dependency directly. Fixed-axis validation failures remain visible. No actual dynamic gain is implied.',rows)

    oracle_path=ROOT/'analysis/fine_cohort_axis_oracle.json'
    if oracle_path.exists():
        rows=json.loads(oracle_path.read_text())['aggregate']
        if rows:
            fig,ax=plt.subplots(figsize=(10,max(4,.3*len(rows)+1.5)))
            for y,r in enumerate(rows):
                value=r['predicted_goodput_ratios']['oracle'];v=value['ratio'];ci=value['ratio_CI95']
                if v is None:continue
                color='#009e73' if r['all_fixed_validation_passed'] else '#999999'
                ax.plot([v],[y],'o',color=color)
                if ci:ax.errorbar([v],[y],xerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],color=color,capsize=3)
            ax.axvline(1,color='gray',ls='--');ax.axvline(1.05,color='#d55e00',ls=':',label='Research target5%')
            ax.set_yticks(range(len(rows)),[f"v{r['model_version']} / {r['stage']} / L{r['layers']} / O{r['output']} / q{r['QPS']:g} / b{r['batch']} / n{r['starts']}" for r in rows],fontsize=7)
            ax.invert_yaxis();ax.set_xlabel('Modeled qualified token/s: feasible beam / per-trace best fixed axis')
            ax.set_title('OFFLINE_ORACLE | Exact aggregate work / one axis per cohort | Beam32');ax.legend(fontsize=8)
            fig.text(.01,.01,'Green: all fixed-trace timing/SLO gates pass. Gray: Diagnostic. CI excludes model uncertainty. Finite beam is not a global bound, measured dynamic speedup or Decode-tail guarantee.',fontsize=8)
            fig.tight_layout(rect=(0,.05,1,1))
            save(fig,'fine_oracle_headroom','OFFLINE_ORACLE',
                'Show bounded-search headroom only with its fixed-trace validation status. Keep Diagnostic results gray.',rows)

    # Each panel is an independent run. P/D lanes inside a panel share one host clock.
    for m in (4,8,12):
        names={a:f'source_window_control_{a}_Last{m}_N72_O16_q6_s2026111901_B2_W32_v1' for a in ('Layer','Token')}
        hosts={a:ROOT/'analysis'/(n+'_host_paths.json') for a,n in names.items()}
        if not all(p.exists() for p in hosts.values()):continue
        fig,axes=plt.subplots(2,1,figsize=(13,6));used=[]
        for ax,(a,path) in zip(axes,hosts.items()):
            host=json.loads(path.read_text());paths=host['request_paths'];origin=min(r['boundaries_monotonic'][0] for r in paths)
            cohorts=host['cohorts'][6:14]
            left=min(c['D_host_submit'] for c in cohorts)-.4;right=max(c['last_client_notice'] for c in cohorts)+.15
            for r in paths:
                p,q=r['boundaries_monotonic'][1:3]
                if p<right and q>left:ax.broken_barh([(p-origin,q-p)],(1.1,.45),facecolors='#56b4e9')
            for c in cohorts:
                start,finish=c['D_host_submit'],c['last_client_notice']
                ax.broken_barh([(start-origin,finish-start)],(.1,.45),facecolors=STYLE[a][0])
                ax.text(start-origin,.67,','.join(c['requests']),fontsize=7,rotation=30)
            ax.set_xlim(left-origin,right-origin);ax.set_ylim(0,2);ax.set_yticks((.33,1.33),('D host cohort guard','P submit→ready'))
            ax.set_title(a+' | independent run '+names[a],fontsize=8);ax.set_xlabel('Seconds since this run first arrival');ax.grid(axis='x',alpha=.2)
            used.append(names[a])
        fig.suptitle(f'TRACE_ANALYSIS | Last{m} exact-row pair | O16/q6/prep1/Decode2 | host boundaries only')
        fig.text(.01,.01,'Host guards include transfer, Forward, Decode, scheduling and delivery. They are not kernel traces or physical GPU utilization. Separate panels do not share an execution.',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.94))
        save(fig,f'last{m}_representative_host_timeline','TRACE_ANALYSIS',
            'Separate actual-run timelines. Cohort guard explains legal admission timing, not GPU idle.',used)
    write(ROOT/'analysis/figure_manifest.json',manifest)
    print('EXTENDED_SWEETSPOT_FIGURES',len(manifest['figures']),flush=True)


if __name__=='__main__':
    main()
