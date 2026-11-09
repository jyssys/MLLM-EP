"""Draw actual observations. Do not substitute model budgets for GPU idle gaps."""
import math
from collections import Counter
from .common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from .campaign_report import event_records
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'figure.dpi':140,'savefig.bbox':'tight','axes.grid':True,'grid.alpha':.18})
    out=ROOT/'figures';out.mkdir(exist_ok=True);manifest=[]
    rows=json.loads((ROOT/'serving_runs.json').read_text())
    by={r['name']:r for r in rows}
    def save(fig,name,scope,sources):
        fig.tight_layout(rect=(0,0,1,.94),h_pad=2.5)
        for suffix in ('png','pdf'):fig.savefig(out/(name+'.'+suffix))
        plt.close(fig);manifest.append(dict(name=name,png=str((out/(name+'.png')).relative_to(REPO)),
            pdf=str((out/(name+'.pdf')).relative_to(REPO)),evidence_scope=scope,sources=sources))
    def panel_missing(ax,label):
        ax.text(.5,.5,label,ha='center',va='center',transform=ax.transAxes);ax.set_xticks([]);ax.set_yticks([])
    def selected(model):
        preferred=[r for r in rows if r['model']==model and r['complete'] and
            r.get('evidence_class')=='MARKER_FREE' and r['name'].startswith('confirm_')]
        if preferred:return preferred
        return [r for r in rows if r['model']==model and r['complete'] and
            r.get('evidence_class')=='MARKER_FREE' and r['name'].startswith('r1_') and 'selection32' in r['name']]

    # Figure 1: real work, including zero acceptance. Zero execution cannot prove slack.
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    names=[]
    for index,model in enumerate(('ouro','huginn')):
        values=[r for r in rows if r['model']==model and r['name'].startswith(f'i_{model}_tile') and r['complete']]
        values.sort(key=lambda r:int(r['name'].split('_tile')[1].split('_')[0]))
        names.extend(r['name'] for r in values)
        if values:
            x=np.arange(len(values));labels=[r['name'].split('_tile')[1].split('_')[0] for r in values]
            axes[index,0].bar(x,[r['offloaded_requests'] for r in values],color='#248b91')
            axes[index,0].set_ylim(0,max(1,1.2*max(r['offloaded_requests'] for r in values)))
            if max(r['offloaded_requests'] for r in values)<=1:axes[index,0].set_yticks([0,1])
            axes[index,0].set_xticks(x,labels);axes[index,0].set_ylabel('Actual offloaded requests / 24')
            axes[index,0].set_title(model.title()+': initial admission intervention')
            axes[index,0].set_xlabel('Configured execution tile (tokens)')
            axes[index,1].plot(x,[r['ITL_p99_ms'] for r in values],'o-',color='#446baa')
            axes[index,1].axhline(65.0828764215 if model=='ouro' else 192.821253818,color='#bd4937',ls='--',label='Frozen ITL p99 SLO')
            axes[index,1].set_xticks(x,labels);axes[index,1].set_ylabel('Observed delivery ITL p99 (ms)');axes[index,1].legend(fontsize=8)
        else:panel_missing(axes[index,0],'NOT_MEASURED');panel_missing(axes[index,1],'NOT_MEASURED')
    fig.suptitle('1. Initial real-intervention attempts: acceptance is measured, not assumed')
    save(fig,'figure_1_controlled_intervention','Marker-free selection runs. Zero acceptance provides no positive evidence of safe added work.',names)

    # Figure 2: actual D admission decisions, not forced mixed cuts.
    decisions=json.loads((ROOT/'actual_decisions.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4));names=[]
    for ax,model in zip(axes,('ouro','huginn')):
        chosen=selected(model)
        preferred=[r for r in chosen if r['method'] in ('dynamic_loop','dynamic_tile')]
        if not preferred:
            fallback=by.get(f'r1_{model}_tile_selection32')
            preferred=[fallback] if fallback else []
        ns={r['name'] for r in preferred};names.extend(ns)
        ds=[r for r in decisions if r['name'] in ns]
        for kind,color,label in (('rejected','#aaaaaa','Rejected'),('prepared','#d6a34a','Prepare only'),('committed','#168d83','Committed offload')):
            subset=[r for r in ds if ('committed' if r.get('source_committed') else 'prepared' if r['accepted'] else 'rejected')==kind]
            ax.scatter([r.get('future_quantum_budget_ms',r['budget_ms']) for r in subset],
                [r['remaining'] for r in subset],s=32 if kind=='committed' else 12,alpha=.65,c=color,label=label)
        ax.set_title(model.title());ax.set_xlabel('Predicted admission quantum budget (ms)')
        ax.set_xscale('symlog',linthresh=8)
        ax.set_ylabel('Remaining loops offered to D');ax.legend(fontsize=8)
        ax.text(.02,.02,'Pooled starts do not establish same-run Cut diversity',transform=ax.transAxes,fontsize=7)
    fig.suptitle('2. Actual online decisions: forecast budget is not physical idle time')
    save(fig,'figure_2_budget_and_actual_loop_count','Actual fresh boundary negotiation. Prepare acceptance can still expire before commit. Points can pool starts. Same-run adaptive diversity requires the separate per-run gate.',names)

    fig,axes=plt.subplots(1,2,figsize=(12,4));names=[]
    for ax,model in zip(axes,('ouro','huginn')):
        name='r10_ouro_actual_mixed_state32' if model=='ouro' else 'r9_huginn_actual_mixed_state32'
        names.append(name)
        subset=[r for r in decisions if r['name']==name]
        for committed,color,label in ((False,'#bbbbbb','No source commit'),(True,'#258e91','Actual source commit')):
            group=[r for r in subset if bool(r.get('source_committed'))==committed]
            ax.scatter([r.get('future_quantum_budget_ms',r['budget_ms']) for r in group],
                [r['remaining'] for r in group],s=42 if committed else 10,c=color,alpha=.7,label=label)
        ax.set_xscale('symlog',linthresh=8)
        ax.set_xlabel('Actual admission forecast (ms)')
        ax.set_ylabel('Remaining Loops offered')
        ax.set_title(model.title()+': one actual mixed-cut N32 state run')
        ax.legend(fontsize=8)
    fig.suptitle('2b. Full-byte correctness diagnostics — NOT clean Decode-SLO evidence')
    save(fig,'figure_2b_actual_mixed_state_decisions','Actual variable online cuts within each named N32 correctness stress. Observers alter timing and diagnostic feedback. These runs do not establish clean serving benefit or Decode safety.',names)

    # Figure 3: honest host-protocol timeline. No made-up kernel activity.
    fig,axes=plt.subplots(2,1,figsize=(12,7));names=[]
    for ax,model in zip(axes,('ouro','huginn')):
        values=[r for r in selected(model) if r.get('offloaded_requests',0)>0 and r['method'] in ('dynamic_tile','dynamic_loop')]
        name=values[0]['name'] if values else f'r1_{model}_tile_selection32'
        folder=ROOT/'raw'/name
        if not (folder/'measurement_begin.json').exists():panel_missing(ax,'NOT_MEASURED');continue
        begin=json.loads((folder/'measurement_begin.json').read_text())['time'];events=event_records(folder,begin)
        names.append(name)
        for y,kind,color in ((3,'GPU_COMPLETED_BOUNDARY_OFFER','#3d69a5'),(2,'SOURCE_COMMIT','#a766b0'),
                (1,'D_QUANTUM_SUBMITTED','#1f9a8e'),(0,'DECODE_INTERVAL_SAMPLE','#b98b2b')):
            ts=[r['time']-begin for r in events if r['event']==kind]
            ax.scatter(ts,[y]*len(ts),marker='|',s=70,c=color)
        queue=[r for r in events if r['event']=='SUPPLY_SAMPLE' and r.get('role')=='prefill']
        if queue:
            right=ax.twinx();right.plot([r['time']-begin for r in queue],
                [r.get('executable_P_ready',r.get('ready_P_leases',0)) for r in queue],
                color='#637381',alpha=.3,lw=.8)
            right.set_ylabel('P executable-ready requests (host)',fontsize=8)
        for item in [r for r in events if r['event']=='SOURCE_COMMIT'][:12]:
            ax.annotate(str(item['remaining'])+' loops',(item['time']-begin,2),
                xytext=(3,5),textcoords='offset points',fontsize=7,color='#78478c')
        ax.set_yticks([0,1,2,3],['Decode output receipt','D quantum submission','P cut commit','GPU-complete offer receipt'])
        ax.set_xlabel('Seconds after measurement begins');ax.set_title(model.title()+' — '+name,fontsize=9)
    fig.suptitle('3. Measured host protocol timeline — NOT a GPU kernel-idle trace')
    save(fig,'figure_3_online_protocol_timeline','Host timestamps and GPU-event completion observations. Physical kernel union and yield gaps are NOT_MEASURED.',names)

    # Figure 4: separate serving metrics. Never sum component quantiles.
    fig,axes=plt.subplots(2,4,figsize=(17,7));names=[]
    for index,model in enumerate(('ouro','huginn')):
        values=selected(model);groups={}
        for row in values:groups.setdefault(row['method'],[]).append(row)
        methods=list(groups);names.extend(r['name'] for r in values)
        for ax,key,title in zip(axes[index],('TTFT_p95_ms','ITL_p99_ms','E2E_p95_ms','SLO_goodput'),
                ('TTFT p95 (ms)','ITL p99 (ms)','E2E p95 (ms)','Joint SLO-goodput (requests/s)')):
            if not groups:panel_missing(ax,'NOT_MEASURED');continue
            vals=[np.mean([r[key] for r in groups[m]]) for m in methods]
            ax.bar(range(len(methods)),vals,color=['#637381' if m in ('native','zero') else '#258e91' for m in methods])
            for x,m in enumerate(methods):ax.scatter([x]*len(groups[m]),[r[key] for r in groups[m]],s=13,c='black',zorder=5)
            ax.set_xticks(range(len(methods)),[m.replace('dynamic_','D-')+f' (n={len(groups[m])})' for m in methods],rotation=40,ha='right',fontsize=8)
            ax.set_title(model.title()+': '+title,fontsize=9)
            if key=='ITL_p99_ms':ax.axhline(65.0828764215 if model=='ouro' else 192.821253818,c='#bd4937',ls='--')
    fig.suptitle('4. Actual marker-free serving — bars average per-start metrics; dots show individual starts')
    save(fig,'figure_4_actual_serving_comparison','Actual selected or held-out measurements. Bars average the displayed per-start metrics, not pooled percentiles. Finite cohorts are not sustainable capacity. Service TTFT is NOT_MEASURED.',names)

    paired_file=ROOT/'paired_start_ratios.json'
    if paired_file.exists():
        paired=json.loads(paired_file.read_text())
        fig,axes=plt.subplots(1,3,figsize=(14,4));names=[]
        keys=[('ouro','dynamic_tile'),('ouro','dynamic_token'),('huginn','dynamic_tile'),('huginn','dynamic_token')]
        labels=[model.title()+(' Loop' if method=='dynamic_tile' else ' Token') for model,method in keys]
        for ax,metric,title in zip(axes,('TTFT_p95_ms','ITL_p99_ms','SLO_goodput'),
                ('TTFT p95 ratio — lower is better','ITL p99 ratio — lower is better','Goodput ratio — higher is better')):
            for y,(model,method) in enumerate(keys):
                found=[r for r in paired if r['model']==model and r['method']==method and r['reference']=='native'
                    and r['metric']==metric and r['eligible_complete_three_start_comparison']]
                if not found:continue
                r=found[0];low,high=r['bootstrap_95_CI'];value=r['geometric_ratio']
                ax.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='o',capsize=3,
                    color='#278e91' if method=='dynamic_tile' else '#467eac')
                names.extend(s['candidate'] for s in r['sources'])
                names.extend(s['reference'] for s in r['sources'])
            ax.axvline(1.,c='#666666',ls='--',lw=1)
            if metric=='ITL_p99_ms':ax.axvline(1.05,c='#bd4937',ls=':',lw=1,label='Native-relative target')
            if metric=='SLO_goodput':ax.set_xscale('log')
            ax.set_yticks(range(len(keys)),labels);ax.set_title(title,fontsize=10)
            ax.set_xlabel('Candidate / paired Original Native')
        fig.suptitle('4b. Paired independent starts: geometric ratios and limited three-start 95% bootstrap intervals')
        save(fig,'figure_4b_paired_native_effects','Three paired N128 starts per method. Intervals resample starts, not tokens. All cohorts remain included. Finite goodput is not sustainable capacity.',sorted(set(names)))

    # Figure 5: structural rows and actual payload, not invented P compute savings.
    fig,axes=plt.subplots(2,3,figsize=(14,7));names=[]
    for index,model in enumerate(('ouro','huginn')):
        values=selected(model);names.extend(r['name'] for r in values)
        if not values:
            for ax in axes[index]:panel_missing(ax,'NOT_MEASURED')
            continue
        groups={}
        for r in values:groups.setdefault(r['method'],[]).append(r)
        methods=list(groups);x=np.arange(len(methods));labels=[m.replace('dynamic_','D-') for m in methods]
        fractions={m:[100*r.get('actual_D_prompt_recurrent_rows',0)/(r['count']*(4096*4 if model=='ouro' else 2048*32)) for r in groups[m]] for m in methods}
        payloads={m:[(r.get('measured_KV_payload_bytes',0)+r.get('measured_hidden_payload_bytes',0))/2**30 for r in groups[m]] for m in methods}
        axes[index,0].bar(x,[np.mean(fractions[m]) for m in methods],color='#258e91')
        axes[index,0].set_ylabel('Recurrent rows moved (%) — not FLOPs')
        axes[index,1].bar(x,[np.mean(payloads[m]) for m in methods],color='#866ca4')
        for j,m in enumerate(methods):
            axes[index,0].scatter([j]*len(fractions[m]),fractions[m],s=12,c='black',zorder=5)
            axes[index,1].scatter([j]*len(payloads[m]),payloads[m],s=12,c='black',zorder=5)
        axes[index,1].set_ylabel('Measured NIXL payload (GiB)')
        panel_missing(axes[index,2],'P critical-path compute saving\nNOT_MEASURED\nNo causal speedup from row counts')
        for ax in axes[index,:2]:ax.set_xticks(x,labels,rotation=40,ha='right',fontsize=8);ax.set_title(model.title())
    fig.suptitle('5. Offloading accounting: structural work and measured communication remain separate')
    save(fig,'figure_5_offloading_accounting','Exact active recurrent rows and successful native write bytes. Exclusive P saved time and D kernel union are NOT_MEASURED.',names)

    # Figure 6: calibration is explicitly diagnostic, with terminal and core costs.
    fig,axes=plt.subplots(2,2,figsize=(12,8));names=[]
    costs=json.loads((ROOT/'new_primitive_costs.json').read_text())
    actual_quanta=json.loads((ROOT/'actual_quanta.json').read_text())
    for column,model in enumerate(('ouro','huginn')):
        ax=axes[0,column]
        points=[]
        for value in costs:
            if value['model']!=model:continue
            names.append(value['name'])
            for g in value['groups']:
                if g['role']=='decode' and g['loops']>0:
                    points.append(g)
        for terminal,color,label in ((False,'#4278a8','Nonterminal core'),(True,'#c76b42','Terminal core + boundary work')):
            gs=[g for g in points if g['terminal']==terminal]
            ax.scatter([g['tokens'] for g in gs],[g['p99_ms'] for g in gs],c=color,s=[16+8*g['loops'] for g in gs],alpha=.65,label=label)
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Tokens per measured submission')
        ax.set_ylabel('CUDA Event enclosure p99 (ms)');ax.set_title(model.title());ax.legend(fontsize=8)
        ax.text(.02,.02,'Point area indicates grouped Loop count',transform=ax.transAxes,fontsize=7)
        ax=axes[1,column]
        candidates={r['name']:r for r in selected(model) if r['method'] in ('dynamic_loop','dynamic_tile')}
        cells=Counter()
        for row in actual_quanta:
            if row['name'] in candidates and row['event']=='D_QUANTUM_SUBMITTED':
                cells[(row['rows'],row['loops'],bool(candidates[row['name']].get('Decode_SLO_screen')))]+=1
        if cells:
            for (tokens,loops,passed),count in cells.items():
                ax.scatter([tokens],[loops],s=35+8*count,c='#278e91' if passed else '#bd4937',alpha=.7)
                ax.annotate(str(count),(tokens,loops),xytext=(4,4),textcoords='offset points',fontsize=7)
            ax.set_xscale('log',base=2)
            ax.set_xlabel('Actual submitted token rows')
            ax.set_ylabel('Actual grouped recurrent loops')
            actual_loops=sorted({loops for tokens,loops,passed in cells})
            ax.set_yticks(actual_loops)
            ax.set_ylim(min(actual_loops)-.4,max(actual_loops)+.4)
            ax.set_title(model.title()+': actual Loop execution; labels count submissions',fontsize=9)
            ax.text(.01,.02,'Color: whole-run Decode gate, not per-tile safety',transform=ax.transAxes,fontsize=7)
        else:panel_missing(ax,'Actual Loop execution geometry\nNOT_MEASURED')
        names.extend(candidates)
    fig.suptitle('6. Serial cost calibration and actual execution geometry remain separate')
    save(fig,'figure_6_full_prompt_and_tiled_costs','Top: serial CUDA Event enclosures, not certified WCET or safe windows. Bottom: actual submitted tiles. Color records the parent cohort Decode gate, not per-tile deadline safety.',sorted(set(names)))

    # Figure 7: actual fallback and reasons. No hypothetical adaptive counts.
    fig,axes=plt.subplots(2,2,figsize=(13,7));names=[]
    for index,model in enumerate(('ouro','huginn')):
        values=[r for r in selected(model) if r['method'] in ('dynamic_loop','dynamic_tile','dynamic_token')]
        names.extend(r['name'] for r in values)
        if not values:
            panel_missing(axes[index,0],'NOT_MEASURED');panel_missing(axes[index,1],'NOT_MEASURED');continue
        groups={}
        for r in values:groups.setdefault(r['method'],[]).append(r)
        methods=list(groups);x=np.arange(len(methods));labels=[m.replace('dynamic_','D-') for m in methods]
        total={m:sum(r['count'] for r in groups[m]) for m in methods}
        off={m:sum(r['offloaded_requests'] for r in groups[m]) for m in methods}
        fallback=[100*(total[m]-off[m])/total[m] for m in methods]
        axes[index,0].bar(x,fallback,color='#b6bdc4',label='Native fallback')
        bottom=np.array(fallback)
        counts=sorted({int(k) for r in values for k in r.get('accepted_loop_counts',{})})
        for loops in counts:
            amount=[100*sum(r.get('accepted_loop_counts',{}).get(str(loops),0) for r in groups[m])/total[m] for m in methods]
            axes[index,0].bar(x,amount,bottom=bottom,label=f'{loops} loops')
            bottom+=amount
        amount=[100*off[m]/total[m] if m=='dynamic_token' else 0. for m in methods]
        if any(amount):axes[index,0].bar(x,amount,bottom=bottom,color='#278e91',label='Token suffix')
        axes[index,0].set_ylim(0,100);axes[index,0].set_ylabel('Requests (%)');axes[index,0].set_xticks(x,labels,rotation=25)
        axes[index,0].set_title(model.title());axes[index,0].legend(fontsize=8)
        reasons=Counter()
        for r in values:reasons.update(r.get('rejection_reasons',{}))
        labels,counts=zip(*reasons.most_common(5)) if reasons else ([],[])
        axes[index,1].barh(range(len(counts)),counts,color='#778ba3')
        axes[index,1].set_yticks(range(len(counts)),[s.replace('_',' ').lower() for s in labels],fontsize=8)
        axes[index,1].set_xlabel('Rejected offers');axes[index,1].set_title(model.title()+': recorded rejection reasons')
    fig.suptitle('7. Observed online decisions and Native fallback')
    save(fig,'figure_7_actual_decision_statistics','Actual commits, completed D work, fallback, and recorded refusal reasons. Counts pool the displayed starts and do not establish within-run diversity. Unrun arms are omitted, not zero-filled.',names)

    matched=ROOT/'matched_rows_primitive_comparison.json'
    if matched.exists():
        values=json.loads(matched.read_text());fig,axes=plt.subplots(1,2,figsize=(12,4));names=[]
        for ax,model in zip(axes,('ouro','huginn')):
            for family,color,label in (('token','#4782b2','Strong Token candidates'),('loop','#d17b45','Whole-prompt Loop')):
                group=[r for r in values if r['model']==model and r['family']==family]
                names.extend(r['name'] for r in group)
                ax.scatter([r['largest_group_p99_ms'] for r in group],
                    [r['D_total_enclosure_p50_ms'] for r in group],c=color,s=42,label=label)
            ax.set_xlabel('Largest submission enclosure p99 (ms)')
            ax.set_ylabel('Total D serial enclosure p50 (ms)');ax.set_title(model.title());ax.legend(fontsize=8)
        fig.suptitle('Supplement: efficiency vs packing granularity — equal recurrent rows, NOT equal FLOPs')
        save(fig,'figure_8_matched_rows_efficiency_pareto','Serial disjoint-eight CUDA Event calibration. Token geometry is not restricted to C256. This is not an online speedup or safe-slack proof.',sorted(set(names)))
    write(ROOT/'figure_manifest.json',dict(figures=manifest,created=time.time(),
        limitations=['Physical kernel idle is NOT_MEASURED.','Primary Service TTFT is NOT_MEASURED.',
            'Structural row movement is not measured FLOPs or P critical-path compute saving.']))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
