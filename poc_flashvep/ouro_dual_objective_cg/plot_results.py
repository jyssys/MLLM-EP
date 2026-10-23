"""Editable figures with explicit evidence scopes. No policy selection."""
import csv
import gzip
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write

DEST=ROOT/'figures'
COLORS={'O6_J':'#cc3311','O6_G':'#0077bb','O6_G1024':'#33bbee','O6_C':'#009988',
    'BEST_FIXED':'#ee7733','MIXED_STOCK':'#662288','MIXED_FAIR':'#aa3377','ATOMIC':'#777777'}

def records(name):
    p=ROOT/name
    if not p.exists():return []
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f))

def number(row,key):
    try:return float(row[key])
    except (KeyError,ValueError,TypeError):return np.nan

def save(fig,name,scope):
    fig.text(.01,.005,scope,fontsize=8,ha='left',va='bottom',wrap=True)
    fig.tight_layout(rect=(0,.045,1,.96))
    for ext in ('png','svg','pdf'):fig.savefig(DEST/f'{name}.{ext}',dpi=150)
    plt.close(fig)

def main():
    DEST.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,4))
    ax.barh(['Captured selector calls','Continuation, C frozen','Fresh boundaries','Single clipped span','Multiple feasible C','U2 differs from fixed C'],
        [660,480,180,104,76,0],color=['#777777','#999999','#0077bb','#33bbee','#009988','#cc3311'])
    ax.invert_yaxis();ax.set(xlabel='Captured decision count',title='A. Historical opportunity funnel')
    for i,v in enumerate([660,480,180,104,76,0]):ax.text(v+4,i,str(v),va='center')
    save(fig,'A_decision_funnel','TRACE REPLAY: 144 request pairs. Rejected admission polls are not exhaustively logged. Branch counts are not additive stages.')

    gradients=records('objective_gradients.csv');interactions=records('objective_interactions.csv')
    ps=sorted({int(r['P']) for r in gradients});bs=sorted({float(r['B_ms']) for r in gradients})
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(axis,field,label) in zip(axes.flat,[('C','delta_E','C effect on E'),('C','delta_R','C effect on R'),
        ('I','interaction_E','C×G interaction on E'),('G','delta_E','G effect on E'),('G','delta_R','G effect on R'),('I','interaction_R','C×G interaction on R')]):
        data=interactions if axis=='I' else [r for r in gradients if r['axis']==axis]
        matrix=np.full((len(ps),len(bs)),np.nan)
        for i,p in enumerate(ps):
            for j,b in enumerate(bs):
                vals=[number(r,field) for r in data if int(r['P'])==p and float(r['B_ms'])==b]
                if vals:matrix[i,j]=np.mean(vals)
        finite=matrix[np.isfinite(matrix)];bound=max(.001,float(np.max(np.abs(finite)))) if len(finite) else 1
        view=ax.imshow(np.ma.masked_invalid(matrix),cmap='coolwarm',vmin=-bound,vmax=bound,aspect='auto')
        ax.set(xticks=range(len(bs)),xticklabels=[f'{v:g}' for v in bs],yticks=range(len(ps)),yticklabels=ps,
            xlabel='Budget (ms)',ylabel='Historical prompt tokens',title=label)
        for i in range(len(ps)):
            for j in range(len(bs)):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center')
        fig.colorbar(view,ax=ax,shrink=.7)
    save(fig,'B_feature_cross_effects','OFFLINE ESTIMATE: normalized frozen TRAIN feature differences. Missing cells have no matched corners. Not measured alternative latency.')

    grid=records('objective_grid.csv');families=[f'O{i}' for i in range(14)]
    fig,axes=plt.subplots(1,3,figsize=(18,4))
    for family in families:
        values=sorted((r for r in grid if r['objective']==family),key=lambda r:float(r['weight']))
        axes[0].plot([number(r,'weight') for r in values],
            [number(r,'C_differs_E_only_fraction')*100 for r in values],marker='.',label=family)
    axes[0].set(xlabel='Risk weight λ',ylabel='C differs from E-only (%)',title='C. Offline objective response, multi-C denominator')
    shortlist=json.loads((ROOT/'objective_shortlist.json').read_text())['validation_candidates']
    axes[1].bar([r['family'] for r in shortlist],[100*r['fresh_C_change_fraction'] for r in shortlist],color='#0077bb')
    axes[1].axhline(5,c='#cc3311',ls='--',label='Registered 5% screen')
    axes[1].set(ylabel='Fresh choices different from fixed-C G-only (%)',title='Fresh VALIDATION: 4 starts per width')
    axes[0].legend(ncol=4,fontsize=8);axes[1].legend(fontsize=8)
    summary_file=ROOT/'Joint_C_change_summary.json'
    if summary_file.exists():
        summary=json.loads(summary_file.read_text())
        axes[2].bar(['≤512','513–2048','>2048'],[summary['changed_remaining_le_512'],summary['changed_remaining_513_to_2048'],summary['changed_remaining_gt_2048']],color='#cc3311')
        axes[2].set(xlabel='Remaining prompt at changed C choice (tokens)',ylabel='Actual changed-C decision count',
            title=f'FINAL: {summary["real_C_changes"]}/{summary["fresh_choices"]} fresh C changes')
    save(fig,'C_objective_action_diversity','Left: OFFLINE TRAIN geometry,132 multi-C states. Center/right: new LIVE choices. Different choices alone are not a serving gain.')

    audit_files=sorted((ROOT/'raw').glob('dual_p2_EVENT_N8_B120_*/O6_J_OBS_*.json'))
    if audit_files:
        audit=json.loads(audit_files[0].read_text());q=audit['service_observations'];origin=audit['start']
        fig,ax=plt.subplots(figsize=(12,4))
        for unit in q:
            left=unit['aligned_service_begin_s']-origin;duration=unit['service_ms']/1000
            for span in unit['spans']:
                if not span['phase'].startswith('prefill'):continue
                ax.broken_barh([(left,duration)],(int(span['id'])-.3,.6),facecolors=f'C{span["G"]}')
        ax.set(xlabel='Aligned time since cohort start (s)',ylabel='Request ID',title='F2. CALIBRATION ready-stream prefill service intervals')
        bound=max((unit['alignment_bound_ms'] for unit in q),default=0)
        save(fig,'F2_event_service_timeline',f'LIVE CUDA EVENT AUDIT: {len(q)} units. Alignment bound≤{bound:.3f}ms. Excludes unobserved decode intervals. Includes native preparation/launch gaps, not pure kernel busy.')

    all_rows=records('serving_results.csv');final=[r for r in all_rows if r['split']=='FINAL']
    quiet=[r for r in final if 'SINGLE_MEASUREMENT' in r['lane']]
    char=[r for r in final if 'CHARACTERIZATION' in r['lane']]
    if quiet:
        workloads=sorted({r['workload'] for r in quiet});fig,axes=plt.subplots(1,len(workloads),figsize=(12,4),squeeze=False)
        for ax,workload in zip(axes.flat,workloads):
            for policy in COLORS:
                rs=[r for r in quiet if r['workload']==workload and r['policy']==policy]
                if not rs:continue
                x=np.mean([number(r,'TTFT_P95_ms') for r in rs]);y=np.mean([number(r,'ITL_P99_ms') for r in rs])
                ax.scatter(x,y,color=COLORS[policy],label=policy);ax.annotate(policy,(x,y),fontsize=8,xytext=(3,3),textcoords='offset points')
            ax.set(xlabel='Mean of cohort TTFT P95 (ms)',ylabel='Mean of cohort ITL P99 (ms)',title=workload)
            ax.margins(x=.15,y=.15)
        save(fig,'D_quiet_latency_tradeoff','LIVE MEASURED: serial measurement lane, 3 starts ×12 offered requests per arm/workload. Fixed GPU0–3 background. Not pooled quantiles.')
    if char:
        workloads=['CHAT','MIXED','LONG','BURSTY','LONG8K_STRESS'];fig,axes=plt.subplots(1,5,figsize=(19,4))
        for ax,workload in zip(axes,workloads):
            for policy in ('O6_J','O6_G','O6_G1024','O6_C','BEST_FIXED','MIXED_STOCK','MIXED_FAIR'):
                rs=[r for r in char if r['workload']==workload and r['policy']==policy]
                rates=sorted({number(r,'offered_rate_rps') for r in rs})
                if not rates:continue
                ax.plot(rates,[np.mean([number(r,'joint_attainment_TTFT5s_TBT200ms_pct') for r in rs if number(r,'offered_rate_rps')==rate]) for rate in rates],
                    '.-',color=COLORS[policy],label=policy)
            ax.axhline(90,c='grey',ls=':');ax.set(xlabel='Arrival intensity parameter (req/s)',ylim=(0,103),title=workload)
        axes[0].set_ylabel('Complete every-gap joint attainment (%)');axes[-1].legend(fontsize=7,loc='lower left')
        save(fig,'E_finite_attainment_vs_rate','LIVE MEASURED: 12 requests/cohort, 3 starts, 4 workers. TTFT≤5s, every ITL≤200ms. NOT qualified capacity. Bursty uses a base intensity, not mean rate.')

        policies=['O6_J','O6_G','O6_G1024','O6_C','BEST_FIXED','MIXED_STOCK','MIXED_FAIR']
        fig,ax=plt.subplots(figsize=(10,5));matrix=np.full((5,len(policies)),np.nan)
        for i,w in enumerate(workloads):
            for j,p in enumerate(policies):
                rs=[r for r in char if r['workload']==w and r['policy']==p and number(r,'offered_rate_rps')==.9]
                if rs:matrix[i,j]=np.mean([number(r,'joint_attainment_TTFT5s_TBT200ms_pct') for r in rs])
        v=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set(xticks=range(len(policies)),xticklabels=policies,yticks=range(5),yticklabels=workloads,title='G. Locked O6 workload transfer at arrival parameter0.9')
        for i in range(5):
            for j in range(len(policies)):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]<55 else 'black')
        fig.colorbar(v,ax=ax,label='Joint attainment (%)')
        save(fig,'G_workload_transfer','LIVE MEASURED: mean finite-cohort attainment, n=3 independent starts/cell. Same frozen objective, budget, controller target. No sustainable-rate claim.')

    if quiet:
        target=next((r for r in quiet if r['policy']=='O6_J' and r['workload']=='LONG8K_STRESS'),None)
        if target:
            data=json.loads((ROOT/target['source']).read_text());fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
            t0=data['start'];spans=[]
            for row in data['dispatch_descriptors']:
                for s in row['spans']:
                    if s['phase'].startswith('prefill'):spans.append((row,s))
            # Descriptor keys are native host submission metadata, not CUDA finishes.
            for row,s in spans:
                clock=row.get('host_return_s')
                if clock is None:continue
                axes[0].scatter(clock-t0,int(s['id']),s=18+15*s['G'],c=f'C{min(9,s["G"])}',marker='s')
            for d in data['optimizer_decisions']:
                axes[1].scatter(d['host_s']-t0,d['selected']['C'],c=f'C{d["selected"]["G"]}',s=10)
            for r in data['requests']:
                axes[0].scatter(np.array(r['token_times'])-t0,[int(r['id'])+.2]*len(r['token_times']),s=1,color='black')
            axes[0].set(ylabel='Request ID (outputs offset+0.2)',title='F. Actual host submissions and delivered output tokens')
            from matplotlib.lines import Line2D
            axes[0].legend(handles=[Line2D([],[],marker='s',linestyle='',color=f'C{g}',label=f'G={g}'+(' prelude' if g==0 else '')) for g in range(5)],
                loc='lower right',ncol=5,fontsize=8)
            axes[1].set(xlabel='Time since serving-window start (s)',ylabel='Chosen actual C')
            save(fig,'F_policy_host_timeline','LIVE HOST CLOCKS: squares=prompt descriptors, size/color=actual G. Dots=delivered tokens. This is NOT a GPU-kernel overlap timeline.')

        fig,ax=plt.subplots(figsize=(8,5))
        for policy in ('O6_J','O6_G','O6_C','MIXED_FAIR'):
            selected=[r for r in quiet if r['policy']==policy and r['workload']=='LONG8K_STRESS'];gaps=[];n=0
            for r in selected:
                data=json.loads((ROOT/r['source']).read_text());n+=len(data['requests'])
                gaps.extend(1000*g for q in data['requests'] for g in q['ITLs_s'])
            if not gaps:continue
            x=np.sort(gaps);y=(len(x)-np.arange(len(x)))/len(x)
            ax.step(x,y,where='post',label=f'{policy}, {n} requests, {len(x)} gaps',color=COLORS[policy])
        ax.set(xscale='log',yscale='log',xlabel='Delivered-token ITL (ms)',ylabel='P(ITL ≥ x)',title='H. Quiet 8K stress: pooled-gap CCDF')
        ax.legend(fontsize=8)
        save(fig,'H_quiet_ITL_CCDF','LIVE MEASURED: gaps are correlated within requests. No token-independent CI. Pooled tail is not request-level every-gap attainment.')

    overhead=records('optimizer_overhead.csv');pairs=records('paired_comparisons.csv')
    if final and overhead:
        fig,ax=plt.subplots(figsize=(9,5))
        for workload in sorted({r['workload'] for r in char}):
            rs=[r for r in overhead if r['split']=='FINAL' and 'CHARACTERIZATION' in r['lane'] and r['workload']==workload and r['policy']=='O6_J' and number(r,'rate')==.9]
            p=next((r for r in pairs if r['split']=='FINAL' and 'CHARACTERIZATION' in r['lane'] and r['workload']==workload and
                number(r,'offered_rate_rps')==.9 and r['candidate']=='O6_J' and r['reference']=='O6_G' and r['metric']=='drain_goodput_TTFT5s_TBT200ms_rps'),None)
            if not rs or not p or not np.isfinite(number(p,'ratio')):continue
            x=np.mean([number(r,'selector_wall_ratio')*100 for r in rs]);y=number(p,'ratio')
            ax.scatter(x,y);ax.annotate(workload,(x,y),xytext=(4,3),textcoords='offset points')
        ax.axhline(1,c='grey',ls='--');ax.set(xlabel='Measured selector CPU total / drained wall (%)',
            ylabel='O6 Joint / G2048 finite SLO-goodput ratio',title='I. Selector cost versus observed outcome')
        save(fig,'I_overhead_vs_outcome','LIVE MEASURED: CPU time is not removable E2E share. Both axes use arrival parameter0.9. No causal overhead attribution.')
    write(DEST/'figure_manifest.json',dict(files=[p.name for p in sorted(DEST.iterdir()) if p.suffix in ('.png','.svg','.pdf')],
        scopes='A replay. B/C-left offline estimates. C-right/D/E/F/G/H/I live, with scope in each caption.'))
    print('Figure files',len(list(DEST.glob('*.svg'))))

if __name__=='__main__':main()
