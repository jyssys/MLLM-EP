"""Plots keep HF quality, native Libra prefill and EP request timing separate."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);a=p.parse_args()
r=a.result;out=r/'plots/resume_20260908';out.mkdir(parents=True,exist_ok=True)
policies=['sere_s2_r05_official','sere_s4_r05_official','sere_s2_r07_official','modes_official_target70','modes_official_target85']
labels=['SERE S2/rho .5','SERE S4/rho .5','SERE S2/rho .7','MoDES 70','MoDES 85']
sources=[('B1','official_confirmatory_b1_20260908'),('B16','sere_official_chart128_b16_20260908'),('B16','sere_official_gqa128_b16_20260908')]
quality=[]
for batch,source in sources:
    for row in json.loads((r/'analysis'/source/'summary.json').read_text()):quality.append(dict(row,batch=batch))
q=pd.DataFrame(quality)
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
for ax,dataset in zip(axes,['chartqa','gqa']):
    for batch,color,shift in [('B1','#b24927',-.11),('B16','#247188',.11)]:
        for index,policy in enumerate(policies):
            g=q[(q.dataset==dataset)&(q.policy==policy)&(q.batch==batch)]
            if not len(g):continue
            x=g.iloc[0];lo,hi=x.paired_delta_95ci_pp
            ax.errorbar(index+shift,x.paired_delta_pp,yerr=[[x.paired_delta_pp-lo],[hi-x.paired_delta_pp]],
                fmt='o',color=color,capsize=4,label=batch if index==0 else None)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(5),labels,rotation=30,ha='right')
    ax.set_title(dataset.upper()+' | 128 matched requests');ax.legend();ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Quality change vs same-batch vanilla (percentage points)')
fig.suptitle('Official calibration: HF quality only, image-cluster 95% CI\nNot EP latency; HF fixed cohort is not continuous-batching membership',fontsize=11)
fig.tight_layout();fig.savefig(out/'official_quality_controls.png',dpi=170);plt.close(fig)

ep=[]
for folder in sorted((r/'analysis').glob('ep_clean_renderer_*_20260908')):
    source=r/'online'/folder.name
    if not (folder/'summary.json').exists() or (source/'INVALIDATED.json').exists():continue
    args=json.loads((source/'completed.json').read_text())['arguments']
    for row in json.loads((folder/'summary.json').read_text()):
        ep.append(dict(row,batch=int(args['batch_per_dp']),run=folder.name))
if ep:
    e=pd.DataFrame(ep);e.to_csv(out/'clean_ep_summary.csv',index=False)
    clean=e[~e.fixed_output]
    # The main Pareto uses the prespecified run0 SERE/run1 corrected MoDES
    # screen. Additional independent engines have their own plot below.
    clean=clean[~clean.run.str.contains('_run[23]_')]
    corrected={(row.dataset,row.batch) for _,row in clean.iterrows() if '_modes_cached_' in row.run}
    clean=clean[[not (row.policy.startswith('modes') and (row.dataset,row.batch) in corrected
                     and '_modes_cached_' not in row.run) for _,row in clean.iterrows()]]
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    markers={1:'o',4:'s',16:'^'};colors=dict(zip(policies,['#ab3b35','#dc9075','#845075','#357996','#579652']))
    for ax,dataset in zip(axes,['chartqa','gqa']):
        for _,row in clean[(clean.dataset==dataset)&(clean.policy!='vanilla')].iterrows():
            base_policy=row.policy.removesuffix('_metadata_cached')
            ax.scatter(row.paired_quality_delta_pp,row.paired_e2e_reduction_median_percent,
                marker=markers[row.batch],color=colors[base_policy],s=80,
                label=f'{labels[policies.index(base_policy)]}, B{row.batch}'+(' (cache-fixed)' if '_metadata_cached' in row.policy else ''))
        ax.axhline(0,color='black',lw=.8);ax.axvline(0,color='black',lw=.8)
        ax.set_xlabel('Paired task-quality change (pp)');ax.set_title(dataset.upper());ax.grid(alpha=.2)
        if len(clean[clean.dataset==dataset]):ax.legend(fontsize=6,loc='best')
    axes[0].set_ylabel('Paired median request E2E reduction (%)')
    fig.suptitle('Clean EP4 natural-EOS serving; same requests, cold images\nPositive upward is faster; no quality-match claim from point estimates alone',fontsize=11)
    fig.tight_layout();fig.savefig(out/'clean_ep_quality_efficiency.png',dpi=170);plt.close(fig)
libra_source=r/'analysis/libra_eight_groups_20260908/summary.json'
if libra_source.exists():
    aggregate=json.loads(libra_source.read_text())
    if not aggregate['partial']:
        groups=aggregate['group_results'];fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for i,g in enumerate(groups):
            detail=json.loads((r/'analysis'/f"libra_native_vl48_{g['group']}_frozen_control_20260908"/'summary.json').read_text())
            oracle=next(x for x in detail if x['policy']=='libra_frozen_oracle')
            value=oracle['oracle_rank_critical_gain_percent'];lo,hi=oracle['oracle_rank_critical_gain_95ci_percent']
            axes[0].errorbar(i,value,yerr=[[value-lo],[hi-value]],fmt='o',capsize=3,color='#357996')
            axes[1].scatter(i,g['vanilla_critical_p50_ms'],color='#777777',label='Native vanilla' if i==0 else None)
            axes[1].scatter(i,g['libra_critical_p50_ms'],color='#b24927',label='Native Libra' if i==0 else None)
        for ax in axes:
            ax.set_xticks(range(len(groups)),[g['group'] for g in groups],rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
        axes[0].axhline(0,color='black',lw=.8);axes[0].axhline(10,color='gray',ls='--',lw=.8)
        axes[0].set_ylabel('Perfect prediction gain, same current routes (%)')
        axes[1].set_ylabel('Rank-critical post-vision prefill (ms)');axes[1].legend()
        fig.suptitle('Supplied Libra runtime: 32 real VL inputs, 8 groups × 10 paired executions\nNot full request E2E; no reproduction of original 8-H200 model scale',fontsize=11)
        fig.tight_layout();fig.savefig(out/'libra_prediction_oracle.png',dpi=170);plt.close(fig)
restart=r/'analysis/modes_b16_independent_20260908/runs.csv'
if restart.exists():
    d=pd.read_csv(restart);fig,ax=plt.subplots(figsize=(8,4.5))
    for policy,color,shift in [('modes_official_target70_metadata_cached','#357996',-.08),
                               ('modes_official_target85_metadata_cached','#579652',.08)]:
        g=d[d.policy==policy];v=g.paired_e2e_reduction_percent
        ax.errorbar(g.run+shift,v,yerr=[v-g.within_engine_ci_low,g.within_engine_ci_high-v],
            fmt='o',capsize=4,color=color,label='target70' if '70_' in policy else 'target85')
    ax.axhline(0,color='black',lw=.8);ax.axhline(5,color='gray',ls='--',lw=.8)
    ax.set(xticks=[1,2,3],xlabel='Independent engine restart',ylabel='Paired request E2E reduction (%)',
           title='GQA B16: initial positive challenged across three engines\nBars are within-engine cohort95%CI, not cross-engine CI')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'modes_independent_restarts.png',dpi=170);plt.close(fig)
print(json.dumps(dict(plots=str(out),completed_ep_condition_policy_rows=len(ep))))
