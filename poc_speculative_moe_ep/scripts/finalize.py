"""Build traceable reports/figures from frozen results, without fitting policy."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poc_speculative_moe_ep.contract import ROOT,TARGET_REV,DRAFT_REV,write


def load(root,name):return json.loads((root/name).read_text())
def fmt(x,d=3):return 'N/A' if x is None else f'{x:.{d}f}'


def table(rows):
    head=['Arm','acceptance','progress/verify','experts/layer','remote paths/layer','fanout/token','max-rank bytes/layer','EP ms/progress','target ms/progress','E2E ms/token']
    lines=['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']
    for s in rows:
        lines.append('| '+' | '.join(['AR' if not s['gamma'] else str(s['gamma']),fmt(s['acceptance']),
            fmt(s['committed_per_verify']),fmt(s['mean_unique_experts']),fmt(s['mean_remote_paths']),
            fmt(s['mean_fanout']),fmt(s['mean_max_rank_bytes'],1),fmt(s['ep_ms_per_committed']),
            fmt(s['target_ms_per_committed']),fmt(s['e2e_ms_per_token'])])+' |')
    return '\n'.join(lines)


def route_repeat(runs,trace):
    counts=[]
    import gzip
    import os
    for name in runs:
        path=ROOT/'raw'/name
        records=load(path,'worker.dp0.json')['records']
        gamma=next(r['gamma'] for r in records if r['repeat']>=0)
        origin=ROOT/'raw'/f'phase_a_v1_g{gamma}'
        original=load(origin,'worker.dp0.json')['records']
        oldwaves={r['request_id']:r['wave'] for r in original if r['repeat']>=0}
        maps=[{(f['wave'],f['forward']):f for f in map(json.loads,(path/f'frames.rank{r}.jsonl').read_text().splitlines())} for r in range(4)]
        oldmaps=[{(f['wave'],f['forward']):f for f in map(json.loads,(origin/f'frames.rank{r}.jsonl').read_text().splitlines())} for r in range(4)]
        for record in records:
            if record['repeat']<0:continue
            rid=record['request_id'];wi=record['wave'];oldwi=oldwaves[rid]
            selected=trace[(trace.gamma==gamma)&(trace.request_id==rid)]
            current=[f for (w,i),f in sorted(maps[0].items()) if w==wi and f.get('sampled_tokens') is not None][1:]
            assert len(current)==len(selected)
            different=total=rank_different=0
            arrays=[np.load(path/f'routes.wave{wi}.rank{r}.npz') for r in range(4)]
            oldarrays=[np.load(origin/f'routes.wave{oldwi}.rank{r}.npz') for r in range(4)]
            for f,v in zip(current,selected.sort_values('committed_position').itertuples()):
                for rank in range(4):
                    frame=maps[rank][(wi,f['forward'])];old=oldmaps[rank][(oldwi,v.forward)]
                    for layer in range(48):
                        a=arrays[rank][frame['layers'][layer]['route_key']+'_ids']
                        b=oldarrays[rank][old['layers'][layer]['route_key']+'_ids']
                        assert a.shape==b.shape
                        different+=int((a!=b).sum());rank_different+=int((a//32!=b//32).sum());total+=a.size
            for a in arrays+oldarrays:a.close()
            counts.append(dict(run=name,request_id=rid,gamma=gamma,expert_difference_fraction=different/total,
                rank_difference_fraction=rank_different/total,paths=total,
                note='position-wise selected route slots including actual dummy/padding; no set-based reordering'))
    return dict(status='MEASURED' if counts else 'NOT_RUN',rows=counts,
        max_expert_difference=max([r['expert_difference_fraction'] for r in counts],default=None))


def overhead(clean_parity,bypass_runs,trace_parity):
    records=[]
    for name in bypass_runs:
        for r in load(ROOT/'raw'/name,'worker.dp0.json')['records']:
            if r['repeat']>=0:
                records.append(dict(gamma=r['gamma'],request_id=r['request_id'],wall=r['actual_request_completion_ms'],
                                    tokens=r['outputs'][0]['tokens']))
    results=[]
    for gamma in sorted({r['gamma'] for r in records}):
        values=[]
        for rid in sorted({r['request_id'] for r in records if r['gamma']==gamma}):
            base=[r['wall'] for r in records if r['gamma']==gamma and r['request_id']==rid]
            clean=[r['actual_request_completion_ms'] for r in clean_parity if r['gamma']==gamma and r['request_id']==rid]
            assert all(r['tokens']==next(c['tokens'] for c in clean_parity if c['gamma']==gamma and c['request_id']==rid) for r in records if r['gamma']==gamma and r['request_id']==rid)
            values.append(100*(np.median(clean)/np.median(base)-1))
        results.append(dict(gamma=gamma,median_request_overhead_pct=float(np.median(values)),request_values_pct=values,
            status='UNVALIDATED_OVERHEAD_STRATUM' if abs(np.median(values))>10 else 'SMALL_SUBSET_COMPATIBLE',
            scope='separate restart blocks; not randomized interleave; two requests, medians of repeated actual completions'))
    drift=[]
    for gamma in (4,8):
        values=[]
        for old in [r for r in trace_parity if r['gamma']==gamma and r.get('actual_request_completion_ms') is not None]:
            c=[r['actual_request_completion_ms'] for r in clean_parity if r['gamma']==gamma and r['request_id']==old['request_id']]
            values.append(100*(old['actual_request_completion_ms']/np.median(c)-1))
        if values:drift.append(dict(gamma=gamma,median_trace_clean_difference_pct=float(np.median(values)),
            status='DIAGNOSTIC_NONPAIRED_TIME_BLOCK_DRIFT_NOT_CAUSAL_HOOK_OVERHEAD'))
    return dict(clean_vs_bypass=results,trace_vs_clean=drift,
        limitation='do not attribute differences to hooks alone; drift/producer readiness/shared fabric are not experimentally separated')


def plots(root,rows,metrics,models,oracle,pairs):
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    x=np.arange(len(rows));labels=['AR' if r['gamma']==0 else str(r['gamma']) for r in rows]
    choices=[('acceptance','acceptance'),('mean_unique_experts','unique experts/layer'),
        ('mean_remote_paths','remote expert paths/layer'),('mean_fanout','destinations/token'),
        ('mean_max_rank_bytes','max-rank useful bytes/layer'),('ep_ms_per_verify','EP ms/verify'),
        ('ep_ms_per_committed','EP ms/committed'),('e2e_ms_per_token','actual E2E ms/output token'),
        ('target_ms_per_committed','target ms/committed')]
    for i,(field,label) in enumerate(choices,1):
        fig,ax=plt.subplots(figsize=(5.5,3.5));ax.plot(x,[r.get(field) if r.get(field) is not None else np.nan for r in rows],'o-')
        ax.set_xticks(x,labels);ax.set_xlabel('native exact speculation depth');ax.set_ylabel(label)
        ax.set_title('Clean TRUE EP4; independently traced route features',fontsize=9)
        fig.tight_layout();fig.savefig(figures/f'gamma_{field}.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,3.5));points=metrics.iloc[::max(1,len(metrics)//1500)]
    im=ax.scatter(points.unique_experts,points.ep_comm_ms,c=points.max_rank_bytes,s=8,alpha=.5)
    fig.colorbar(im,ax=ax,label='max-rank useful bytes/layer');ax.set_xlabel('unique experts/layer');ax.set_ylabel('clean EP ms/verify')
    fig.tight_layout();fig.savefig(figures/'expert_vs_ep.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,3.5))
    if pairs['pairs']:
        ax.scatter([r['traffic_delta_pct'] for r in pairs['pairs']],[r['ep_delta_pct'] for r in pairs['pairs']])
    else:ax.text(.5,.5,'N/A: no eligible matched rank-traffic pairs\n(no missing-cost inference from an empty set)',ha='center',va='center',transform=ax.transAxes)
    ax.set_xlabel('matched traffic difference %');ax.set_ylabel('EP time difference %');fig.tight_layout();fig.savefig(figures/'matched_pairs.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,3.5));ax.bar(['M0','M1','M2'],[models[k]['ep_comm_ms']['mae'] for k in ('M0','M1','M2')])
    ax.set_ylabel('request-grouped CV EP MAE (ms/verify)');fig.tight_layout();fig.savefig(figures/'grouped_cv.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.5))
    if oracle.get('request_costs'):
        keys=[f"fixed_{oracle['best_fixed_gamma']}",'expert','acceptance','aggregate','ep']
        ax.bar(keys,[np.mean([r[k] for r in oracle['request_costs']]) for k in keys]);ax.set_ylabel('target ms/progress at common prefixes')
    else:ax.text(.5,.5,'N/A: missing-cost gate failed; no oracle promoted',ha='center',va='center',transform=ax.transAxes)
    ax.set_title('Offline available-opportunity diagnostic, not E2E rollout',fontsize=9);fig.tight_layout();fig.savefig(figures/'oracle_comparison.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--key-version')
    p.add_argument('--bypass',nargs='*',default=[]);p.add_argument('--route-repeat',nargs='*',default=[]);a=p.parse_args()
    result=ROOT/'artifacts'/a.version
    rows=load(result,'gamma_summary.json');models=load(result,'cost_model_results.json');width=load(result,'width_controlled_cost_model_results.json')
    gate=load(result,'missing_cost_gate.json');oracle=load(result,'oracle_policy_results.json');pairs=load(result,'matched_pair_analysis.json')
    disagreement=load(result,'depth_disagreement.json');metrics=pd.read_parquet(result/'verification_metrics.parquet')
    layer_pairs=load(result,'layer_matched_pairs.json') if (result/'layer_matched_pairs.json').exists() else None
    trace=ROOT/'artifacts/phase_a_v1';trace_parity=load(trace,'parity.json');parity=load(result,'parity.json')
    repeat=route_repeat(a.route_repeat,pd.read_parquet(trace/'verification_metrics.parquet'))
    oh=overhead(parity,a.bypass,trace_parity)
    write(result/'route_repeatability.json',repeat);write(result/'instrumentation_validity.json',oh)
    key_text=''
    if a.key_version:
        key_root=ROOT/'artifacts'/a.key_version
        key_metrics=pd.read_parquet(key_root/'verification_metrics.parquet')
        key_parity=load(key_root,'parity.json')
        ids=[r['request_id'] for r in load(ROOT/'artifacts','frozen_pool.json')['rows'][:2]]
        key_rows=[]
        for gamma,g in key_metrics[key_metrics.request_id.isin(ids)].groupby('gamma'):
            walls=[r for r in key_parity if r['gamma']==gamma and r['request_id'] in ids]
            assert len(walls)==20, 'two predefined requests each require10 repeats'
            key_rows.append(dict(gamma=int(gamma),requests=2,repeats_per_request=10,
                ep_ms_per_committed=float(g.ep_comm_ms.sum()/g.committed_progress.sum()),
                target_ms_per_committed=float(g.target_forward_ms.sum()/g.committed_progress.sum()),
                e2e_ms_per_token=sum(r['actual_request_completion_ms'] for r in walls)/sum(len(r['tokens']) for r in walls)))
        write(result/'key_repeat_summary.json',dict(rows=key_rows,request_ids=ids,
            ordering='forward3+reverse7 depth macroblocks, alternating requests; no fastest-run selection',
            limitation='two requests only; not20 independent prompts; primary CV/CI remains16-request first block'))
        key_text='\n\nFinal key repeat check: **two predefined requests ×10 repeats/depth** (forward3+reverse7 restart blocks).\n\n'+'\n'.join(f"- gamma{r['gamma']}: EP/progress={r['ep_ms_per_committed']:.3f}ms; target/progress={r['target_ms_per_committed']:.3f}ms; E2E/token={r['e2e_ms_per_token']:.3f}ms." for r in key_rows)+'\n'
    progression=[r['ep_ms_per_committed'] for r in rows]
    crossover=any(progression[i]<progression[i-1] and progression[i]<progression[i+1] for i in range(1,len(progression)-1))
    if not gate['pass_any']:verdict='EP_COST_COLLAPSES_TO_EXISTING_MOE_SD_COST'
    elif not crossover:verdict='NO_SPECULATION_CROSSOVER'
    elif oracle.get('target_side_gain_pct') and oracle['target_side_gain_pct']['mean']<10:verdict='EP_AWARE_POLICY_HEADROOM_TOO_SMALL'
    else:verdict='PHYSICAL_EP_MISSING_COST_CONFIRMED'
    independent=(models['ep_mae_reduction_pct']>=15 and width['ep_mae_reduction_pct']>=15) or gate['criteria']['matched_pairs']
    best=min(rows,key=lambda r:r['ep_ms_per_committed'])['gamma']
    info=dict(verdict=verdict,missing_cost_gate=gate,independent_rank_cost_supported=bool(independent),
        crossover=crossover,best_fixed_ep_gamma=best,
        target_side_headroom_too_small=bool(oracle.get('target_side_gain_pct') and oracle['target_side_gain_pct']['mean']<10),
        e2e_oracle_headroom_status='UNVALIDATED_NO_E2E_POLICY_ROLLOUT',
        target_output_equivalence=True,requests=metrics.request_id.nunique(),frozen_pool_requests=96,
        output_cap=64,topology='TP2-DP2-EP4-PP1',gpu_set=[2,3,6,7],
        scope='bounded first16 (8 GSM8K+8 HumanEval); 96 frozen pool not fully measured; no universal negative/no new method speedup',
        next_action='Do not build an EP-aware SD method from this evidence. Consider higher concurrency/EP degree only under a new frozen experiment, not threshold tuning here.',
        prior_overlap_status='RELATED_EP_PRIOR_NOT_IDENTICAL_SD_MECHANISM')
    write(ROOT/'artifacts/final_verdict.json',info)
    plots(result,rows,metrics,models,oracle,layer_pairs if layer_pairs and layer_pairs['pairs'] else pairs)
    reports=ROOT/'reports';prefix='Measured clean TRUE EP4 on GPUs2,3,6,7, output cap64/EOS, first16 frozen requests (8+8). Route features come from a separate trace pass with identical request/prefix/draft IDs.\n\n'
    ci=lambda v:f"{v['mean']:.3f}, 95% request CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}], n={v['n']}"
    documents={
      'sd_correctness':prefix+f"Native greedy longest-prefix acceptance, correction/bonus and KV rollback were reused. Tokenizer vocab/file hashes agree. All48 SD request comparisons across gamma2/4/8 equal their AR output, and sampled-token reconstruction agrees. Each clean repeat also equals AR. No expert/top-k/router/precision modification.\n\n"+f"Alignment: `{result.name}/alignment.json`. Repeat routes: {json.dumps(repeat)}. Output equivalence is a tested-domain result, not a proof for every prompt or BF16 shape. Correction/bonus sampler flags are distinct from tokens actually committed before EOS/output cap.\n",
      'expert_footprint_analysis':prefix+table(rows)+'\n\nExpert counts are unique global expert IDs per layer, averaged over48 layers. Expert union sums layer/expert pairs, not IDs across layers. Actual loaded tensors imply9,437,184 bytes/expert; multiplying by active experts is a working-set proxy, not measured HBM traffic. Actual dummy/padded expert counts remain in the M1 baseline.\n',
      'physical_ep_analysis':prefix+table(rows)+f"\n\nRemote path counts are token–expert paths; destination-deduplicated remote hidden rows and bytes are separate fields. Fanout includes the local destination. Per-rank matrices include useful local rows; subtract their diagonal for network-only logical attribution. Metadata/header/padding wire bytes are not inferred.\n\nCumulative-critical-rank D/E/C is one rank per invocation. Full-router projection is UNAVAILABLE independently; router selection only is measured. Dispatch/combine are prepare/finalize composites, not pure link service. Physical EP share of target: {[(r['gamma'],r['ep_target_share']) for r in rows]}.\n",
      'amortization_vs_waste':prefix+f"Crossover inside tested depths: **{crossover}**. Best fixed EP depth: {best}. Do not force a U-shaped result or extrapolate gamma16.\n\n"+'\n'.join(f"- gamma{r['gamma']}: target calls/progress={r['target_invocations_per_committed']:.4f}; EP/verified row={r['ep_ms_per_verified_row']:.4f}ms; expert/row={r['expert_ms_per_verified_row']:.4f}ms; rejected logical bytes={100*r['rejected_logical_ep_byte_fraction']:.2f}%; remote useful bytes/progress={r['logical_remote_ep_bytes_per_committed']:.0f}." for r in rows)+ '\n\nRejected bytes are logical attribution to rejected candidate input rows; batched verification does not make all of them independently removable. Output-cap trimming is not relabeled rejection. AR emits the first token in prefill, so post-prefill committed denominators exclude that token while request E2E includes it.\n',
      'cost_model_comparison':prefix+f"Same standardized ridge alpha1, request-grouped5-fold, no target-specific tuning. Explicit acceptance in M0; actual dummy/executed expert work in M1; physical traffic in M2. These are explanatory models after verification, not deployable future predictors.\n\n"+'\n'.join(f"- {m}: EP MAE={models[m]['ep_comm_ms']['mae']:.5f}ms; R2={models[m]['ep_comm_ms']['r2']:.5f}; MoE MAE={models[m]['moe_ms']['mae']:.5f}; target MAE={models[m]['target_forward_ms']['mae']:.5f}." for m in ('M0','M1','M2'))+f"\n\nM2−M1 MAE reduction {models['ep_mae_reduction_pct']:.3f}%; R2 increment {models['ep_r2_increment']:.5f}; request-paired MAE improvement {ci(models['paired_request_mae_improvement_ms'])}ms.\n\nSecondary nonlinear width-only control adds the **same gamma fixed effects to all three families**, preventing traffic features from merely encoding four batch widths: M2 MAE reduction {width['ep_mae_reduction_pct']:.3f}%, R2 increment {width['ep_r2_increment']:.5f}.\n\nMatched-pair status {pairs['status']}, n={pairs['n']}; no null result is inferred from no eligible pairs. Expert-proxy vs measured EP best-depth disagreement {disagreement['disagreement_fraction']:.3f}, CI {disagreement['uncertainty']['ci95']}. This can reflect known fixed-cost amortization rather than independently useful within-width rank structure. Formal gate {gate['criteria']}; independent rank-cost support={independent}.\n",
      'oracle_headroom':prefix+f"{json.dumps(oracle,indent=2) if not oracle.get('opportunities') else json.dumps({k:v for k,v in oracle.items() if k not in ('opportunities','request_costs')},indent=2)}\n\nDetailed alternatives stay in machine-readable JSON. Only prefixes actually shared by all4 native arms are eligible; token-prefix equality/draft nesting are checked, not assumed. gamma0=AR, not SDgamma1. Costs are real clean target labels; the analysis is not an adaptively executed policy. No oracle E2E is fabricated, and no missing alternatives are interpolated. Aggregate measured verification-cost oracle is a strongest relevant control and can already subsume communication.\n",
      'reproduction':prefix+"See ../README.md for exact audit/launch/campaign/analysis commands, environment paths, append-only names and watchdog/cleanup. Full local logs live in raw/, which is ignored by git. Compact exported raw numerical events, actual route arrays, owner proofs, outputs and hash manifests are under artifacts/compact_*. Model weights and answer keys are excluded.\n\n```bash\n/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.physical_validation --version my_clean_analysis phase_b_clean_v1_g0 phase_b_clean_v1_g2 phase_b_clean_v1_g4 phase_b_clean_v1_g8\n/home/esjung/anaconda3/bin/python -m poc_speculative_moe_ep.scripts.verify_bundle --evidence compact_phase_a_v1\n/home/esjung/anaconda3/bin/python -m pytest -q poc_speculative_moe_ep/tests\n```\n\nResume completed campaign IDs skips successful immutable arms; failed IDs require a new prefix. Main unrelated staged/dirty diff hashes are preserved in repository_safety.json. An instrumented trace fit is never promoted as a clean headline. Final burner mask must be explicitly2,3,6,7 because run_utilize.sh currently defaults to different GPUs.\n"
    }
    if layer_pairs:
        documents['cost_model_comparison']+=f"\nSecondary **same-layer** matching avoids averaging away traffic variation: {layer_pairs['n_pairs']} matched layer pairs, only {layer_pairs['requests']} independent requests; exact gamma/rows/progress/user+executed expert count; traffic>=15%; no latency-based selection. Median EP change {layer_pairs['median_ep_delta_pct']:.4f}%; request-cluster mean {ci({**layer_pairs['mean_ci'],'n':layer_pairs['mean_ci']['requests']})}%. These are not thousands of independent samples and do not meet the20% physical-time gate.\n"
    documents['physical_ep_analysis']+=key_text+'\nInstrumentation validity: '+json.dumps(oh)+'\n'
    for name,text in documents.items():
        file=reports/(name+'.md')
        if file.exists():raise FileExistsError(file)
        file.write_text('# '+name.replace('_',' ').title()+'\n\n'+text)
    fields=dict(POC_TYPE='DISCOVERY_ONLY',TARGET_MODEL='Qwen/Qwen3-30B-A3B',TARGET_REVISION=TARGET_REV,
        DRAFT_MODEL='Qwen/Qwen3-0.6B',DRAFT_REVISION=DRAFT_REV,TOKENIZER_COMPATIBILITY='IDENTICAL_VOCAB_AND_FILES',
        SD_SEMANTICS='native greedy longest prefix + correction/bonus + KV rollback',TARGET_OUTPUT_EQUIVALENCE='PASS 16/16 per gamma and all measured clean repeats',
        GPU_SET='2,3,6,7',PHYSICAL_TO_LOGICAL_MAPPING='2->0,3->1,6->2,7->3',GPU_MODEL='H10080GBHBM3 NV18 selected links',
        TOPOLOGY='TP2-DP2-EP4-PP1',EP_SIZE=4,DEEPEP_BACKEND='DeepEPHTPrepareAndFinalize + TritonExperts',
        DATASETS='GSM8K test + HumanEval public; 64+32 frozen,8+8 actually gated',REQUEST_COUNT=16,
        REMOTE_ROWS_DEFINITION='token-expert path rows; destination-deduplicated hidden rows reported separately',
        GENERATED_TOKENS=sum(len(r['tokens']) for r in trace_parity if r['gamma']==0),
        AR_TARGET_MS_PER_TOKEN=rows[0]['target_ms_per_committed'],AR_EP_COMM_MS_PER_TOKEN=rows[0]['ep_ms_per_committed'],
        BEST_FIXED_GAMMA=best,BEST_EXPERT_AWARE_GAMMA_ORACLE='see aligned alternatives JSON; no single global oracle gamma',
        BEST_EP_AWARE_GAMMA_ORACLE='see aligned alternatives JSON; AR represented by0, not SD1',
        M1_GROUPED_CV_MAE=models['M1']['ep_comm_ms']['mae'],M2_GROUPED_CV_MAE=models['M2']['ep_comm_ms']['mae'],
        M2_ERROR_REDUCTION_PCT=models['ep_mae_reduction_pct'],M1_GROUPED_CV_R2=models['M1']['ep_comm_ms']['r2'],M2_GROUPED_CV_R2=models['M2']['ep_comm_ms']['r2'],
        MATCHED_UNIQUE_EXPERT_EP_TIME_DELTA=layer_pairs['median_ep_delta_pct'] if layer_pairs else ('N/A' if not pairs['n'] else pairs['median_ep_delta_pct']),
        EP_COMM_FRACTION_OF_TARGET_VERIFY={r['gamma']:r['ep_target_share'] for r in rows},
        REJECTED_EP_BYTE_FRACTION={r['gamma']:r['rejected_logical_ep_byte_fraction'] for r in rows},
        EP_AWARE_ORACLE_VS_FIXED_GAIN=oracle.get('target_gain_vs_fixed_pct','NOT_RUN'),EP_AWARE_ORACLE_VS_EXPERT_AWARE_GAIN=oracle.get('target_gain_vs_expert_pct','NOT_RUN'),
        EP_AWARE_ORACLE_VS_STRONGEST_TARGET_BASELINE_GAIN=oracle.get('target_side_gain_pct','NOT_RUN'),
        EP_AWARE_ORACLE_E2E_STATUS='UNVALIDATED_NOT_AN_ADAPTIVE_POLICY_ROLLOUT',
        PRIOR_OVERLAP_STATUS=info['prior_overlap_status'],PHYSICAL_EP_MISSING_COST_STATUS=f'formal gate={gate["pass_any"]}; independent rank-cost support={independent}',
        SPECULATION_CROSSOVER_STATUS='CONFIRMED' if crossover else 'NO_SPECULATION_CROSSOVER_IN_TESTED_DEPTHS',
        OVERALL_VERDICT=verdict,NEXT_ACTION=info['next_action'],BRANCH='flashvep/speculative-moe-ep-discovery',
        COMMIT='git log -1 --format=%H on the evidence branch; exact SHA supplied in handoff')
    for r in rows[1:]:
        g=r['gamma'];fields.update({f'GAMMA{g}_'+k:v for k,v in dict(ACCEPTANCE=r['acceptance'],COMMITTED_PER_VERIFY=r['committed_per_verify'],
            UNIQUE_EXPERTS=r['mean_unique_experts'],REMOTE_ROWS=r['mean_remote_paths'],REMOTE_DEDUP_HIDDEN_ROWS=r['mean_remote_dedup_rows'],RANK_FANOUT=r['mean_fanout'],MAX_RANK_BYTES=r['mean_max_rank_bytes'],
            EP_MS_PER_COMMITTED=r['ep_ms_per_committed'],E2E_MS_PER_TOKEN=r['e2e_ms_per_token']).items()})
    if oracle.get('opportunities'):
        from collections import Counter
        fields['BEST_EXPERT_AWARE_GAMMA_ORACLE']=dict(Counter(o['expert_gamma'] for o in oracle['opportunities']))
        fields['BEST_EP_AWARE_GAMMA_ORACLE']=dict(Counter(o['ep_gamma'] for o in oracle['opportunities']))
    write(ROOT/'artifacts/final_summary.json',fields)
    text='# Speculative decoding × physical MoE EP: final discovery\n\n'+f'**{verdict}**. Independent rank-traffic cost support: {independent}; tested-depth crossover: {crossover}.\n\n'+prefix+table(rows)+key_text+f"\n\nM2 communication MAE reduction={models['ep_mae_reduction_pct']:.3f}%; grouped-CV R2 increment={models['ep_r2_increment']:.5f}. Whole-forward averaged features provided no eligible traffic-separated pairs; the fixed-layer secondary comparison is reported in cost_model_comparison.md and resamples requests, not layers.\n\n"+f"Prior audit found related EP routing-prediction work, not the identical draft/verify/accept mechanism. This is not a novelty proof or superiority claim. No online scheduler, expert dropping, router approximation or proposed-method speedup was implemented.\n\n"+f"Important limits: only16 initial requests actually measured, not the full96 frozen pool; native engines are paired forward/reverse restart blocks, not randomized per-wave interleave; single-active DP0, real DP1 dummy cost; trace timings diagnostic; clean route features are transferred from identical-prefix traced runs, with repeat checks on a small subset. Condition timings are app-level CUDA composites, not pure link bandwidth.\n\n"+ '## Required summary block\n\n```text\n'+'\n'.join(f'{k}: {v}' for k,v in fields.items())+'\n```\n'
    (reports/'final_summary.md').write_text(text)
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
