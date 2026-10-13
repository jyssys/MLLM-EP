"""Render only observed results; bootstrap problem units, not layers."""
import collections
import argparse
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from poc_multibd_moe_ep.contract import *
from poc_dllm_ep_comm.metrics import bootstrap_mean


def mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def f(value,digits=3):
    return 'NOT_RUN' if value is None else f'{value:.{digits}f}'


def ci(statistic):
    if statistic.get('mean') is None:return 'NOT_RUN'
    low,high=statistic['ci95']
    return f"{f(statistic['mean'])} [{f(low)}, {f(high)}]; n={statistic['n']}"


def scientific_lane_status(root,session,validity):
    """An optional control failure never validates it or erases a valid primary.

    Raw/global failure flags are retained. Promotion is strictly lane-specific;
    TP comparisons require a COMPLETE, parity-valid supported control.
    """
    cases={}
    for backend in ['ep4','tp4']:
        for r in [1,2,3,4]:
            directory=root/f'artifacts/R{r}_{backend}_held_out_paired_{session}'
            status=directory/'status.json';process=directory/'process_status.json'
            cases[f'{backend}_R{r}']=bool(status.exists() and process.exists() and
                json.loads(status.read_text()).get('success') and
                json.loads(process.read_text()).get('exit_code')==0)
    return dict(cases=cases,
        primary_ep_complete=all(cases[f'ep4_R{r}'] for r in [1,2,3,4]),
        optional_tp_valid=all(cases[f'tp4_R{r}'] for r in [1,2,3,4]) and
            validity.get('tp4_held_out_pass_all',False))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True)
    args=parser.parse_args();results=ROOT/'results'/args.session
    data=json.loads((results/'conversion_summary.json').read_text())
    validity=json.loads((results/'validity.json').read_text())
    physical=json.loads((results/'physical_ep4_validation.json').read_text())
    lanes=scientific_lane_status(ROOT,args.session,validity)
    quality_path=results/'quality_evalplus.json'
    quality=json.loads(quality_path.read_text())['rows'] if quality_path.exists() else []
    observations=[r for r in data['requests'] if r['split']=='held_out' and
                  (r['backend']=='ep4' or lanes['optional_tp_valid'])]
    paired=[r for r in data['paired'] if r['split']=='held_out' and
            (r['backend']=='ep4' or lanes['optional_tp_valid'])]
    aggregate={}
    metrics=['TPF','step_ms','TPS','wall_ms','nfe','useful_committed',
             'timing_D_ms','timing_E_ms','timing_C_ms','timing_EP_ms_per_commit',
             'timing_D_ms_per_forward','timing_E_ms_per_forward','timing_C_ms_per_forward',
             'timing_EP_ms_per_forward','trace_D_C_bytes_per_commit',
             'trace_active_expert_union','trace_tokens_per_active_expert',
             'trace_max_rank_active_experts','trace_max_rank_active_weight_bytes_proxy',
             'trace_edge_density','trace_destination_fanout','trace_rank_max_mean',
             'trace_rank_cv','trace_max_rank_load','timing_observer_wall_ratio']
    metrics+=['maximum_actual_active_blocks','mean_actual_active_blocks','mean_dummy_row_fraction']
    quality_summary={}
    for backend in ['ep4','tp4']:
        aggregate[backend]={}
        for r in [1,2,3,4]:
            rows=[x for x in observations if x['backend']==backend and x['R']==r]
            values={key:mean(rows,key) for key in metrics};values['requests']=len(rows)
            values['PCE']=mean([x for x in paired if x['backend']==backend and x['R']==r],'PCE')
            values['TPF_gain']=mean([x for x in paired if x['backend']==backend and x['R']==r],'TPF_gain')
            values['TPS_gain']=mean([x for x in paired if x['backend']==backend and x['R']==r],'TPS_gain')
            aggregate[backend][r]=values
            q=[x for x in quality if x['split']=='held_out' and x['backend']==backend and x['R']==r
               and (backend=='ep4' or lanes['optional_tp_valid'])]
            scored=[x for x in q if x['evaluable']]
            quality_summary[backend+str(r)]=dict(n=len(q),evaluated=len(scored),
                pass_at_1=mean(scored,'passed'),unresolved=[x['task_id'] for x in q if not x['evaluable']],
                domains={domain:dict(n=len([x for x in scored if x['domain']==domain]),
                    pass_at_1=mean([x for x in scored if x['domain']==domain],'passed'))
                    for domain in ['humaneval_plus','mbpp_plus']})
    quality_deltas={}
    for r in [2,3,4]:
        base={(x['domain'],x['selected_index']):x for x in quality if x['split']=='held_out'
              and x['backend']=='ep4' and x['R']==1 and x.get('evaluable')}
        current={(x['domain'],x['selected_index']):x for x in quality if x['split']=='held_out'
                 and x['backend']=='ep4' and x['R']==r and x.get('evaluable')}
        quality_deltas[str(r)]=bootstrap_mean([float(current[k]['passed'])-float(base[k]['passed'])
                                             for k in base.keys()&current.keys()])
    observer={}
    for r in [1,2,3,4]:
        values=[x['timing_observer_wall_ratio']-1 for x in observations
                if x['backend']=='ep4' and x['R']==r and 'timing_observer_wall_ratio' in x]
        stat=bootstrap_mean(values)
        positive=max(0,stat['mean'] or 0)
        # Negative apparent observer cost is not "negative instrumentation";
        # retain signed drift and require a small upper uncertainty bound.
        upper=stat['ci95'][1]
        stat['status']=('NOT_RUN' if not values else
            'UNTRUSTED' if positive>.10 or (upper is not None and upper>.10) else 'VALIDATED_BOUNDED')
        observer[str(r)]=stat
    tp_comparison={}
    for r in [2,3,4]:
        ep={x['request_id']:x['PCE'] for x in paired if x['R']==r and x['backend']=='ep4'}
        tp={x['request_id']:x['PCE'] for x in paired if x['R']==r and x['backend']=='tp4'}
        tp_comparison[str(r)]=bootstrap_mean([tp[k]-ep[k] for k in ep.keys()&tp.keys()])
    gates={}
    delta_diagnostics={}
    for r in [2,3,4]:
        a=aggregate['ep4'][r]
        pairs=[x for x in paired if x['backend']=='ep4' and x['R']==r]
        ep_rows=[x for x in observations if x['backend']=='ep4' and x['R']==r]
        timing_ok=observer[str(r)]['status']=='VALIDATED_BOUNDED' and observer['1']['status']=='VALIDATED_BOUNDED'
        stage_share=mean([dict(share=x['timing_EP_ms_per_forward']/
                                   (x['step_ms']*x['timing_observer_wall_ratio']))
                          for x in ep_rows if x.get('timing_EP_ms_per_forward') is not None],'share')
        increments=bootstrap_mean([x['EP_incremental_share'] for x in pairs
                                  if x.get('EP_incremental_share') is not None])
        beyond_tp=tp_comparison[str(r)]
        time_ratio=mean(pairs,'timing_EP_ms_per_commit_ratio')
        byte_ratio=mean(pairs,'trace_D_C_bytes_per_commit_ratio')
        quality_stat=quality_deltas[str(r)]
        positive=[x for x in pairs if x['Delta_step_ms']>0 and x.get('Delta_EP_ms') is not None]
        delta_diagnostics[str(r)]=dict(
            n=len(pairs),positive_added_step_n=len(positive),
            nonpositive_added_step_n=len(pairs)-len(positive),
            mean_delta_step_ms=mean(pairs,'Delta_step_ms'),
            mean_delta_EP_ms=mean(pairs,'Delta_EP_ms'),
            positive_subgroup_ratio_of_delta_sums=(
                sum(x['Delta_EP_ms'] for x in positive)/sum(x['Delta_step_ms'] for x in positive)
                if positive else None),
            purpose='secondary denominator-stability diagnostic; does not replace the frozen mean-of-ratios gate')
        gates[str(r)]=dict(
            algorithmic_TPF_gain_ge_1_30=a['TPF_gain'] is not None and a['TPF_gain']>=1.30,
            PCE_le_0_75=a['PCE'] is not None and a['PCE']<=.75,
            event_attribution_valid=timing_ok,
            EP_incremental_share=increments,
            EP_incremental_share_ge_0_50=timing_ok and increments['mean'] is not None and increments['mean']>=.50,
            TP_minus_EP_PCE_ge_0_15=beyond_tp['mean'] is not None and beyond_tp['mean']>=.15 and beyond_tp['ci95'][0]>0,
            EP_ms_per_commit_ratio=time_ratio,
            D_C_bytes_per_commit_ratio=byte_ratio,
            cost_per_commit_worsens=(timing_ok and time_ratio is not None and time_ratio>=1.20) or
                                   (byte_ratio is not None and byte_ratio>=1.25),
            routed_EP_share_of_step=stage_share if timing_ok else None,
            stage_share_denominator='event-only observed step, reconstructed from clean step * observer wall ratio',
            EP_share_ge_0_30=timing_ok and stage_share is not None and stage_share>=.30,
            quality_delta=quality_stat,
            quality_point_within_2pp=quality_stat['mean'] is not None and quality_stat['mean']>=-.02,
            quality_noninferiority_proven=False,
            individual_low_PCE_fraction=mean([dict(low=x['PCE']<=.75) for x in pairs],'low'))
    case_status=lanes['cases']
    complete=lanes['primary_ep_complete'] and all(aggregate['ep4'][r]['requests']==24 for r in [1,2,3,4])
    full_sweep_complete=complete and lanes['optional_tp_valid']
    quality_complete=all(quality_summary['ep4'+str(r)]['evaluated']==24 for r in [1,2,3,4])
    supported=[r for r,g in gates.items() if g['algorithmic_TPF_gain_ge_1_30'] and g['PCE_le_0_75'] and
               (g['EP_incremental_share_ge_0_50'] or g['TP_minus_EP_PCE_ge_0_15']) and
               g['cost_per_commit_worsens'] and g['quality_point_within_2pp']]
    generic_loss=[r for r,g in gates.items() if g['PCE_le_0_75'] and
                  tp_comparison[r]['mean'] is not None and abs(tp_comparison[r]['mean'])<.15]
    if physical['status']!='PASS' or not validity['ep4_held_out_pass_all']:
        verdict='RUNTIME_BLOCKED'
    elif not complete or not quality_complete:
        verdict='CONDITIONAL'
    elif supported:
        # Small benchmark CIs do not prove 1-2pp quality noninferiority.
        verdict='CONDITIONAL'
    elif generic_loss:
        verdict='GENERIC_MULTIBD_ONLY'
    else:
        verdict='NO-GO'
    final=dict(verdict=verdict,measurement_session=args.session,gpu_mask=MASK,
        gates=gates,full_sweep_complete=full_sweep_complete,primary_ep_sweep_complete=complete,
        optional_tp_control_valid=lanes['optional_tp_valid'],
        completed_cases=case_status,quality_complete=quality_complete,
        physically_supported_Rs=supported,
        next_action=('Do not implement an EP-aware controller; close this fixed-R direction.' if verdict=='NO-GO' else
                     'Validate the observed limitation before any method implementation.'),
        no_adaptive_method_implemented=True)
    write(results/'final_verdict.json',final)
    write(results/'headline_summary.json',dict(aggregate=aggregate,
        quality=quality_summary,quality_paired_deltas=quality_deltas,
        event_observer=observer,tp_minus_ep_PCE=tp_comparison,
        incremental_delta_diagnostics=delta_diagnostics,
        final_verdict=final,
        exact_stage_boundaries='routed sparse D/E/C; shared expert and full-source/router TP restoration are outside',
        warm_prefix_cache=True,generated_code_sandbox='bubblewrap + resource limits'))

    figures=ROOT/'figures'/args.session;figures.mkdir(parents=True,exist_ok=True)
    def plot_metrics(filename,keys,labels):
        fig,axes=plt.subplots(1,len(keys),figsize=(5*len(keys),3.7),squeeze=False)
        for ax,key,label in zip(axes[0],keys,labels):
            for backend in ['ep4','tp4']:
                vals=[aggregate[backend][r].get(key) for r in [1,2,3,4]]
                if all(v is None for v in vals):continue
                ax.plot([1,2,3,4],vals,'o-',label=backend)
            ax.set(xlabel='Block Buffer capacity R',ylabel=label,xticks=[1,2,3,4]);ax.grid(alpha=.3);ax.legend()
        fig.tight_layout();fig.savefig(figures/filename,dpi=160);plt.close(fig)
    plot_metrics('tpf_tps_step_vs_running_set.png',['TPF','step_ms','TPS'],
        ['Useful output tokens / forward','Clean warm-generation ms / forward','Useful output tokens / second'])
    plot_metrics('conversion_efficiency.png',['PCE'],['Paired-request PCE (TPS gain / TPF gain)'])
    plot_metrics('ep_cost_per_commit.png',['timing_EP_ms_per_commit','trace_D_C_bytes_per_commit'],
        ['Event-only routed EP ms / final output token','Deduplicated remote logical D+C bytes / token'])
    plot_metrics('expert_workingset.png',['trace_active_expert_union','trace_tokens_per_active_expert'],
        ['Mean active expert union / sparse layer','Assignments / active expert'])
    plot_metrics('rank_pressure.png',['trace_edge_density','trace_rank_max_mean','trace_rank_cv'],
        ['Source-destination edge density','Assignment rank max / mean','Assignment rank CV'])
    plot_metrics('buffer_utilization.png',['mean_actual_active_blocks','mean_dummy_row_fraction'],
        ['Actual active blocks before native forward','Physical dummy-row fraction (decode forwards)'])
    fig,ax=plt.subplots(figsize=(7,4));bottom=np.zeros(4)
    for key,label in [('timing_D_ms_per_forward','Dispatch'),('timing_E_ms_per_forward','Owner expert'),
                      ('timing_C_ms_per_forward','Combine')]:
        vals=np.array([aggregate['ep4'][r][key] or 0 for r in [1,2,3,4]],dtype=float)
        ax.bar([1,2,3,4],vals,bottom=bottom,label=label);bottom+=vals
    ax.set(xlabel='R',ylabel='Event-only cumulative-critical rank ms / forward',
           title='Sparse routed path; observer validity reported separately');ax.legend()
    fig.tight_layout();fig.savefig(figures/'latency_breakdown.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for backend in ['ep4','tp4']:
        for r in [1,2,3,4]:
            q=quality_summary[backend+str(r)]['pass_at_1'];a=aggregate[backend][r]
            if q is not None and a['TPS'] is not None:
                ax.scatter(a['TPS'],100*q,label=f'{backend} R{r}');ax.annotate(f'R{r}',(a['TPS'],100*q))
    ax.set(xlabel='Clean useful TPS',ylabel='Full EvalPlus pass@1 (%)',title='Frozen held-out coding subset')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'quality_vs_parallelism.png',dpi=160);plt.close(fig)

    table=['| R | N | TPF | Step ms | TPS | PCE | EP ms/commit | D+C bytes/commit | Active experts | Rank max/mean | Quality |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r,a in aggregate['ep4'].items():
        q=quality_summary['ep4'+str(r)]
        table.append('| '+' | '.join([str(r),str(a['requests'])]+[f(a[k]) for k in
            ['TPF','step_ms','TPS','PCE','timing_EP_ms_per_commit','trace_D_C_bytes_per_commit',
             'trace_active_expert_union','trace_rank_max_mean']]+[f(q['pass_at_1'])])+' |')
    paired_table=['| R | TPF gain | TPS gain | PCE | EP ms/commit ratio | Logical D+C bytes/commit ratio |',
                  '|---|---|---|---|---|---|']
    tp_table=['| R | TPF | Step ms | TPS | PCE | Plus pass@1 |',
              '|---|---:|---:|---:|---:|---:|']
    observer_table=['| R | Event-only wall overhead; 95% CI | Status |',
                    '|---|---|---|']
    state_table=['| R | Mean real active blocks | Mean dummy-row fraction | Dispatch ms/forward | Expert ms/forward | Combine ms/forward |',
                 '|---|---:|---:|---:|---:|---:|']
    for r in [1,2,3,4]:
        stats=data['summary']['held_out_ep4'][str(r)]
        paired_table.append('| '+str(r)+' | '+' | '.join(ci(stats[key]) for key in
            ['TPF_gain','TPS_gain','PCE','timing_EP_ms_per_commit_ratio','trace_D_C_bytes_per_commit_ratio'])+' |')
        tp_label=('UNVALIDATED' if (ROOT/f'artifacts/R{r}_tp4_held_out_paired_{args.session}').exists()
                  else 'NOT_RUN')
        tp_values=([f(aggregate['tp4'][r][key]) for key in ['TPF','step_ms','TPS','PCE']]+
                   [f(quality_summary['tp4'+str(r)]['pass_at_1'])] if lanes['optional_tp_valid'] else
                   [tp_label]*5)
        tp_table.append('| '+str(r)+' | '+' | '.join(tp_values)+' |')
        observer_table.append('| '+str(r)+' | '+ci(observer[str(r)])+' | '+observer[str(r)]['status']+' |')
        state_table.append('| '+str(r)+' | '+' | '.join(f(aggregate['ep4'][r][key]) for key in
            ['mean_actual_active_blocks','mean_dummy_row_fraction','timing_D_ms_per_forward',
             'timing_E_ms_per_forward','timing_C_ms_per_forward'])+' |')
    report=f'''# MultiBD × physical MoE EP discovery

## Contract and reproducibility

Same non-DMax checkpoint `{MODEL_ID}@{MODEL_REVISION}`; official Diffulex
`mbd-lms@{UPSTREAM_SHA}`. The primary intervention is ONLY `buffer_size=R`
for R=1/2/3/4. No model training, router/expert approximation, or controller.
Physical GPUs {MASK} map in order to logical 0/1/2/3 on four H100 80GB GPUs.
This is the explicit 2026-10-13 user override from original GPUs 2/3/6/7.
New hardware confirmation session: `{args.session}`. Historical calibration
and interrupted held-out evidence are preserved; absolute old/new timing is
not pooled. See `artifacts/environment_{args.session}.json` for UUIDs/topology.
Topology is dense TP4 / strided source-SP4 / physical EP4 / DP1 / PP1.
All six pairs within the selected H100 80GB set report NV18 fabric connectivity
in `nvidia-smi topo -m` (NVLink/NVSwitch, not a PCIe-only EP measurement).
Physical 4/5/6/7 map respectively to logical CUDA 0/1/2/3 and UUIDs
`{UUIDS[0]}`, `{UUIDS[1]}`, `{UUIDS[2]}`, `{UUIDS[3]}`.
The supported EP-disabled control is `TP4/EP1 TPFusedMoE`: every TP rank
receives the full token batch, retains 64 complete contiguous experts, and
sums partial expert outputs through TP all-reduce. It does not split each
expert's intermediate dimension, and does not invoke DeepEP sparse dispatch.
This is the native supported alternative, not a claim of zero communication.
The actual loaded model has 20 layers, 19 sparse layers, hidden 2048,
256 routed experts, top-8, intermediate 512, one shared expert. Each owner
has 64 complete experts/layer (384 MiB/layer); weights are not replicated.
See `artifacts/environment.json`, pins, ownership, and physical validation.

## Runtime integrity

Pinned upstream rejects EP configs despite including experimental EP modules.
The bounded bridge removes ONLY that config guard and repairs DeepEP normal
owner-local IDs (0..63), which upstream mistakenly treated as global IDs.
The failed unmodified physical proof is retained. Fixed real Dispatch →
owner-local GEMM → Combine passed numerical tolerance fixed before held-out.
Official scheduler, sampler, Block Buffer, attention and KV-cache state machine
are not reimplemented. This is Diffulex, NOT SGLang serving: only the previously
validated Python environment has SGLang in its directory name.

R is buffer CAPACITY, not necessarily the number of active blocks. Dummy,
to-cache and active rows are reported separately where captured. Natural EOS
is enabled, temperature 0, block/page32, naive non-DMax sampling, native
thresholds add=.9, semi=.9, accept=.95. Caps: 4096 output tokens, NFE1024,
model context4096, scoped process watchdogs. No failed artifact is removed.

After resuming on 4/5/6/7, a same-R native-repeat acceptance mismatch was
found on frozen HumanEval/11: final 122 tokens and NFE41 matched, but three
intermediate steps differed. The initial session was stopped, not credited
as valid performance. One-request counterbalanced diagnostics used six
repeats per setting plus trace. Fixed cuBLAS workspace and deterministic
operators (automatic scratch poison-fill disabled for DeepEP's asynchronous
buffer contract) passed that diagnostic without changing tokens, thresholds,
precision or routing. The final sweep freezes that recipe for ALL R and
both backends and checks acceptance parity after every problem. This is a
runtime reproducibility repair, not an EP optimization or an acceptance rule.
See interrupted evidence and `results/gpu4567_20261013/determinism_no_fill_audit.json`.
Because both diagnostic settings passed with fixed workspace, the exact
original non-determinism source remains unresolved; index-add alone is not
claimed as the proven cause. The failed poison-fill diagnostic is preserved.

## Workload and measurement

Metadata-only seed3407 freeze: 8 calibration + 24 frozen held-out problems,
half HumanEval and half MBPP. Prompts are official reproduction templates.
Full official EvalPlus 0.3.1 HumanEvalPlus v0.1.10 / MbppPlus v0.2.0 tests
are evaluated in private mount/network/PID namespaces plus resource limits.
The source problem examples alone are NOT the reported Plus quality.

Clean inference restores original unwrapped D/E/C functions. Held-out clean
and event-only runs are counterbalanced within the same loaded model; trace
is a separate heavy lane. Three request warmups and three measured clean/event
repeats; one heavy trace. Probe barriers and all serialization are outside
inference timing. Earlier separate-process calibration had drift and some
cross-rank flush races; all logs are retained and marked, not headline samples.

Prefix caching is enabled and each prompt is primed: headline E2E is therefore
WARM-PROMPT generation, not cold uncached serving latency. Full-request NFE
includes native prefill/refinement forwards; TPF/TPS denominators use returned
final output tokens, not padded buffer rows or raw acceptances later cut by EOS.
Upstream `is_truncated` includes normal EOS: only actual context/NFE/output/
repetition caps exclude quality. Raw output and all stop flags are retained.

Stage timing selects ONE cumulative-critical rank/request and uses its D/E/C
intervals, not a sum of unrelated rank maxima. Sparse routed D/E/C excludes
shared expert and full-source hidden/router-logit all-reduces; whole MoE events
capture those separately. Logical bytes use one BF16 hidden/output row per
remote source-token/destination, validated against actual DeepEP receive rows.
Logical payload excludes protocol/meta overhead and is NOT measured wire/HBM
traffic. Heavy-trace wall time is never headline latency.
Dispatch/Combine are dispatcher API intervals, not pure transport kernels:
Dispatch includes layout/count coordination and expert-major preparation;
Combine includes restoring slot order, FP32 local accumulation, conversion
back to BF16 and real reverse collective completion. Owner execution includes
the native packed expert GEMMs and routing-weight multiplication. GPU events
are resolved only after the request; no per-layer host synchronization is
added by the observer. Full-buffer restoration uses the existing native
all-reduces, rather than an assumed-free oracle layout step.

## Held-out primary table

{chr(10).join(table)}

## Paired conversion ratios, mean [request-bootstrap 95% CI]

{chr(10).join(paired_table)}

Ratios compare each problem to its own R=1 result before aggregation.
Stage ratios are observed values only; interpret them subject to the event
observer validity below, not as unconditional physical attribution.
Incremental EP-share statistics include only requests with positive added
step latency; their sample count and both deltas are retained. Near-zero
denominators can make that statistic unstable, and requests with no added
latency are not assigned an invented positive share.

## Same-checkpoint supported TP4/EP1 control

{chr(10).join(tp_table)}

Optional TP control validated: `{lanes['optional_tp_valid']}`. A failed or
incomplete control is NOT used in plots, paired contrasts, headline timing
or quality claims. Its raw results/failure flags remain in the evidence.
In this session native TP4/EP1 stopped on HumanEval/11 at R3: repeated clean
execution changed final tokens, NFE and acceptance despite the frozen
determinism recipe. See `results/{args.session}/tp_control_parity_audit.json`.
No alternative MoE kernel or engine rewrite was attempted to rescue this
optional comparison. The exact internal cause is unresolved. Thus EP-vs-TP
specificity is unavailable; the primary EP4 sweep is judged independently.

## Real block utilization and sparse stage costs

{chr(10).join(state_table)}

Costs are request-mean sums across all 19 sparse layers per actual forward.
Block utilization excludes native prefill forwards and separates physical
dummy rows from real active blocks. Capacity R is not equated to utilization.

## Event observer validity

{chr(10).join(observer_table)}

## Request-level paired statistics

Full means and >=2000-resample paired request CIs are in
`results/{args.session}/conversion_summary.json`; repeats/layers/forwards are NOT independent
samples. Event observer validity, TP-only contrasts and quality CIs are in
`results/{args.session}/headline_summary.json`. Stage attribution is UNTRUSTED whenever the
same-session observer overhead/upper uncertainty exceeds 10%. Clean E2E
remains reported in that case. Cross-R token identity is not required;
same-R output/NFE/acceptance identity across observers IS required.
R order was frozen as 3/2/4/1, first EP4 then TP4; observer order, not R order,
is counterbalanced within each request. Thus between-arm thermal/system drift
cannot be eliminated by treating repeats as new problems. Other users on
GPUs 0--3 were not interrupted; those GPUs are outside our execution mask.

## Prior boundary and limitations

See `prior_overlap.md`. MultiBD, MultiTF, TPF/TPS conversion loss, B1 splitting
and generic overlap are not new contributions. Earlier 5.77x live-only bound
is NOT measured Epoch. This is an eager bounded experimental EP bridge, not
the paper's optimized TP2 static runner; absolute TPS is not a paper reproduction.
The 24-problem quality subset cannot establish broad benchmark noninferiority.
No synthetic oracle is substituted for physical EP timing. No adaptive method
is implemented. Calibrated tolerances, request selection and R were not retuned
on held-out results.

## Reproduce / evidence

See `reproduce.md`. Code, full route/weight traces, actual ownership, process
failures, clean outputs, event-only intervals, per-forward/request/layer CSV,
quality test identities, bootstrap JSON and eight plots are retained in this
directory. Model weights and isolated third-party environments remain outside
git; exact revisions and commands are pinned.
The complete layer table is `results/{args.session}/ep_metrics.csv.gz`
(lossless compressed CSV). A preserved plain intermediate table is explicitly
named `ep_metrics.intermediate_before_full_sweep.csv`; do not substitute that
partial snapshot for the completed data.

## Final judgment

`{verdict}`. Completed all-R primary EP4 sweep: `{complete}`;
complete validated EP4/TP4 sweep: `{full_sweep_complete}`;
all held-out EP4 quality cases evaluable: `{quality_complete}`.
Gate details and request-level statistics: `results/{args.session}/final_verdict.json`.
Evidence-backed interpretation is finalized below after the complete sweep.
'''
    interpretation=results/'interpretation.md'
    if interpretation.exists():
        report=report.replace('Evidence-backed interpretation is finalized below after the complete sweep.',
                              'The session-specific interpretation and contract answers follow.')
        report+='\n'+interpretation.read_text()
    # This is a generated report, not a rewrite of any prior report.
    (ROOT/'report.md').write_text(report)
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
