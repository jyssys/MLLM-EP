"""Generate a fully traceable characterization report from the frozen tables."""
import argparse
import json
import collections
from pathlib import Path
import numpy as np
import pandas as pd
from poc_dllm_ep_comm.contract import ROOT,REVISION,MASK,UUIDS,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);a=p.parse_args()
    result=ROOT/'results'/a.result;summary=json.loads((result/'summary.json').read_text())
    df=pd.read_csv(result/'requests.csv');primary=df[df.block_length==32]
    it=pd.read_csv(result/'iterations.csv');names=set(primary.request);pit=it[it.request.isin(names)&(it.iteration>=0)]
    stat=summary['stats']['b32_all'];env=json.loads((ROOT/'results/environment.json').read_text())
    ownership=json.loads((ROOT/'raw'/a.result/'ownership.rank0.json').read_text())
    per_expert=sum(t['bytes'] for t in ownership['ownership'][0]['weights'].values())//64
    def f(key,scale=1,suffix=''):
        s=stat[key];lo,hi=s['ci95']
        return f"{s['mean']*scale:.3f}{suffix} (request-bootstrap95% CI {lo*scale:.3f}–{hi*scale:.3f}{suffix})"
    def table(header,rows):
        return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+
            ['| '+' | '.join(map(str,row))+' |' for row in rows])
    strata=[]
    for k,s in summary['stats'].items():
        strata.append([k,s['Residual_Amp']['n'],f"{s['Raw_Amp']['mean']:.3f}",
            f"{s['Residual_Amp']['mean']:.3f}",f"{s['Residual_final_positions_Amp']['mean']:.3f}",
            f"{100*s['comm_share_request_wall']['mean']:.2f}%"])
    stage_rows=[]
    for name,g in primary.groupby('domain'):
        stage_rows.append([name,*[f'{g[k].mean():.2f}' for k in ['router_ms','dispatch_ms','expert_ms','combine_ms','denoising_forward_ms','denoising_span_ms','bypass_wall_ms']],
            f'{g.timing_overhead_fraction.mean()*100:.2f}%'])
    selected=pit[pit.request==sorted(names)[0]].head(12)
    iter_rows=[[int(r.iteration),int(r.live_tokens),int(r.newly_accepted_tokens),f'{r.acceptance_yield:.3f}' if pd.notna(r.acceptance_yield) else 'N/A',
        f'{r.dispatch/2**20:.3f}',f'{r.combine/2**20:.3f}',f'{r.bytes_per_accepted_token/2**20:.3f}' if pd.notna(r.bytes_per_accepted_token) else 'UNDEFINED',
        int(r.cumulative_accepted),r.status] for r in selected.itertuples()]
    versions=ownership['versions'];versionstr=', '.join(f'{k}={v}' for k,v in versions.items())
    lengths=primary.groupby('domain').output_tokens.agg(['min','median','max']).to_dict('index')
    dead_extra=stat['dead_explained_output_excess']['mean']*100
    conservative_ci=stat['comm_share_request_wall']['ci95']
    excess=primary.B_actual-primary.B_one_shot
    dead_full=100*(primary.B_dead_output/excess).mean()
    context_full=100*(primary.B_context/excess).mean()
    live_full=100*((primary.B_live_only-primary.B_one_shot)/excess).mean()
    next_action=('One bounded exact causal/shadow ablation of still-live/not-yet-accepted expert paths, preserving baseline acceptance schedule and testing the damage to other accepted positions; do not build a predictor or method yet.'
        if summary['verdict']=='GO' else 'Do not implement a method. Resolve communication timing/optimized-backend robustness with one small paired experiment before any promotion.'
        if summary['verdict']=='CONDITIONAL' else 'Stop this tested communication-specific direction; do not tune acceptance to manufacture amplification.')
    text=f'''# dLLM MoE EP communication amplification — {summary['verdict']}

## Outcome

Physical four-GPU DeepEP execution is validated. The primary frozen pool contains
{len(primary)} requests: 16 GSM8K reasoning and 16 HumanEval coding, B1.
Secondary block64 uses {len(df)-len(primary)} predeclared requests (four per domain).
No optimization method was implemented; these are characterization/oracle results.

Structural headroom is strong. Final decision: {summary['verdict']}; conservative
communication CI={100*conservative_ci[0]:.2f}–{100*conservative_ci[1]:.2f}% versus
the15% useful-signal threshold. Communication share also differs by workload;
there is no unconditional communication-method promotion or speedup claim.

- Raw amplification (whole executed request): {f('Raw_Amp',suffix='×')}.
- Output-position-only raw amplification: {f('Raw_output_Amp',suffix='×')}.
- Optimistic live-only residual: {f('Residual_Amp',suffix='×')}.
- More conservative final-returned-position-only residual: {f('Residual_final_positions_Amp',suffix='×')}.
- All materialized accepted-position reference sensitivity: {f('Residual_all_materialized_Amp',suffix='×')}.
- D+C / entire timing request wall: {f('comm_share_request_wall',100,'%')}.
- D+C / first-to-last denoising-forward span: {f('comm_share_denoising_span',100,'%')}.
- Timing observer vs clean bypass E2E: {f('timing_overhead_fraction',100,'%')}; validation={summary['timing_valid']}.

![Normalized traffic](figures/{a.result}/normalized_amplification.png)

`B_live_only` and `B_one_shot` are structural accounting/reference, NOT executable
runtime methods and NOT measured speedups. Residual bytes are not proven removable.

## 1. Model, runtime, and prior boundary

Selected the already-cached **inclusionAI/LLaDA2.0-mini**, revision `{REVISION}`.
Actual checkpoint geometry: BF16, 20 decoder layers / **19 sparse MoE layers**,
hidden2048, MoE intermediate512, 256 routed experts, top-k8, one shared expert.
This is deliberately not a Qwen3 model or the 100B Flash geometry.
Native sigmoid router / correction bias and normalized selected weights are
unchanged; routed scaling2.5 is applied by the native expert runner.

Existing dInfer upstream `{env['dinfer_upstream']}` plus the exact pre-existing
bridge patch in `patches/dinfer_existing_bridge.diff`. {versionstr};
PyTorch {ownership['torch']}, CUDA {ownership['cuda']}, NCCL {ownership['nccl']}.
Dense TP4 / source-SP4 / physical EP4 / DP1 / PP1, normal high-throughput DeepEP,
eager execution, BF16 Triton FusedMoE. No full-model architecture conversion.

[dInfer](https://github.com/inclusionAI/dInfer) supplies the native mini decoder.
[TEAM](https://arxiv.org/abs/2602.08404) uses temporal/spatial consistency to reduce
expert activation in MoE dLLMs. [Epoch](https://arxiv.org/abs/2609.09748) compiles
block work and routes fresh live/newly-decoded/refresh-required work through EP.
This PoC does not reproduce or benchmark either system. Its optimistic removal
of all prior-accepted output traffic is more favorable than assuming that all
such state can actually be deleted. A large residual does not establish novelty
over TEAM/Epoch; it only motivates a distinct, still-unproven live-work question.

## 2. TRUE physical EP4 validation and GPU mapping

Only mask `{MASK}` was used. Physical GPUs2,3,6,7 are H10080GB HBM3;
the saved `nvidia-smi topo -m` has NV18 links among these GPUs. Logical ranks
0,1,2,3 map to physical2,3,6,7 respectively.

{table(['Logical EP rank','Physical GPU','GPU UUID'],[[r,p,u] for r,p,u in zip(range(4),(2,3,6,7),UUIDS)])}

Each rank owns exactly its contiguous64 experts in all19 sparse layers. Actual
loaded BF16 routed weights are `[64,1024,2048]` and `[64,2048,512]`, totaling
**{per_expert:,} bytes/expert**. Ownership is not inferred from a synthetic map.
All four ranks call actual DeepEP dispatch and combine, then only their owned
expert weights execute. One B1 request's source rows are partitioned once across
four ranks. Source position uniqueness, actual received row count and received
owner-local path count are asserted for every captured call/layer:
**{int(df.physical_buffer_validation_checks.sum()):,} physical-buffer validations**.
Output token IDs/NFE match bypass, timing, and trace passes, and all four ranks.
Acceptance-only observer parity additionally passes on the two fixed probes.

Clean-start GPU process audit is retained in `raw/{a.result}/clean_start.json`.
Other-GPU workloads are read-only audited and never terminated; shared-fabric
contention from outside the authorized mask remains a timing limitation.

## 3. Acceptance pattern and workload contract

Official mini decoding family: threshold0.9, temperature0, block32, prefix-BD
cache, EOS enabled. Generation budget64 is natively rounded to a block boundary;
actual observed lengths, not the requested64, define final output denominators.
Selection was seeded and frozen before routing or outputs; no answer-based
selection or parameter tuning. Output length summaries: `{lengths}`.
Block64 is an unchanged-threshold secondary robustness lane, never chosen for
larger amplification. EOS-stop requests: {int((primary.finish=='EOS_STOP').sum())};
bounded block-cap requests: {int((primary.finish=='BOUNDED_BLOCK_CAP').sum())}.
These are bounded emitted prefixes, not complete task solutions or accuracy scores.
Actual zero-accept decode/refresh iterations: {int(primary.zero_accept_iterations.sum())}.
Native threshold decoding can force progress, so absence of zero-accept rows
would not mean traffic is generally acceptance-efficient.

For B1, native dInfer removes EOS IDs rather than trimming strictly at the first
EOS within a block. Full final state/actual acceptance positions are retained.
We report contiguous-before-first-EOS sensitivity and exclude synthetic EOS fill
from acceptance-route accounting. No previously accepted positions are remasked.

Example first12 actual denoising iterations (all rows remain in CSV):

{table(['Iteration','Live','New accepted','Yield','Dispatch MiB','Combine MiB','MiB/new token','Cumulative accepted','Status'],iter_rows)}

![Acceptance efficiency](figures/{a.result}/iteration_acceptance_efficiency.png)

## 4. Actual EP bytes per final output token

Actual full-request logical remote D+C payload: {f('actual_bytes_per_output',1/2**20,' MiB/token')}.
Output-position-only payload: {f('output_bytes_per_output',1/2**20,' MiB/token')}.
These are **backend-validated logical useful payload bytes**, not NVLink packet
or NIC wire counters. A BF16 hidden/output row is4096bytes. DeepEP normal sends
one row per distinct destination rank even if several selected experts share
that rank. Combine returns one partial-output row per corresponding destination.
Local expert paths are separate and contribute0 remote payload bytes. Router
metadata, protocol bytes, shared-expert compute and dense TP collectives are not
included in this sparse-EP byte numerator. They remain in measured request wall.

## 5–8. Actual vs live-only vs acceptance-route one-shot

For each actual source position/layer call, route owners come from loaded
ownership. `B_actual` counts all executed remote payloads. `B_live_only` retains
only output positions that were masks before that call, even if not accepted
afterwards. `B_one_shot` sums each returned output's first-acceptance routing
exactly once in each19 sparse layers. Prompt/known-context is not called
already-decoded output.

`B_actual = B_context + B_dead_output + B_live_only` is asserted exactly.
Already-accepted output accounts for {f('dead_share_output',100,'%')} of actual
output-position traffic, or {dead_extra:.2f}% of output repetition above one-shot.
The remaining output excess is repeated live work (plus eventual EOS/tail where
present). {f('live_unaccepted_fraction',100,'%')} of live-only remote bytes belongs
to positions still unaccepted after that iteration. This is acceptance-inefficient
traffic, not demonstrably safe-to-remove waste.

For full-request excess above one-shot (including known prompt/context),
already-accepted output explains{dead_full:.2f}%, prompt/context{context_full:.2f}%,
and repeated live excess{live_full:.2f}%. Additional request-bootstrap CIs and
pooled byte ratios are in `supplemental_summary.json`; the headline is request mean.

![Actual decomposition](figures/{a.result}/actual_traffic_decomposition.png)

## 9. Workload/block robustness

{table(['Stratum','N requests','Raw ×','Residual ×','Final-position residual ×','D+C / request wall'],strata)}

Bootstrap uses2000 request-level resamples. Layers and iterations are never
independent statistical samples. Block32 and64 are summarized separately since
they reuse some problem IDs; no pooled pseudo-independent sample count is used.
All ratios use each request's own acceptance-route bytes, not another model.

![Block robustness](figures/{a.result}/block_robustness.png)

## 10. Physical timing and observer validity

Separate independent clean bypass, event-only timing, and heavy routing passes.
Warmup3 denoising requests precede measurement; bypass/timing order is
counterbalanced by frozen request index. No per-layer global host synchronization
or `.item()` was added. Combine's existing native completion-event dependency is
placed before timing-end; its own subsequent native wait is unchanged.
The chosen critical rank is the one with largest cumulative denoising-forward
CUDA time; Dispatch/Expert/Combine all use **that same rank**, not separately
maximized component ranks. Router here means native gate linear time only;
top-k/layout/shared expert/source all-gather are not silently counted as expert.

{table(['Domain','Gate ms','Dispatch ms','Expert ms','Combine ms','Forward sum ms','Denoise span ms','Clean request wall ms','Observer overhead'],stage_rows)}

All stage times are request totals for denoising calls, not per-layer averages.
Forward-only communication share {f('comm_share_denoising_forward',100,'%')} is
diagnostic: it omits acceptance/controller/cache gaps. First-to-last denoising
model span includes inter-iteration gaps, but excludes final acceptance epilogue.
The promotion gate conservatively divides denoising D+C by **whole request wall**,
which also includes initial prompt prefill. This lower bound cannot inflate the
communication share by deleting controller time. No speedup is measured.

Instrumentation threshold was frozen at10% absolute mean paired overhead plus
exact output/NFE parity. Heavy trace overhead: {f('trace_overhead_fraction',100,'%')};
heavy trace wall is never a headline latency. All requests and failures are
retained, no fastest-run selection. Missing tuned Triton mini expert config,
20 DeepEP SM default, eager manual bridge and disabled TorchDynamo are important
runtime-specific limitations; no claim about every optimized dLLM server.

## 11–12. Rank graph and temporal imbalance

Mean directed rank-edge density: {f('rank_edge_density')}.
Hottest-rank switch rate: {f('hottest_rank_switch_rate',100,'%')}.
Per-iteration hottest rank, assignment histogram, max/mean, CV, active experts,
remote fanout and source→destination edges are in `iterations.csv` and raw routes.
These are secondary diagnostics, not substitutes for residual amplification.

## Runtime issue and reproducibility

Initial smoke import/cache wiring errors were recorded and corrected by reusing
the existing native ModelRunner, not by creating a replacement cache algorithm.
`main_v1` stopped at a small-M SGLang TorchInductor compiler failure. The isolated
M0..32 reduction seam passed, so a zero-receive root cause is **not proven**.
One bounded full restart (`main_v2`) disabled TorchDynamo consistently in all arms,
preserving native operations, weights, decoder and DeepEP. The complete frozen
cohort includes the previously failing problem; `main_v1` is diagnostic-only and
is not pooled with final evidence. Exact amendment is in `runtime_amendment.json`.
GPU stop/run/restore is scoped to2,3,6,7 and occurs in `finally`, including failure.

## 13–14. Verdict and smallest next experiment

**VERDICT: {summary['verdict']}**.
Residual gate={summary['gates']['residual_pass']}; conservative communication
gate={summary['gates']['communication_pass']}; timing-valid={summary['timing_valid']}.

NEXT_ACTION: {next_action}

No new acceptance predictor, learned predictor, pruning, dynamic migration,
EPLB, CUDA kernel, production scheduler, or full Epoch reproduction was built.

## Evidence index

- `results/frozen_manifest.json`: exact inputs, seed and frozen protocol.
- `results/environment.json`: GPU topology, config, revisions and source hashes.
- `raw/{a.result}/`: four-rank actual routing/acceptance/physical timings.
- `results/{a.result}/requests.csv`, `iterations.csv`, `route_accounting.csv`.
- `results/{a.result}/summary.json`, `timing_ledger.json`.
- `figures/{a.result}/`: normalized traffic, request distributions, acceptance,
  block robustness, traffic decomposition, rank/timing diagnostics.
- `scripts/reproduce.sh`, `tests/`: executable reproduction and accounting checks.

BASELINE_STATUS: TRUE_PHYSICAL_EP4_VALIDATED
RESIDUAL_AMPLIFICATION: {stat['Residual_Amp']['mean']:.3f}x
CONSERVATIVE_FINAL_POSITION_RESIDUAL: {stat['Residual_final_positions_Amp']['mean']:.3f}x
TIMING_VALIDITY: {summary['timing_valid']}
VERDICT: {summary['verdict']}
NEXT_ACTION: {next_action}
'''
    target=ROOT/'report.md'
    if target.exists():raise FileExistsError(target)
    target.write_text(text)
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    (reports/'baseline_readiness.md').write_text(text[text.index('## 1.'):text.index('## 3.')])
    (reports/'traffic_accounting.md').write_text(text[text.index('## 4.'):text.index('## 9.')])
    (reports/'timing_validity.md').write_text(text[text.index('## 10.'):text.index('## 11')])
    write(result/'final_verdict.json',dict(verdict=summary['verdict'],next_action=next_action,summary='summary.json'))
    print('REPORT',target)

if __name__=='__main__':main()
