"""Render evidence tables. Pending work cannot become a completed experiment."""
import csv
import json
import numpy as np
from .common import ROOT

MODELS=['ifm_ouro','ifm_huginn','foil1']

def read(name):
    path=ROOT/name
    return list(csv.DictReader(path.open())) if path.exists() else []

def number(value,digits=3):
    try:return f'{float(value):.{digits}f}'
    except (ValueError,TypeError):return str(value or 'NOT_AVAILABLE')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows])

def put(name,text):
    (ROOT/name).write_text(text.strip()+'\n')

def quality_table(rows):
    result=[]
    for r in rows:
        result.append([r['model'],r['ep'],r['benchmark'],
            f'{r["correct"]}/{r["total"]}' if r.get('correct') else r.get('status','NOT_AVAILABLE'),
            number(r.get('accuracy_pct')),f'[{number(r.get("ci95_low_pct"))}, {number(r.get("ci95_high_pct"))}]',
            number(r.get('reference_pct')),number(r.get('delta_pp')),r.get('compatibility',r.get('status','NOT_AVAILABLE'))])
    return table(['Model','EP','Benchmark','Correct/N or status','Accuracy %','Wilson 95% CI %','Paper %','Delta pp','Reference check'],result)

def build(preview=False):
    if not preview:
        state=ROOT/'attribution_campaign_state.json'
        assert state.exists() and json.loads(state.read_text())['status']=='COMPLETE','Final reports require completed GPU campaigns.'
    native=read('accuracy_native.csv');epquality=read('accuracy_ep.csv');paired=read('paired_accuracy.csv')
    baseline=read('baseline_e2e.csv');generic=read('generic_control.csv');kernels=read('kernel_activity.csv')
    oracle=read('placement_oracles.csv');transitions=read('transition_affinity.csv')
    fidelity=read('generation_fidelity.csv');nll=read('quality/heldout_nll.csv')
    counts=read('full_count_prediction.csv');comparison=read('native_ep_comparison.csv')
    strong=read('strongest_measured_generic.csv');metadata_controls=read('generic_metadata_controls.csv')
    prefix='IN_PROGRESS: this is a preview, not a completed PoC.' if preview else 'Completed evidence review. Missing capabilities remain explicit.'
    quality_bad=any(r.get('compatibility')=='REFERENCE_MISMATCH' for r in native)
    nll_reference_bad=any(r['ep']=='1' and abs(float(r['nll'])-float(r['reference_nll']))>.01 for r in nll)
    ep_reference_bad=any(r.get('compatibility')=='REFERENCE_MISMATCH' for r in epquality)
    quality_bad=quality_bad or nll_reference_bad or ep_reference_bad
    generation_bad=any(r.get('status')=='DIFFERENT_OUTPUT' for r in fidelity)
    verdict='PENDING' if preview else ('QUALITY_REPRO_BLOCKED' if quality_bad else 'NO_GO_FOR_PROTOTYPE')
    quality_rows=native+epquality
    quality_text=quality_table(quality_rows)
    put('accuracy_reproduction.md',f'''# Official quality reproduction

{prefix}

## Complete benchmark scores

{quality_text}

IFM MMLU uses the official five-shot, item-weighted metric.
ARC uses all 1,165 eligible four-choice questions and the official `acc_token` scorer.
The 1,172-question generic ARC variant is not this protocol.
Foil uses its own zero-shot paper benchmarks, not IFM reference scores.
The compatibility rule is the larger of 2 percentage points and the Wilson interval half-width.
This rule checks approximate reproduction. It does not establish statistical equivalence.

## Native versus EP task outcomes

{table(['Model','EP','Benchmark','Paired N','Delta pp','Paired bootstrap 95% CI pp','Correct to wrong','Completion'],[
    [r['model'],r['ep'],r['benchmark'],r['paired_n'],number(r['delta_pp']),
     f'[{number(r["ci95_low_pp"])}, {number(r["ci95_high_pp"])}]',r['native_correct_to_ep_wrong'],r['completion']] for r in paired])}

The paired comparison checks prompt hashes before matching outcomes.
Zero changed outcomes do not prove universal equivalence.
The interval measures observed paired outcome changes, not uncertainty over all future workloads.

## Foil held-out language-model loss

{table(['EP','Sequences','Scored tokens','Pure NLL, nat/token','Paper NLL','Native delta','Paired 95% CI'],[
    [r['ep'],r['sequences'],r['tokens'],number(r['nll'],6),r['reference_nll'],number(r['native_delta_nats'],6),
     f'[{number(r["delta_ci95_low"],6)}, {number(r["delta_ci95_high"],6)}]'] for r in nll])}

An empty loss table means that the GPU loss evaluation did not complete yet.
The official corpus builder checks document identifiers against all 140 training shards.
The rebuilt corpus contains 7,826 documents and 8,000,545 tokens.
The evaluator uses 1,953 complete 4,096-token windows and shifted targets.
Adjacent windows can share source documents. Sequence bootstrap intervals do not remove that dependence.
The paper does not publish a binary hash for its original held-out corpus.
The comparison therefore matches the recipe, not a proven identical token stream.
The 2.499 training loss is not a held-out target.
Native held-out NLL exceeds the declared 0.01-nat reference margin: **{nll_reference_bad}**.
This flag is meaningful only when the native loss row is complete.
An NLL discrepancy requires separate protocol and corpus analysis.
It must not disappear because the downstream anchors match.

## Protocol limitations

The official HellaSwag distribution is unavailable after its repository takedown.
No unofficial mirror substitutes for that distribution.
The full eleven-task IFM paper suite is not complete.
Available tasks use the official entry point and unchanged task specifications.
GSM8K, MATH-500, and DROP were not part of this completed anchor subset.
They receive no invented accuracy result.
Foil retains PyTorch 2.11.0 and Transformers 5.14.1.
Those versions differ from the published requirements file.
See `accuracy_protocol.md`, `accuracy_reference.csv`, and `quality/dataset_manifest.json`.
''')
    signal=[]
    for model in MODELS:
        values=[r for r in transitions if r['model']==model and r['lag']=='1']
        if not values:continue
        conditional=np.mean([float(r['conditional_recall2']) for r in values])
        popular=np.mean([float(r['popularity_recall2']) for r in values])
        signal.append([model,len(values),number(conditional),number(popular),number(conditional-popular)])
    signal_text=table(['Model','Lag-1 cells','Conditional Recall@2','Pass/domain popularity Recall@2','Increment'],signal)
    brief=[]
    for r in baseline:
        if r.get('status')!='PASS' or r['batch']!='1' or r['prefix_tokens']!='128':continue
        g=next(v for v in generic if all(v[k]==r[k] for k in ['model','ep','batch','prefix_tokens']))
        brief.append([r['model'],r['ep'],number(r['prefill_ms_mean']),number(r['tpot_ms_mean']),
            number(r['e2e_ms_mean']),number(r['output_tokens_s']),number(g['static_to_generic_e2e_speedup'])+'×'])
    timing_text=table(['Model','EP','Prefill ms','TPOT ms','32-token E2E ms','Output tokens/s','Static/generic E2E'],brief)
    quality_complete=(sum(bool(r.get('accuracy_pct')) for r in native)==7 and
        sum(bool(r.get('accuracy_pct')) for r in epquality)==14 and len(nll)==3)
    phases=[['P0','PASS','Pinned native checkpoints, source revisions, FA3, and TRUE EP topology.'],
        ['P1',('BLOCKED' if quality_bad else 'CONDITIONAL') if quality_complete else 'IN_PROGRESS',
            'Check full anchor scores and loss margins. HellaSwag distribution unavailable.'],
        ['P2','PASS','90 clean mandatory cells. Heavy observers remain separate.'],
        ['P3','PASS','Prompt-held-out co-activation and recurrent transition statistics.'],
        ['P4','BLOCKED','No validated wall-time oracle above strong production generic EP.'],
        ['P5','NOT_RUN','The 1.15× actionable generic-relative prototype gate is not established.'],
        ['P6','IN_PROGRESS' if preview else 'COMPLETE_WITH_LIMITATIONS','Reports retain unavailable capabilities and incomplete comparisons.']]
    put('report.md',f'''# Looped MoE expert affinity and communication-aware EP

{prefix}

## Verdict

**{verdict}**

The study does not establish a new loop-aware EP architecture with at least 1.15× actionable E2E headroom.
This is not a proof that every loop-aware EP method lacks headroom.
The strongest production generic EP comparator and a validated placement wall-time model remain unavailable.
The narrow prototype gate therefore remains unmet.
The original negative results remain unchanged.

{table(['Phase','Status','Evidence or limitation'],phases)}

## Models and exact execution

| Model | Learned parameters | Native topology | Experts | MoE calls/token |
|---|---:|---|---|---:|
| IFM Ouro | 8,780,889,600 | 28 physical layers repeated four times | 25/site, top-2 | 112 |
| IFM Huginn | 8,783,251,968 | 8 prelude, 12 core repeated eight times, 8 coda | 25/site, top-2 | 112 |
| Foil-1 | 553,730,048 | One prelude, one core repeated sixteen times, one coda | 8/64/8, top-2 | 18 |

All eight H100 GPUs participate in actual EP8 runs.
EP4 uses four actual ranks. Two EP4 evaluation groups are not an EP8 measurement.
Native attention, routers, top-2 score weighting, recurrent state, and BF16 weights remain unchanged.
Every expert has a real resident owner. Dispatch, owner-local expert GEMMs, and combine execute on GPUs.
The bridge also restores replicated dense hidden states after each MoE invocation.

These official execution paths recompute the full prefix for every generated token.
No retained KV cache exists in the measured adapter path.
The measured TPOT is therefore not cached single-token decode TPOT.
This distinction particularly limits conclusions about low-request cached serving.

## Accuracy evidence

{quality_table(native)}

Full native-versus-EP outcomes, paired intervals, and Foil loss appear in `accuracy_reproduction.md`.
Generation output differences detected in the matched hardware matrix: **{generation_bad}**.
See `generation_fidelity.csv` for completed and pending cells.
No implemented loop-aware candidate has a separate accuracy score.

## Actual timing at B1 and prefix 128

{timing_text}

These values are actual warmed 32-token greedy hardware rollouts with EOS disabled.
They are not client-to-server natural serving latency.
Each cell has five forward repetitions and three generation repetitions.
Intervals describe serial repetitions within one process, not independent server starts.
The complete 90-cell table appears in `benchmark_summary.md` and `baseline_e2e.csv`.
Additional counterbalanced process runs appear in `counterbalanced_generic.csv` when complete.

## Whole-E2E component percentages

Read **`e2e_breakdown.md`** for additive component milliseconds and percentages.
The table uses a complete observed endpoint timeline.
It does not sum rank kernel service times or extrapolate a single forward.
Dispatch and combine spans include rank waiting.
Expert spans can include submission gaps and local layout work.
The clean latency, observer latency, timing gate, and denominator remain separate.
Failed timing gates do not produce claims about clean E2E fractions.

## Recurrent affinity

{signal_text}

The table averages site, pass, domain, batch, and context cells uniformly.
It is a descriptive mean, not a pooled independent-prompt significance test.
Each prefill cell has sixteen held-out prompts per domain.
Calibration and held-out prompts are disjoint.
Per-cell bootstrap intervals preserve the prompt as the resampling unit.
The null preserves both top-2 slot marginals and distinct expert choices.
Finite-sample mutual information remains biased. Matched shuffle controls expose that bias.

The transition signal is real predictive evidence.
It does not make a future hidden state available.
It does not prove an exact future dispatch split matrix.
See `full_count_prediction.csv` for full-matrix tests when available.

## Why locality is not E2E gain

The original dispatcher sends one row per selected expert edge.
Two experts on the same GPU still receive two rows.
Co-placement alone therefore does not implement destination deduplication.
The fixed-placement trace model often trades lower fanout for higher rank-load imbalance.
Its transition-augmented graph can also add edges that carry no physical state in this bridge.
All MoE outputs return to a replicated dense state before subsequent computation.
There is no direct expert-to-expert hidden-state channel between recurrent passes.

The measured generic control selects an existing hidden-state copy using the exact current router.
It uses no recurrence information and moves no KV state.
Reduced remote payload does not consistently reduce its complete E2E time.
Dynamic source selection, metadata synchronization, local layout, and rank imbalance all remain possible causes.
This study does not isolate each cause enough to assign the entire result to one operation.

## Stronger generic metadata control

The original bridge already replicates exact current router outputs on every rank.
A generic control derives the full source/expert count matrix from those current outputs.
It removes the redundant count All-Gather, not expert work or authoritative routing.
Every diagnostic checks the matrix against an actual collective result before clean timing.
No future route, KV migration, or recurrent predictor enters this control.

{table(['Model','EP','Policy','Matched static/control speedup','E2E ms','Token-exact to static'],[
    [r['model'],r['ep'],r['policy'],number(r['static_to_control_e2e_speedup'])+'×',number(r['e2e_ms_mean']),r['token_exact_to_static']]
    for r in metadata_controls if r['batch']=='1' and r['prefix_tokens']=='128'])}

An empty table means that these GPU controls did not complete yet.
The full result is `generic_metadata_controls.csv`.
`strongest_measured_generic.csv` selects the lowest observed mean among exact-output controls.
This selection is not a held-out performance certification or a production DeepEP comparison.
Generic metadata savings do not establish recurrent novelty.
These controls do not receive invented downstream accuracy scores.

## Headroom and novelty limits

`placement_oracles.csv` separates live controls from fixed-placement traffic and load models.
The generic and recurrent maps preserve per-rank expert counts and weight capacity.
The search is a finite swap heuristic, not an exact optimal oracle.
Initialization traffic is structural byte accounting, not a measured transfer latency.
No per-loop weight migration receives zero cost.
The study does not report a fabricated generic-to-loop wall-time speedup.

Co-activation placement, destination deduplication, and inter-layer affinity already have generic MoE precedents.
Recurrent correlation alone does not establish novelty beyond those mechanisms.
See `prior_overlap.md` for primary sources and the exact novelty boundary.

## Next research decision

Do not build a large recurrent EP engine from these results alone.
First obtain a cached native path and a stronger generic dispatcher if low-request serving remains the target.
Then profile complete E2E intervals and identify a legal state channel or removable metadata dependency.
Any successor must compare the same memory budget, exact expert work, and GPU count.
It must preserve benchmark quality and demonstrate incremental wall-time gain above the generic control.
''')
    put('summary.md',f'''# Summary

{prefix}

Verdict: **{verdict}**.

The mandatory TRUE EP matrix contains {sum(r.get('status')=='PASS' for r in baseline)}/90 completed cells.
The study uses all three original checkpoints at EP4 and EP8.
The available native cloze and multiple-choice anchors reproduce the published level within the declared compatibility rule.
This compatibility rule is not an equivalence test.

{timing_text}

{signal_text}

Held-out recurrent affinity does not establish an executable shortcut.
The bridge restores replicated dense states after every MoE call.
Its original payload is per expert edge, not per unique destination.
The generic source control reduces some traffic without consistent E2E improvement.
No validated generic-relative loop-aware latency oracle meets the prototype gate.

{'Whole-E2E percentage measurements remain pending. Read `e2e_breakdown.md` for the current status.' if preview else '`e2e_breakdown.md` contains completed whole-E2E percentages and explicit timing warnings.'}
`accuracy_reproduction.md` contains quality counts, intervals, and unavailable tasks.
`report.md` describes scope, evidence, and the narrow conclusion.
''')
    put('go_no_go.md',f'''# Release and prototype gates

{prefix}

Final prototype decision: **{verdict}**.

{table(['Phase','Status','Reason'],phases)}

The 1.15× realistic oracle gate uses optimized generic EP as the denominator.
A traffic reduction or a routing prediction score cannot satisfy that gate.
The current bridge is not a production DeepEP engine.
Its source-owner control is a valid generic control under replicated dense state.
It is not proof that all feasible generic optimizations are exhausted.

No candidate passes the actionable oracle gate.
No candidate runtime, candidate downstream quality, or candidate live E2E speedup is claimed.
`NO_GO_FOR_PROTOTYPE` means insufficient justified headroom for this implementation decision.
It does not mean a measured upper bound below 1.10× for all possible loop-aware methods.
The unavailable strong-generic wall-time comparison remains a limitation.
''')
    timing_rows=[]
    glookup={(r['model'],r['ep'],r['batch'],r['prefix_tokens']):r for r in generic}
    memory=read('memory_per_rank.csv');stages=read('baseline_stage_breakdown.csv')
    for r in baseline:
        key=tuple(r[k] for k in ['model','ep','batch','prefix_tokens']);g=glookup.get(key,{})
        mem=[v for v in memory if tuple(v[k] for k in ['model','ep','batch','prefix_tokens'])==key]
        stage=next((v for v in stages if tuple(v[k] for k in ['model','ep','batch','prefix_tokens'])==key and v['phase']=='prefill'),{})
        timing_rows.append([*key,r['status'],number(r.get('prefill_ms_mean')),number(r.get('ttft_ms_mean')),
            number(r.get('tpot_ms_mean')),number(r.get('e2e_ms_mean')),number(r.get('output_tokens_s')),
            number(g.get('static_to_generic_e2e_speedup'))+'×',number(float(stage['dispatch_remote_payload_bytes'])/1048576) if stage else 'N/A',
            number(max(float(v['peak_allocated_bytes']) for v in mem)/2**30) if mem else 'N/A'])
    put('benchmark_summary.md',f'''# Complete mandatory hardware matrix

{prefix}

All times use milliseconds. Memory uses GiB. Payload uses MiB.
The batch is global, not independently replicated per-rank request work.
The output cap is 32 actual greedy tokens with EOS disabled.
TPOT includes full-prefix recomputation, not cached decode.

{table(['Model','EP','B','Prefix','Status','Prefill','TTFT','TPOT','E2E','Output tok/s','Static/generic','Dispatch MiB','Max rank peak GiB'],timing_rows)}

Dispatch payload excludes protocol headers and internal NCCL traffic.
The peak is a clean-forward CUDA allocation counter, not total NVML HBM or full-generation peak.
`baseline_e2e.csv` contains P50, P90, and mean intervals.
`baseline_stage_breakdown.csv` contains observer API spans, not clean component percentages.
`e2e_component_percentages.csv` contains complete-rollout exclusive component percentages.
`memory_per_rank.csv` separates resident expert weights, dense weights, buffers, and allocation counters.
`rank_imbalance.csv` retains physical-site and loop-level load and event-span dispersion.

## Native and independent-process comparisons

`native_ep_comparison.csv` compares matched one-GPU and EP hardware shapes.
It is not an equal-GPU-budget DP serving comparison.
The recovered IFM B16/prefix-2,048 native cells use expandable allocator segments.
The original EP cells use the default allocator.
The comparison marks those affected rows `allocator_matched=False`.
Other native cells retain the default allocator and their original evidence.
`native_oom_diagnosis.md` contains the actual failure and the successful fixed regression.
`counterbalanced_generic.csv` retains ABBA policy order and physical GPU groups.
Only two additional independent process runs are available at B1 and prefix 128.
They do not establish a precise restart confidence interval for every matrix cell.
''')
    put('profiling.md',f'''# Profiling and critical-path evidence

{prefix}

## Separate heavy observer

{table(['Model','EP','Rank','Observer window ms','Non-NCCL kernel union ms','NCCL union ms','No recorded activity ms','D2H events'],[
    [r['model'],r['ep'],r['rank'],number(r['observer_window_ms']),number(r['compute_kernel_union_ms']),
     number(r['nccl_kernel_union_ms']),number(r['no_recorded_gpu_activity_ms']),r['DtoH_events']] for r in kernels])}

These CUPTI intervals come from B1, prefix-128 diagnostic forwards.
They are not the clean serving interval and not a complete generated-request E2E breakdown.
Kernel unions avoid summing overlapping services.
NCCL activity includes waits and cannot identify pure network transfer time.
No recorded GPU activity excludes visible kernels, copies, and memset.
It does not measure SM occupancy or prove hardware idle.

The bridge performs one count All-Gather, two All-to-All calls, and one restoration All-Reduce per MoE invocation.
IFM executes 112 MoE invocations per forward. Foil executes 18.
Count conversion and Python split preparation cause CPU/GPU coordination.
The original local layout also creates metadata and gather/scatter work.
Those mechanisms are generic runtime costs, not demonstrated loop-specific overhead.

## Whole-E2E percentages

`e2e_breakdown.md` uses complete rollouts, not extrapolated diagnostic forwards.
Its event spans include submission gaps and stream waiting.
The observer checks token parity, overhead, and event-to-host alignment.
Failed timing gates remain visible.
Fine and coarse observers do not enter one pooled component distribution.

## Rank imbalance and clocks

`rank_imbalance.csv` records actual routing counts and API event-span dispersion.
Max-minus-mean event duration is not an exact rank-wait measurement.
`collective_arrival_skew.csv` uses same-host profiler submission timestamps.
Submission skew is neither isolated network latency nor exact GPU waiting.
Editable SVG timelines retain the distinction between compute, NCCL, and memory copies.

## Unavailable counters

Nsight Compute hardware-counter measurements are unavailable in this campaign.
Actual HBM read traffic, effective HBM bandwidth, tensor-core occupancy, and L2 hit rates remain NOT_AVAILABLE.
Checkpoint weight size is not a measurement of HBM traffic.
Persistent KV memory is N/A because these official adapter paths retain no cache.
''')
    put('dependency_dag.md',f'''# Exact dependency and communication graph

{prefix}

```mermaid
flowchart LR
    A[Native attention and recurrent hidden state] --> B[Exact current top-2 router]
    B --> C[Count All-Gather and split preparation]
    C --> D[BF16 expert-input All-to-All]
    D --> E[Resident owner-local expert GEMM]
    E --> F[BF16 expert-output All-to-All]
    F --> G[Top-2 weighting and dense restoration]
    G --> H[Native residual and intervening layers]
    H --> A
    B -. Recorded previous-loop route, metadata prediction only .-> C
```

The dotted edge is predictive information, not an independent activation operand.
The actual next router still depends on the next hidden state.
Original expert outputs cannot skip the intervening native computation.
All dense states become replicated after restoration.
One fixed resident expert copy gives no alternate rank for a route without changing residency.
The trace placement analysis does not move expert weights at loop boundaries.

`routing_lead_time.csv` measures diagnostic intervals between repeated visits to the same physical site.
That interval can permit metadata preparation in principle.
It does not imply available future activation or an exact future split matrix.
Prediction errors require exact validation and safe correction.
No speculative collective receives incorrect split counts in this study.

## Analytical tied-unrolled control

Replace each recurrent visit with a distinct logical node in the dependency graph.
Keep its expert tensor storage aliased to the original physical expert.
Keep current routes, exact values, resident owners, and operation order unchanged.
The dispatch, compute, combine, and restoration sequence remains the same.
This is a graph accounting control, not an executed fourth model.

Untying those weights would instead increase resident memory and violate the matched storage condition.
The study does not treat that larger-memory counterfactual as a fair live baseline.
Shared weights provide a real storage benefit.
That storage benefit alone is not a measured EP latency speedup.
''')
    put('oracle.md',f'''# Placement and actionable-headroom analysis

{prefix}

The detailed table is `placement_oracles.csv`.
It contains every model, EP degree, batch, and required prefix length.

| Oracle | Available evidence | Missing evidence |
|---|---|---|
| O0 | One destination per token is an unconstrained fanout lower bound. | A legal complete-model wall-time schedule. |
| O1 | Calibration-only generic fixed maps preserve expert counts and HBM. | A validated dispatch/GEMM/layout E2E cost model for these maps. |
| O2 | Transition-augmented maps use the same fixed-memory constraints. | An additional real state-transfer edge removed by recurrent information. |
| O3 | Held-out route prediction measures information, not perfect foresight. | A legal online policy with future exact routing. |
| O4 | Separate live static and current-router generic controls exist. | A validated oracle above strong production generic EP. |

The maps use finite pair swaps, not a proven global optimum.
Each physical expert has one fixed owner.
Every rank preserves its original expert count and BF16 weight capacity.
Changed initialization bytes appear in `runs/fixed_placement_maps.json`.
They are not measured transfer latency.
No per-loop weight remapping receives zero cost.

Destination fanout, payload, and rank-load dispersion are mechanism metrics.
They must not become E2E milliseconds without a validated cost model.
The original dispatcher sends one activation for each selected expert.
Co-location does not itself deduplicate that payload.
Destination deduplication also overlaps existing generic MoE work.

Static-to-generic measured ratios appear in `generic_control.csv`.
Generic-to-loop and static-to-loop live ratios remain NOT_AVAILABLE because no loop-aware runtime passed the gate.
This table does not silently replace missing speedups with 1.000×.
The 1.15× generic-relative prototype gate is not established.
''')
    put('failures.md',f'''# Failures, deviations, and unresolved evidence

{prefix}

1. The official IFM evaluator rejects FA4. A separate FA3 build passed its guard and checkpoint forwards.
2. A whole-vocabulary finite check exhausted memory at B16 and prefix 2,048. Bounded views now check every logit.
3. Concurrent state writers shared one temporary filename. Private temporary files fixed the reproduced publication race.
4. Unicode line splitting inflated a dataset count. Byte-level line counting preserved the unchanged JSONL dataset.
5. The original HellaSwag distribution is unavailable. No unofficial replacement enters the results.
6. The Foil evaluation environment differs from the paper requirements. Actual versions remain recorded.
7. The official adapters retain no KV cache. The measured decode is full-prefix recomputation.
8. The research EP bridge is not a production optimized generic EP engine. Its metadata and restoration costs remain material.
9. A fixed-placement traffic model is not a validated wall-time oracle. The O4 comparison remains blocked.
10. Hardware counters for HBM traffic and SM occupancy are unavailable. These metrics receive no guessed values.
11. Heavy CUPTI observations perturb execution. They remain separate from clean timing.
12. A failed whole-E2E observer gate blocks clean-E2E percentage claims. Observer-only evidence remains labeled.
13. Native IFM generation at B16/prefix-2,048 encountered allocator fragmentation. The recovery preserves model arithmetic and records allocator differences.

Huginn initialization depends on the batch-row index.
Matched native and EP evaluations preserve the official batch geometry.
Four independent B1 evaluations are not a substitute for one four-choice batch.

The study does not change the previous NO-GO results.
It does not repeat expert skipping, expert-output reuse, or loop partition pipelines.
No synthetic or analytical result appears as an actual E2E speedup.
''')
    from .aggregate_quality import csv_write
    quality_by={(r['model'],r['ep']):[] for r in quality_rows}
    for r in quality_rows:quality_by[(r['model'],r['ep'])].append(r)
    joined=[]
    for r in baseline:
        for score in quality_by.get((r['model'],r['ep']),[]):
            joined.append(dict(model=r['model'],ep=r['ep'],batch=r['batch'],prefix_tokens=r['prefix_tokens'],
                e2e_ms=r.get('e2e_ms_mean','NOT_AVAILABLE'),benchmark=score['benchmark'],accuracy_pct=score.get('accuracy_pct','NOT_AVAILABLE'),
                quality_status=score.get('compatibility',score.get('status','NOT_AVAILABLE')),
                scope='Task evaluation geometry is separate from each hardware shape. Not per-shape benchmark accuracy.'))
    csv_write(ROOT/'quality_vs_latency.csv',joined)
    return dict(preview=preview,verdict=verdict,completed_performance_cells=sum(r.get('status')=='PASS' for r in baseline),
        native_reference_mismatch=quality_bad,hardware_generation_difference=generation_bad,
        native_nll_reference_mismatch=nll_reference_bad,
        ep_reference_mismatch=ep_reference_bad,
        official_ep_tasks=len([r for r in epquality if r.get('accuracy_pct')]),
        matched_native_ep_generation_cells=sum(r.get('status')!='NOT_COMPLETE' for r in fidelity),
        native_comparison_rows=len(comparison),full_count_prediction_rows=len(counts),placement_oracle_rows=len(oracle),
        strong_generic_cells=len(strong),heldout_nll_configurations=len(nll))

if __name__=='__main__':
    import sys
    print(build(preview='--preview' in sys.argv))
