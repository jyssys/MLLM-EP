"""Build new report tables from completed evidence. Never synthesize a measurement."""
import csv
import json
from collections import defaultdict
import numpy as np

from .common import ROOT,MODELS,write
from .analyze import estimate,table,jsonl

NAMES={'ifm_ouro':'IFM MoE Ouro','ifm_huginn':'IFM MoE Huginn','foil1':'Foil-1'}

def read(name):
    return list(csv.DictReader((ROOT/name).open()))

def number(value,digits=3):
    if value in ('',None):
        return 'NOT_RECORDED'
    return f'{float(value):.{digits}f}'

def ci(row,key='mean',lo='ci95_lo',hi='ci95_hi',digits=3):
    return f"{number(row[key],digits)} [{number(row[lo],digits)}, {number(row[hi],digits)}]"

def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(c) for c in row)+' |' for row in rows])+'\n'

def save(name,text):
    (ROOT/name).write_text(text.strip()+'\n')

def main():
    baseline=read('baseline_summary.csv')
    timing=read('timing_summary.csv')
    quality=read('quality_summary.csv')
    overall=read('model_overall_summary.csv')
    generation=read('generation_quality_summary.csv')
    fractions=read('generation_fraction_summary.csv')
    micro=read('expert_microbench_summary.csv')
    pins=json.loads((ROOT/'model_pins.json').read_text())
    validation=json.loads((ROOT/'validation.json').read_text())
    control=json.loads((ROOT/'plan_independent_controls.json').read_text())
    for job in control['jobs']:
        if job['model']=='foil1':
            job['id']=job['id'].removesuffix('_v1')+'_v2'
    write(ROOT/'plan_independent_controls_successful.json',control)
    plans=['p0_diagnostic_v5','p1','calibration','heldout','timing','generation_fidelity',
        'generation_quality','micro_v2','heavy_profile','independent_controls_successful','strict_foil_load']
    coverage=[]
    for name in plans:
        jobs=json.loads((ROOT/f'plan_{name}.json').read_text())['jobs']
        done=sum((ROOT/'raw'/j['id']/'complete.json').exists() for j in jobs)
        coverage.append(dict(plan=name,jobs=len(jobs),complete=done,evidence='ACTUAL_RUN_COMPLETION_RECORDS'))
    table('coverage.csv',coverage)

    def qrow(model,arm):
        return next(r for r in quality if r['model']==model and r['split']=='heldout' and r['arm']==arm)
    def trow(model):
        return next(r for r in timing if r['model']==model and r['input_length']=='512')
    def metric(model,name,pairing='temporal_gap1'):
        return next(r for r in overall if r['model']==model and r['split']=='heldout'
            and r['regime']=='prefill' and r['dataset']=='language' and r['pairing']==pairing and r['metric']==name)

    selected={}
    decision=[]
    for model in MODELS:
        rows=[r for r in quality if r['model']==model and r['split']=='heldout' and float(r['delta_nll_ci95_hi'])<=.02]
        best=max(rows,key=lambda r:float(r['effective_removed_fraction_work_weighted']))
        selected[model]=best
        f=float(trow(model)['expert_fraction_clean_optimistic'])
        q=float(best['effective_removed_fraction_work_weighted'])
        speed=1/(1-f*q)
        decision.append(dict(model=model,p0='CONDITIONAL_BACKEND_FIDELITY' if model.startswith('ifm') else 'PASS',
            p1='PASS',p2='NO_GO_HEADROOM',p3='NOT_RUN_GATE_NOT_PASSED',
            best_tested_nll_eligible_arm=best['arm'],effective_q=q,delta_nll=float(best['delta_nll']),
            delta_nll_ci95_lo=float(best['delta_nll_ci95_lo']),delta_nll_ci95_hi=float(best['delta_nll_ci95_hi']),
            f_expert_clean_denominator_optimistic=f,optimistic_proportional_speedup=speed,
            optimistic_latency_reduction_percent=100*(1-1/speed),measured_candidate_speedup=None,
            downstream_quality_margin='NOT_ESTABLISHED',verdict='NO_GO_HEADROOM_TESTED_NATIVE_B1_GRID'))
    table('decision_summary.csv',decision)
    dtable=md(['Model','Best tested NLL-eligible arm','Omitted work','ΔNLL [95% CI], nats/token',
        'Optimistic timing model','Achieved E2E speedup','P2 / P3'],[
        [NAMES[r['model']],r['best_tested_nll_eligible_arm'],number(100*r['effective_q'],2)+'%',
         ci(r,'delta_nll','delta_nll_ci95_lo','delta_nll_ci95_hi',4),number(r['optimistic_proportional_speedup'],4)+'×',
         'NOT_MEASURED','NO_GO_HEADROOM / NOT_RUN'] for r in decision])

    route_table=md(['Model','Top-k Jaccard [95% CI]','Exact-set match','Shuffled-position Jaccard',
        'Matched-input relative L2','Matched-output cosine','Matched-output relative L2'],[
        [NAMES[m],ci(metric(m,'jaccard')),number(metric(m,'exact_set')['mean']),
         number(metric(m,'jaccard','random_position_frequency_preserving')['mean']),
         number(metric(m,'same_expert_input_relative_l2')['mean']),
         ci(metric(m,'expert_output_cosine')),ci(metric(m,'expert_output_relative_l2'))] for m in MODELS])

    arch=md(['Model / revision','Learned parameters','Physical / recurrent sites','Passes / logical MoE calls',
        'Experts / top-k','Training tokens','Backend / GPU'],[
        [f"{NAMES[b['model']]} / `{pins[b['model']]['revision']}`",f"{int(b['parameter_count']):,}",
         f"{b['physical_moe_sites']} / {b['recurrent_moe_sites']}",f"{b['loops']} / {b['logical_moe_executions']}",
         '25 / 2' if b['model'].startswith('ifm') else '64 / 2 at core',
         '500B' if b['model'].startswith('ifm') else '100B',
         'xLLM FA4 / H100 SM90' if b['model'].startswith('ifm') else 'Official Foil SDPA / H100 SM90'] for b in baseline])
    timing_table=md(['Model','Input tokens','Clean forward ms [95% CI]','Expert span ms','Observer / clean',
        'Expert / observer','Expert / clean, optimistic','Peak allocated GiB','Validity'],[
        [NAMES[r['model']],r['input_length'],ci(r,'prefill_ms','ci95_lo_ms','ci95_hi_ms',2),
         number(r['expert_event_span_ms'],2),number(r['observer_to_clean_ratio']),
         number(100*float(r['expert_fraction_observer']),1)+'%',number(100*float(r['expert_fraction_clean_optimistic']),1)+'%',
         number(float(r['peak_allocated_bytes'])/2**30,2),r['fraction_validity']] for r in timing])

    genbase=read('baseline_generation_summary.csv')
    gen_groups=defaultdict(list)
    for r in genbase:
        gen_groups[(r['model'],r['prefix'],r['generated_tokens'])].append(r)
    gen_rows=[]
    for (model,prefix,tokens),rows in sorted(gen_groups.items()):
        gen_rows.append([NAMES[model],prefix,tokens,ci(estimate([float(r['e2e_ms']) for r in rows]),digits=1),
            number(np.mean([float(r['ttft_ms']) for r in rows]),1),
            number(np.mean([float(r['tpot_ms']) for r in rows]),1),
            number(np.mean([float(r['output_tps']) for r in rows]),2)])
    gen_table=md(['Model','Prefix','Output tokens','E2E ms [95% CI]','Mean TTFT ms','Mean TPOT ms','Mean output tokens/s'],gen_rows)

    downstream=[r for r in generation if r['scope']=='downstream_pilot']
    accuracy_table=md(['Model','Dataset','Arm','Correct / N','Accuracy','Δ accuracy pp [95% paired CI]',
        'Parse failures','Capped outputs','Token-exact / N','Peak expert cache MiB'],[
        [NAMES[r['model']],r['dataset'],r['arm'],f"{r['correct']} / {r['examples']}",number(r['accuracy_percent'],1)+'%',
         ci(r,'accuracy_delta_pp','delta_ci95_lo_pp','delta_ci95_hi_pp',1),r['parsing_failures'],r['truncated_outputs'],
         f"{r['token_exact_sequences']} / {r['examples']}",number(float(r['peak_expert_cache_tensor_bytes'])/2**20,2)] for r in downstream])

    micro_table=[]
    for model in MODELS:
        for length in [128,512,2048]:
            rows=[r for r in micro if r['model']==model and int(r['input_length'])==length]
            zero=next(r for r in rows if float(r['target_q'])==0)
            half=next(r for r in rows if float(r['target_q'])==.5)
            micro_table.append([NAMES[model],length,ci(zero,'component_ms','ci95_lo_ms','ci95_hi_ms',4),
                ci(half,'component_ms','ci95_lo_ms','ci95_hi_ms',4),
                number(float(zero['component_ms'])/float(half['component_ms']),3)+'×',
                zero['groups_nonempty']+' → '+half['groups_nonempty']])
    micro_md=md(['Model','Prefix geometry','q=0 component ms [95% CI]','q=0.5 component ms [95% CI]',
        'Ratio of means','Nonempty expert groups'],micro_table)

    backend_rows=[]
    for model in ['ifm_ouro','ifm_huginn']:
        p0=ROOT/'raw'/f"{model}_p0_gpu{0 if model=='ifm_ouro' else 3}_native_v5"
        old=json.loads((p0/'official_attention_backend_check.json').read_text())
        strict=json.loads((ROOT/'raw'/f'{model}_heavy_profile_v1/backend_routing_diagnostic.json').read_text())
        a,b=strict['native_generation']['ids'],strict['fp32_attention_generation']['ids']
        divergence=next((i+1 for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
        backend_rows.append([NAMES[model],number(old['relative_l2'],6),'FAIL at original 0.05',
            number(max(r['relative_l2'] for r in strict['operator_records']),6),
            number(strict['whole_model_relative_l2'],6),str(strict['greedy_8_token_parity']),divergence])
    backend_table=md(['Model','Original BF16-math whole-model L2','Original result',
        'Strict FP32 operator max L2','Strict FP32 whole-model L2','Strict 8-token parity','First different token (1-based)'],backend_rows)

    trade=[]
    for model in MODELS:
        best=selected[model]
        labels=['B0','q0',best['arm'],'O2_reuse_0.1','C1_router_0.3','C2_reuse_guardNone']
        suffix=best['arm'].split('_')[2]
        labels.append(f'N1_independent_site_drop_{suffix}')
        for label in labels:
            r=qrow(model,label)
            q=float(r['effective_removed_fraction_work_weighted'])
            f=float(trow(model)['expert_fraction_clean_optimistic'])
            trade.append([NAMES[model],label,number(100*q,2)+'%',
                ci(r,'delta_nll','delta_nll_ci95_lo','delta_nll_ci95_hi',4),number(r['top1_agreement'],4),
                number(1/(1-f*q),4)+'×',number(float(r['cache_peak_tensor_bytes'])/2**20,2) if r['cache_peak_tensor_bytes'] else 'NOT_RECORDED',
                'NOT_MEASURED'])
    trade_table=md(['Model','Arm','Actual omitted expert edges','ΔNLL [95% CI]','Mean top-1 agreement',
        'Optimistic timing model','Expert cache MiB','Measured E2E speedup'],trade)

    count=validation['checks']
    save('README.md',f'''# Looped MoE expert-efficiency headroom

Final decision: **NO_GO_HEADROOM for the tested native B1 candidate grid**.
P3 optimization did not run.
No measured candidate E2E speedup is claimed.

{dtable}

The timing column is an analytical estimate, not an achieved speedup.
It uses prefix-512 expert event spans and realized omitted rows.
It assumes proportional time removal and zero bookkeeping cost.
The NLL frontier does not establish downstream accuracy equivalence.

All three specified pretrained BF16 checkpoints ran on actual single GPUs.
The IFM baseline retains a backend-fidelity qualification.
Both official generation APIs and the adapters produced the same eight-token smoke output.
The strict FP32 attention diagnostic produced a token difference on another prompt.

Read [report.md](report.md) for Tables A–D and model-specific findings.
Read [go_no_go.md](go_no_go.md) for the decision and [reproduce.md](reproduce.md) for commands.
[validation.json](validation.json) records integrity checks.
[run_manifest.json](run_manifest.json) links runs to versions, hardware, and source hashes.

Raw data and failed runs are preserved in lossless [archives/](archives/).
The archive manifest records file-level SHA-256 checks.
Local originals remain in raw/ and logs/.
Checkpoint weights and full language passages are not committed.
''')
    save('go_no_go.md',f'''# Executive decision

## Verdict

**NO_GO_HEADROOM**, limited to the tested native B1 runtimes and candidate grid.
Do not implement a new expert-reuse system from these results.
The experiment does not prove that all future expert policies lack headroom.

{dtable}

## Why P3 did not run

The best tested points satisfy an upper-CI NLL sensitivity target of +0.02 nats/token.
Neither IFM timing estimate reaches the 1.10× discovery threshold.
The two IFM models therefore do not meet the 1.15× GO_PROTOTYPE target.
These are favorable estimates with no selector, cache, or gather overhead.
The native-weight microbenchmark further rejects exact proportional timing for sparse row removal.

Unconditional same-expert reuse increased held-out NLL by 3.3565, 4.3506, and 1.5990 nats/token.
The order is Ouro, Huginn, then Foil-1.
Shared expert identity does not imply interchangeable expert outputs.

## Evidence and limits

All three native no-op paths and q=0 paths passed their logit checks.
All three official generation APIs matched their adapters on an eight-token smoke prompt.
The IFM FA4-versus-math whole-model check failed its original threshold.
Later strict FP32 diagnostics also changed one of eight generated tokens on another prompt.
These failures remain visible.
Comparisons within the same official FA4 path remain internally controlled.

There are {count['full_model_quality_records']:,} primary full-model quality records and {count['free_running_sequences']} free-running sequences.
The language held-out set has 128 passages per model.
GSM8K and MATH-500 natural-generation pilots contain only eight questions each.
Weak MATH and Foil baselines cannot certify the 1 percentage point task-quality margin.
The study did not expand these pilots after the latency gate failed.

All native generation paths recompute the complete prefix without a KV cache.
The study does not describe production cached-decode serving or physical EP.
No positive loop-specific acceleration contribution is established.
''')

    save('report.md',f'''# Three-model Looped MoE expert-efficiency discovery

## 1. Outcome and evidence classes

The overall verdict is **NO_GO_HEADROOM for the tested native B1 candidate grid**.
P3 is NOT_RUN because the measured-denominator discovery gate did not pass.
No actual candidate E2E acceleration is reported.
No EP, TP, PP, quantization, training, or proxy model entered the experiment.

LIVE_MEASURED denotes an actual GPU timing.
LIVE_PROBE_DERIVED denotes statistics from actual model tensors.
LIVE_FULL_MODEL_INTERVENTION denotes quality after a changed full-model trajectory.
ANALYTICAL denotes a conditional timing model.
NOT_MEASURED and UNRESOLVED remain missing evidence, not zero costs.

{dtable}

## 2. Table A — Checkpoint and architecture verification

{arch}

The runtime pins appear in [environment.json](environment.json) and [model_pins.json](model_pins.json).
All learned weights remain BF16 and GPU-resident.
Native FP32 reductions, logits, and positional buffers retain their original precision.
Each forward uses one visible GPU.
The eight workers are independent experiments, not a distributed model.

Ouro repeats 28 physical layers four times.
Huginn executes 8 prelude layers, 12 core layers eight times, and 8 coda layers.
Foil-1 repeats one shared expert site sixteen times with untied attention.
Its outer eight-expert sites are not recurrence statistics.
Foil has 553,730,048 learned parameters and 9,437,184 FP32 RoPE buffer elements.
Their persistent element count is 563,167,232, which explains the spec's rounded 563.2M count.
The [official model card](https://huggingface.co/ShourenWSR/how-to-loop-moe) also reports 553.7M learned parameters.

Both IFM checkpoints use the strict native xLLM loader.
Foil uses its official remote code and Foil-1 subfolder.
The final strict Foil load reports no missing, unexpected, or mismatched tensors.
No tracked upstream model source changed.

## 3. P0 numerical qualification

Native no-op observers preserve exact logits and short greedy outputs for all three models.
The q=0 interventions preserve shifted NLL and KL.
All three adapters match their official generation APIs on an eight-token smoke prompt.
Those checks compare the same backend and checkpoint.

{backend_table}

The later strict diagnostic disables TF32 in its FP32 reference calculations.
Its operator error is small, but full-model greedy parity fails on that different prompt.
The report does not convert the original backend failure into a pass.
P0 is CONDITIONAL_BACKEND_FIDELITY for both IFM models and PASS for Foil-1.
No FA3-versus-FA4 benchmark accuracy equivalence is established.

## 4. Workloads and denominators

The sampling seed is 20261026.
Each domain has 8 smoke, 32 calibration, and 128 held-out IDs.
The domains are general language, GSM8K, and MATH-500.
Four synthetic examples test repeated and changing token patterns.
All models receive the same raw text, not assumed-identical token IDs.
Pretraining overlap is UNKNOWN.

The primary quality sweep uses 32 calibration and 128 held-out language passages per model.
Calibration prefixes have cap 128 and held-out prefixes have cap 512.
The stronger independent-site random-control follow-up uses cap 512 for both splits.
The summary does not duplicate its repeated O1 examples.
Actual token counts remain in quality_summary.csv.

P1 includes eight real GSM8K and eight real MATH prompts per model.
Those probes measure four free-running steps at the last position of each recomputed prefix.
Synthetic probes use prefix targets 128, 512, and 2,048.
Every model also has separate baseline generation timing at 16 and 64 output tokens.

The natural-generation accuracy pilot evaluates eight GSM8K and eight MATH questions per model.
It tests B0, O1 q=0.1, C1 threshold 0.2, and unconditional C2 reuse.
The broader 16-token fidelity grid runs all 27 primary arms on six prompts per model.
These capped rollouts are not full benchmark accuracy evaluations.
The 128-question downstream expansion did not run after the latency gate failed.
The default task-quality margin therefore remains UNRESOLVED, not PASS.

## 5. Table B — Clean native latency and the expert denominator

Each cell uses three independent model starts on physical GPU0.
Each start uses five warmups and twenty clean forward repetitions.
Intervals resample the three process-start means.
With three starts, interval coverage is only approximate.

{timing_table}

Event spans and clean wall time come from separate forwards.
The event spans include internal launch gaps.
They are not summed kernel-service times.
Observer overhead above 10% produces a PARTIAL label.
Attention and routing spans appear separately in component_spans.csv.
Dispatch, combine, and unassigned host intervals remain UNRESOLVED as clean critical-path percentages.
No invented 100% E2E partition is provided.

### Baseline model-only generation

{gen_table}

Both native runtime families recompute the entire prefix at every output token.
TPOT here is not cached-decode TPOT.
These hardware-shape runs disable EOS to fix output counts.
Natural-generation quality runs use real stop conditions instead.
Model loading, tokenizer work, network delivery, and request queueing are outside this timer.

The supplementary generation observer measures all sixteen actual steps.
It does not extrapolate the last replay or last forward.
It uses a separate representative prompt and process start.
Its clean latency need not equal the three-start macro table above.

{md(['Model','Clean 16-token E2E ms','Observer E2E ms','Expert span ms','Expert / clean, optimistic','Expert / observer'],[[NAMES[r['model']],number(r['clean_e2e_ms'],1),number(r['observer_e2e_ms'],1),number(r['expert_span_ms'],1),number(100*float(r['optimistic_fraction']),1)+'%',number(100*float(r['observer_fraction']),1)+'%'] for r in fractions])}

### GPU concurrency control

The operational control repeats one independent IFM worker and eight independent workers three times each.
On GPU0, parallel/isolated forward time is 0.99308×.
Other GPUs run owned burners in the isolated condition and model workers in the parallel condition.
This control does not establish complete machine-idle timing or cross-model interference equivalence.
Primary timing remains in the sequential GPU0 lane.
GPU UUIDs, NVLink topology, temperatures, and worker allocation remain in raw provenance.

## 6. Table C — Routing does not imply redundant output

This table pools adjacent passes at the same physical site and token position.
It uses 128 held-out language sequences per model, with prefix cap 512.
Intervals resample sequences, not individual expert events.
Each metric retains its own matched-observation denominator.

{route_table}

Input-relative-L2 conditions on the same expert.
The aggregate input cosine table instead covers all sampled tokens.
Those are different denominators.
Per-pass and gap-1/2/4/8 results appear in routing_summary.csv and representation_summary.csv.
Within-frame P10/P50/P90 averages are not pooled token quantiles.

Foil's late-pass output cosine can exceed 0.95 while relative L2 remains material.
Ouro's routing overlap falls in its final recurrent pass.
Huginn's final-pass output change also increases.
The negative examples show why an expert-ID match cannot certify output reuse.
Spatial and frequency-preserving shuffled controls remain separate from temporal comparisons.
Router entropy, confidence, margins, contribution fractions, and cancellation ratios appear in confidence_contribution_summary.csv.
Those confidence summaries use the smaller P1 smoke sample.
Mean relative contribution is 0.5 by top-2 normalization, so its distribution matters more than its mean.

## 7. Table D — Full-model quality versus work

{trade_table}

O1 is a contribution-ranked hindsight heuristic, not an optimal oracle.
O2 ranks same-expert output-reuse opportunities using baseline error.
Both require offline information unavailable at deployment.
C1 uses current router weights before expert arithmetic.
C2 uses its own previous-pass output and optional input-delta guards.
Current gate weights always multiply the cached raw output.

The work denominator includes all expert executions, including unmodified outer sites.
Reuse cannot replace outputs on the first pass.
Changed-router constraints also reduce actual omissions below the requested quota.
Rows disappear before native expert GEMMs.
This proves logical row omission, not proportional hardware-instruction reduction.
The primary teacher-forced runs did not record expert-cache peaks.
Those cells are NOT_RECORDED, not zero.
Natural-generation runs separately record controller cache tensors, not total HBM or all temporary allocations.
Unmeasured cache traffic and selector overhead are not assumed free in an implementable system.
They are free only in the explicitly optimistic timing model.

The independent-site N1 control matches realized O1 row counts at every site and pass.
Its NLL penalty is generally larger than contribution-informed O1.
This supports contribution selectivity, not a novel recurrence-specific method.
Legacy random masks were correlated across sites and remain labeled separately.
Free-running random controls match nominal quota, not necessarily realized work after divergence.

### Native-weight granularity test

{micro_md}

This microbenchmark covers one recurrent site per model.
It excludes row selection, compaction, routing, cache lookup, and combine.
Quota order changes between process starts.
The Huginn q=0 prefix-512 cell has a large between-start interval.
Its ratio of means must not be treated as a stable system speedup.
Half-row omission does not generally halve the measured component time.
The experiment did not measure tensor-core instruction counts or HBM bandwidth.

## 8. Natural-generation quality pilot

{accuracy_table}

The maximum output length is 256 new tokens.
Greedy generation permits EOS and registered next-question stops.
GSM8K uses the official IFM eight-shot text format.
MATH uses a declared zero-shot Problem/Answer format and a strict answer parser.
The parser is not the official MATH mathematical-equivalence evaluator.
Raw outputs, tokens, first divergence, answers, and stop reasons remain preserved.

Eight questions cannot establish a 1 percentage point accuracy margin.
Foil's baseline is weak on both tasks.
MATH baselines are weak and often truncated for all models.
Matching zero scores is not evidence of preserved quality.
The report makes no paper-quality reproduction claim.

## 9. Analytical headroom and decision

The conditional timing model is S = 1 / (1 - f_expert × q).
The optimistic f uses expert event spans divided by clean forward time.
That fraction comes from a separate representative passage, not each held-out quality request.
The more conservative observer denominator remains in oracle_summary.csv.
The model assumes fixed other costs, proportional expert-time removal, and zero added overhead.
It is not a proven universal bound.

The selected points are post hoc summaries of a predeclared diagnostic sweep.
They are not newly tuned deployable thresholds.
Even this favorable frontier stays below 1.10× in both IFM models at prefix 512.
Using the larger prefix-2,048 expert fractions still yields less than 1.10× for those same q values.
That length transfer is an estimate, not a prefix-2,048 quality measurement.
Downstream accuracy equivalence remains unproven and can only weaken this gate.

Unconditional C2 reuse causes large NLL increases in all three models.
Stricter guards preserve more fidelity but admit little reusable work.
The same shared expert weights still compute different refinements.
There is no measured recurrence-specific acceleration opportunity sufficient to justify P3 here.

## 10. Limits and next research

The native reference paths are cache-free research runtimes.
Their large launch and host costs need not match optimized cached serving.
The IFM backend qualification and downstream sample size limit generalization.
No claim extends to larger batches, long-context production decoding, or EP.
No machine-wide idle baseline, end-to-end candidate speedup, or hardware bandwidth result is available.

A separate optimized-runtime study could first establish the cached-decode expert fraction.
It must use the same model semantics and repeat the quality-work frontier.
This experiment does not authorize a new engine or an EP method.
Do not repeat unconditional expert-output reuse without a different, testable error-control mechanism.

## 11. Artifact map

- [methodology.md](methodology.md): alignment, intervention, and statistical definitions.
- [correctness.md](correctness.md): numerical checks and quality limits.
- [timing_contract.md](timing_contract.md): timer and observer domains.
- [prior_overlap.md](prior_overlap.md): TEAM, xLLM, Foil, and novelty boundaries.
- [reproduce.md](reproduce.md): actual commands and archive reconstruction.
- [validation.json](validation.json): completed integrity tests.
- [plots/](plots/): editable SVG figures and PNG previews.
- [archive_manifest.json](archive_manifest.json): lossless raw files and failed attempts.
''')

    for model in MODELS:
        d=next(r for r in decision if r['model']==model)
        rows=[r for r in quality if r['model']==model and r['split']=='heldout']
        save(f'{model}_analysis.md',f'''# {NAMES[model]} — model-specific result

Verdict: **NO_GO_HEADROOM for the tested native B1 grid**.
P0 status: {d['p0']}.
P1 completed. P2 completed with stated measurement limits. P3 did not run.

Checkpoint: `{MODELS[model]}` at `{pins[model]['revision']}`.
Foil uses only the Foil-1 subfolder.
No other checkpoint acts as a proxy.

The best tested upper-CI-NLL-eligible point is `{d['best_tested_nll_eligible_arm']}`.
It omits {100*d['effective_q']:.3f}% of total expert edges.
Its ΔNLL is {ci(d,'delta_nll','delta_nll_ci95_lo','delta_nll_ci95_hi',5)} nats/token.
The optimistic prefix-512 timing estimate is {d['optimistic_proportional_speedup']:.5f}×.
That estimate is not measured E2E acceleration.
The downstream 1 percentage point margin remains unverified.

## All held-out language arms

{md(['Arm','Actual omitted work','ΔNLL [95% CI]','Mean top-1 agreement','Cache MiB'],[[r['arm'],number(100*float(r['effective_removed_fraction_work_weighted']),2)+'%',ci(r,'delta_nll','delta_nll_ci95_lo','delta_nll_ci95_hi',4),number(r['top1_agreement'],4),number(float(r['cache_peak_tensor_bytes'])/2**20,2) if r['cache_peak_tensor_bytes'] else 'NOT_RECORDED'] for r in rows])}

All 128 held-out examples enter each primary arm.
No-op and q=0 rows provide the strict zero-change reference.
The larger report contains actual generation accuracy counts and parsing limitations.
See [report.md](report.md) for timing definitions, counterexamples, and the overall decision.
''')
    save('benchmark_summary.md',f'''# Baseline benchmark summary

Evidence: LIVE_MEASURED native model-only timing on physical GPU0.
Every forward cell has three starts, five warmups, and twenty repeats per start.
All parameters remain BF16.
The runtimes use no generation KV cache.

{timing_table}

{gen_table}

No candidate E2E speedup was measured.
Oracle and causal quality-probe wall times are not comparable clean benchmarks.
The model families have different runtimes, so cross-model raw latency is not a controlled architecture comparison.
''')
    print('REPORTS_WRITTEN',len(decision),'models')

if __name__=='__main__':
    main()
