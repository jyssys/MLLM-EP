"""Write English reports from completed artifacts. Keep evidence categories explicit."""
import statistics
from pathlib import Path
from poc_decode_first_depth_aware_dp_v3.contract import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])

def number(value,digits=2):return f'{value:.{digits}f}'
def ci(values):return '['+', '.join(number(v,3) for v in values)+']'
def report(name,text):
    path=ROOT/'reports'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text.strip()+'\n')

def main():
    env=read(ROOT/'artifacts/runtime_environment.json')
    quality=read(ROOT/'artifacts/quality_recheck_summary_v2.json')
    depth=read(ROOT/'artifacts/depth_heterogeneity.json')
    fit=read(ROOT/'artifacts/cdb_cost_model.json')
    metrics=read(ROOT/'artifacts/live_dp4_metrics.json')
    refined=read(ROOT/'artifacts/oracle_refinement_v2.json')
    predictor=read(ROOT/'artifacts/causal_predictor_v2.json')
    physical=read(ROOT/'artifacts/live_hindsight_confirmation.json')
    observer=read(ROOT/'artifacts/DP4_observer_check.json')
    parity=read(ROOT/'artifacts/scheduler_output_parity.json')
    kv=read(ROOT/'outputs/batched_kv_probe_v2/status.json')
    cost=read(ROOT/'outputs/cost_surface_v1/cost_surface.json')
    verdict='NO_GO_SCHEDULING'
    write(ROOT/'artifacts/final_decision.json',dict(verdict=verdict,
        source_selection='INCUMBENT_CONTINUE',quality='QUALITY_WARNING',
        G0R='RUN: usable online original-gate CDB',G1='RUN: physical depth heterogeneity',
        G2='RUN: live DP4 and validated P95/ITL cost model. Some TTFT strata remain partial.',
        G3='NOT_CONFIRMED: modeled C64 gain 12.77%, live C64 gain 4.41% with CI crossing zero. No statistically clear quality-matched scheduling gain.',
        G4='PREDICTION_BLOCKED: prompt-only ridge fails to recover the oracle gain.',
        prefill='PREFILL_FULL_DEPTH',RL='NOT_RUN',
        claim='No deployable depth-aware scheduler or quality-preserving scheduling speedup is established.',
        next_action='Do not start RL. First measure teacher-aligned batch-shape effects on logits and exit decisions with fixed prefixes.'))

    report('environment.md',f'''# Environment

The physical GPUs are 4, 5, 6, and 7. Each is an NVIDIA H100 with 81,559 MiB reported HBM.
The recorded topology reports NV18 links between the allowed GPU pairs.
The topology record is `artifacts/environment_initial.json`.
We did not change GPU clocks or operate another user's GPU processes.
The scoped guard stops only recognized owned burners. It restores burners after each run and after failures.

| Replica | Physical GPU | Logical CUDA device | UUID suffix |
|---|---:|---:|---|
| 0 | 4 | 0 | 6076e2f2-5b63-3761-5586-56ceb7df8139 |
| 1 | 5 | 0 | a1a1cfcf-93a1-3544-9a5e-e58144b68730 |
| 2 | 6 | 0 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 3 | 7 | 0 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

The runtime uses independent DP4, TP1 per replica, and PP1.
It does not execute TP or EP collectives between these replicas.
The checkpoint is `{MODEL_ID}`, revision `{REVISION}`.
The CDB source SHA is `{CDB_REVISION}`.
The upstream source tree remained clean.

| Component | Version |
|---|---|
| Python | {env['python']} |
| PyTorch | {env['torch']} |
| CUDA used by PyTorch | {env['CUDA']} |
| Transformers | {env['transformers']} |
| Tokenizers | {env['tokenizers']} |

The isolated dependency directories are unchanged from v2.
The native Python executable is `{PYTHON}`.
The analysis Python executable is `{REFERENCE_PYTHON}`.
`runtime_environment.json` records the model config and its SHA256.

Decode uses CUDA graphs. Prefill graphs remain disabled.
The selected synchronous gate path uses q=0.5, min R2, max R4, and `last_exited` KV.
The optional async CDB path is not a validated baseline in this study.
No threshold, checkpoint, precision, or cache-policy search was performed.
''')

    depth_table=table(['Domain','N','Mean depth','95% request-bootstrap CI','Between-request SD','EOS / N'],[
        [r['dataset'],r['n'],number(r['mean_depth'],4),ci(r['CI95_mean_depth']),number(r['between_request_std'],4),f"{r['EOS']}/{r['n']}"] for r in depth['domains']])
    regression=table(['Retrospective features','Held-out N','MAE, ms','R²'],[
        [r['name'],r['heldout_n'],number(r['MAE_ms'],3),number(r['R2'],4)] for r in depth['retrospective_cost_regressions']])
    report('decode_depth_traces.md',f'''# Stage 1: online decode depth

The new pilot contains 128 distinct requests. Each of four domains contributes 32 requests.
Each domain has eight calibration requests and 24 held-out requests.
The metadata split preceded new inference.
The pilot uses GSM8K, MATH-500, Alpaca, and ShareGPT.
The discovery math questions exclude the old quality-pilot IDs.
Exact prompts, sources, revisions, and response hashes appear in `discovery_frozen.json`.

The measurement contains {depth['decode_tokens']:,} real online decode exits.
The original gate selected R2 for 3,789 tokens, R3 for 24,438, and R4 for 3,719.
The first output token uses full-R4 prefill and is absent from this histogram.
The request-mean depth range is {number(depth['request_mean_depth_range'][0],4)} to {number(depth['request_mean_depth_range'][1],4)}.

{depth_table}

The cap is 256 with natural EOS. We did not force continuation after EOS.
Only seven requests reached EOS within this bounded discovery run.
This pilot is not full-response benchmark quality evidence.
Equal caps do not imply equal actual output lengths.
The equal-progress analysis uses only requests that naturally reach each selected window.

## Cost relevance

{regression}

These models predict trace-lane B1 decode duration.
Actual loop work is a retrospective feature, not an admission-time feature.
The comparison controls prompt length, output length, and declared domain.
It does not establish that a prompt can predict future depth.
It also does not establish a DP scheduling gain.

The physical heterogeneity gate passes for this pilot.
The campaign does not expand to 512 requests before resolving the predictor and baseline limitations.
''')

    validation=table(['Metric','Median RR/queue relative error'],[
        [key,number(value*100,3)+'%'] for key,value in fit['median_RR_queue_relative_errors'].items()])
    partial=[v for v in fit['validation'] if v['policy'] in ['round_robin','queue'] and v['relative_errors']['TTFT_P95_ms']>.10]
    partial_table=table(['Case','P95 TTFT prediction error'],[[r['case'],number(r['relative_errors']['TTFT_P95_ms']*100,2)+'%'] for r in partial])
    report('cdb_cost_model.md',f'''# Stage 2: native CDB cost model

The cost model uses real host spans and CUDA-event spans from calibration requests.
It fits separate prefill, recurrent, and coda costs.
The recurrent cost uses active batch size, context volume, and terminal-KV copy activity.
The fit also includes a common measured host-loop residual.
It does not use a constant time per request-depth unit.

The simulator preserves slot-homogeneous kernel batches, survivor priority, coda priority, and pinned KV ownership.
It includes full-depth prefill and finite resident/KV capacity.
It does not migrate running requests or independently redistribute their tokens.

Burst transition counts matched physical execution in 28/28 rank cases.
That check covers prefill, recurrent, and coda launch counts.
Transition-count parity does not prove timing accuracy.

## Held-out live timing validation

{validation}

The overall status is `{fit['status']}` for the exploratory cost model.
The primary P95 E2E and ITL estimates meet the approximate median-error target.
The complete per-case errors remain in `cdb_cost_model.json`.
Some TTFT strata exceed 10% error:

{partial_table}

Those TTFT strata are `PARTIAL`. We do not promote their detailed TTFT oracle estimates.
The model-validation result is not a guarantee for unseen context lengths or workloads.

## Observer and cost surfaces

The eight-request B1 trace observer added {number(quality['observer']['graph_trace']['mean_decode_delta_percent'],3)}% mean decode time.
Its paired request-bootstrap CI is {ci(quality['observer']['graph_trace']['CI95_mean_decode_delta_percent'])}%.
All eight token outputs and depth sequences matched.
The paired C64 burst had {observer['exact_output_requests']}/64 exact outputs between trace and clean lanes.
Its median E2E difference was {number(observer['median_E2E_delta_percent'],3)}%.
That is one paired system run, not 64 independent observer repeats.

The 1K/4K/8K context sweep uses repeated synthetic prompts on physical GPU7.
It tests B1/B2/B4/B8/B16 within the explicit KV budget.
B32 and capacity-exceeding combinations remain documented skips.
These samples show physical batch/context costs, not task quality or serving gains.
The task model does not extrapolate its short-context validation to the full synthetic grid.

Fixed-R4, adaptive-refill, and adaptive-no-refill diagnostics use the same eight calibration prompts.
The adaptive arms retain the same q and KV policy.
Fixed R4 has a different quality operating point. It is only a contextual work-cost control.
See `outputs/ablation_*` and `artifacts/ablation_summary.json`.
''')

    selected=[r for r in metrics if not r['trace'] and r['arrival']=='burst']
    live_table=table(['C total','Policy','P95 E2E, ms','P95 ITL, ms','Requests/s','Idle/eligible witnesses'],[
        [r['concurrency'],r['policy'],number(r['E2E_P95_ms'],1),number(r['ITL_P95_ms'],2),number(r['requests_per_s'],3),r['idle_while_other_runnable_witnesses']]
        for r in selected])
    parity_table=table(['C','Policy vs RR','Exact token requests','Exact depth-sequence requests'],[
        [r['concurrency'],r['policy'],f"{r['exact_token_requests']}/{r['n']}",f"{r['exact_depth_records']}/{r['n']}"]
        for r in parity if r['concurrency']>=16])
    report('live_dp4_baselines.md',f'''# Live DP4 baselines

These are physical online runs on four independent CDB replicas.
Requests receive a rank before prefill. No KV migration occurs.
The gate, weights, precision, cache policy, and generation cap remain unchanged across policies.
The controller sends requests to the authors' native arrival/scheduler loop.

Round-robin uses request order. Queue placement uses actual outstanding requests.
Length placement uses calibration output-length priors and known prompt length.
The cost placement uses calibration service-time priors, queue state, and nominal equal rank rates.
It is an inexpensive causal cost baseline, not a proven optimal CDB-aware scheduler.
Its label `strong_cost` must not imply dominance over every generic service-cost policy.

## Burst sweep

{live_table}

The concurrency is the total burst size across four GPUs, not the per-GPU size.
Closed-loop and Poisson discovery use the C16 setting and 32 requests per case.
The Poisson rate is 1 request/s. Actual active concurrency is not fixed at 16.
The full non-burst concurrency/arrival-rate matrix is `NOT_RUN`.
All additional results and the frozen relative-goodput limit appear in the CSV/JSON artifacts.

The idle/eligible witness requires a completed rank and another rank's sampled runnable queue.
It is not a GPU timestamp measurement of collective waiting.
Multiple witnesses in one run are not independent statistical samples.
The witnesses show residual cross-rank work after one rank becomes idle.

## Output consistency

{parity_table}

Identical placement controls retain their output traces.
Changed batch composition can produce different outputs and exit sequences.
We do not assert that every such difference comes only from BF16 rounding.
No full benchmark establishes quality parity between these placement arms.

The separate batched correctness probe passed {kv['checks']} exact terminal-KV copy checks.
Its actual decoder-layer count also matched executed recurrent work.
The first probe failed because its counting assertion omitted the four prefill passes.
The corrected v2 probe uses the same model and cache code. The failed v1 log remains intact.
This diagnostic is not clean latency or a benchmark-quality pass.
''')

    oracle_table=table(['C','Best observed non-depth control','Control P95, ms','Oracle P95, ms','Model gain','95% paired CI'],[
        [r['C'],r['control'],number(r['control_E2E_P95_ms'],1),number(r['oracle_E2E_P95_ms'],1),number(r['gain_percent'],3)+'%',ci(r['CI95_gain_percent'])+'%']
        for r in refined['rows']])
    physical_table=table(['C','Repeats','Generic mean P95, ms','Hindsight mean P95, ms','Measured gain','95% paired CI'],[
        [r['C'],r['repeats'],number(statistics.mean(r['generic_P95_E2E_ms_by_repeat']),1),number(statistics.mean(r['hindsight_P95_E2E_ms_by_repeat']),1),
            number(r['gain_percent'],3)+'%',ci(r['CI95_gain_percent'])+'%'] for r in physical['rows']])
    report('decode_oracle.md',f'''# Stage 3: admission-only depth oracle

The oracle knows future online depth traces and, in its combined arm, true output lengths.
That information is unavailable to a deployable admission policy.
The oracle chooses a GPU before prefill and keeps each request on that GPU.
The local CDB execution obeys the selected cache and recurrence rules.

## Controls and search

The comparison includes causal queue/length/cost policies, true-length information, mean-depth information, and full depth trajectories.
The true-length-only heuristic uses calibration depth profiles, not future true depth.
That heuristic can perform worse than a generic baseline.
Therefore, its larger raw gap is not evidence of depth-specific headroom.
The main comparison uses the best observed non-depth control, including the true-length candidate.

Tiny C1/C2/C4 bursts have a solved symmetric placement optimum and zero additional gain.
Larger bursts use three complete one-request-move neighborhoods and bounded pair swaps.
The stronger search evaluates hundreds of placements and preserves the first search as v1 evidence.
It is best-known clairvoyant search, not a proof of global optimality.

{oracle_table}

The table contains validated-cost estimates. It does not contain measured GPU speedups.
The intervals resample paired request IDs at fixed selected placements.
They exclude model-fit uncertainty and selection uncertainty.
The natural-EOS traces have cap 256.
The large-burst oracle is exploratory and does not establish a universal upper bound.

## Physical hindsight-placement checks

{physical_table}

These runs use the real online gate and physical DP4, not synthetic depth replay.
The assignments come from the frozen trace oracle.
Each check has three paired repeats with alternating arm order.
They reuse the pilot requests used for oracle selection. They are not untouched confirmation.
Their outputs and actual depths can differ after placement changes.
Thus they test a hindsight plan's physical realization, not exact portability of every B1 trace.
Generated-token counts, exact-output fractions, and natural K-token controls remain in `live_hindsight_confirmation.json`.
No benchmark-quality equivalence is established for those changed placements.

Depth-homogeneous grouping is an offline diagnostic, not a new production scheduler.
Joint microbatch optimization is `NOT_RUN`.
Free token migration, free KV movement, and independent cross-rank token stealing are forbidden.
No zero-cost bound is presented as a physical result.
''')

    pred_table=table(['Target','Calibration N','Held-out N','Ridge MAE','Constant MAE'],[
        [r['target'],r['calibration_n'],r['heldout_n'],number(r['heldout_MAE'],6),number(r['constant_MAE'],6)] for r in predictor['models']])
    scheduling=table(['C','Prompt-only policy','Modeled gain vs generic','Oracle recovery'],[
        [r['C'],r['policy'],number(r['gain_vs_generic_percent'],3)+'%',number(r['recovered_oracle_fraction']*100,2)+'%'] for r in predictor['scheduling']])
    report('causal_predictor.md',f'''# Stage 4: bounded causal predictor probe

The C64 refined oracle passed the exploratory headroom threshold.
This allowed a small supervised probe. No RL or exit-gate training occurs.
The predictor uses known prompt length, prompt text statistics, and declared domain metadata.
It does not use future output tokens, true exit traces, or post-prefill hidden states as inputs.

Thirty-two calibration requests fit separate depth and output-length ridge models.
Calibration leave-one-out error selects the ridge coefficient.
The 96 held-out requests do not select model parameters.

{pred_table}

{scheduling}

These scheduling values are cost-model estimates, not measured predictor-runtime speedups.
The matched-length arm keeps the output-length prior fixed to isolate predicted depth.
The simple predictor does not recover useful broad oracle headroom.
Its held-out depth error is worse than the constant-depth control.
The status is `PREDICTION_BLOCKED` for this probe.
This is not proof that every possible predictor must fail.

The derived work forecast multiplies predicted depth by predicted decode-token count.
Its held-out MAE is {number(predictor['derived_total_work']['heldout_MAE_loop_tokens'],3)} loop-token operations.
The constant control's MAE is {number(predictor['derived_total_work']['constant_MAE_loop_tokens'],3)} loop-token operations.

We do not train a more complex predictor to rescue this result.
We do not start RL. We do not implement a new serving framework.
''')

    report('final_report.md',f'''# Decode-first Dynamic-Depth-Aware DP Scheduling: v3

## Verdict

The final verdict is **`{verdict}`**.
The source-selection decision is **`INCUMBENT_CONTINUE`**.
The quality status remains **`QUALITY_WARNING`**, with **`REFERENCE_MATCH_PARTIAL`**.
The v2 `QUALITY_FAIL` result remains unchanged.
It did not stop the v3 scheduling discovery.

The campaign found physical decode-depth heterogeneity and residual cross-GPU work.
It also found conditional clairvoyant scheduling headroom.
However, the tested prompt-only predictor did not recover that headroom.
The largest modeled gain did not reproduce as a statistically clear live gain.
The evidence does not justify RL training or a new multi-GPU framework.
This is a NO-GO for proceeding from this pilot, not proof that all depth-aware schedulers must fail.

## Model and selected runtime

The model is `{MODEL_ID}`, revision `{REVISION}`.
The CDB authors' source SHA is `{CDB_REVISION}`.
The incumbent already uses that implementation and the checkpoint's original gate.
No verified independent better-quality replacement was available in the bounded source audit.
The selection therefore preserves q=0.5, min R2, max R4, and `last_exited` KV.

Four independent BF16 replicas use physical GPUs 4/5/6/7.
Each process sees logical CUDA device 0. There is no inter-replica TP or EP collective.
Decode uses CUDA graphs. The quality recheck uses the retained eager evaluation path.
We do not report eager-to-graph gains as a research contribution.

## Quality and reference recovery

| Domain | Old Fixed R4 | Old Adaptive | New 8192-cap Adaptive | Fixed minus Adaptive 95% CI |
|---|---:|---:|---:|---:|
| GSM8K | 22/32, 68.75% | 21/32, 65.625% | 21/32, 65.625% | [-6.25, 12.5] pp |
| MATH-500 | 24/32, 75.00% | 19/32, 59.375% | 19/32, 59.375% | [3.125, 28.125] pp |

The old primary cap was 4096, followed by a paired 8192 truncation-union sensitivity check.
The new adaptive recheck uses cap 8192 for all 64 frozen IDs.
All shared token prefixes match the old adaptive outputs exactly.
Two MATH outputs still reach the new cap. Each domain has two parse failures.
The old physical decode-work savings were 22.86% and 27.97%.
The new MATH work denominator includes longer capped outputs and is not the old primary denominator.
We do not interpret that denominator change as an optimization gain.

The literature evaluation protocols differ from this greedy answer-extraction pilot.
No paper-equivalent absolute quality claim or fixed-R4 quality equivalence is established.
Changed CDB batch shapes also produce some changed greedy outputs.
Full benchmark quality across placement policies is `NOT_RUN`, not `PASS`.

## Stage 1: physical heterogeneity

The new pilot has 128 requests, including 32 calibration and 96 held-out requests.
Its four domains are GSM8K, MATH-500, Alpaca, and ShareGPT.
It records 31,946 online decode exits and 95,768 actual loop-token operations.
Mean request depth spans about 2.475–3.867.
The retrospective B1 cost model improves from R²=0.8712 to 0.9926 when actual depth work is included.
Its held-out MAE changes from 148.214 ms to 20.814 ms.
This is cost relevance, not causal predictability or a scheduling speedup.

The generation cap is 256 with natural EOS. Only seven requests reach EOS within that cap.
The traces characterize bounded progress, not complete-answer serving quality.
The campaign does not label this pilot as a 512-request or final large-scale study.

## Stage 2: live baselines and model validity

The live burst sweep covers C=1/2/4/8/16/32/64 total requests.
It compares round-robin, queue, calibrated length, and inexpensive calibrated cost placement.
Closed-loop and Poisson discovery use the C16 setting.
No request changes GPU after prefill.

Native CDB requires slot-homogeneous recurrent kernels under the selected KV layout.
This creates local batch fragmentation and interacts with cross-GPU load balance.
The simulator reproduces all 28 burst/rank operator-count comparisons exactly.
Median RR/queue prediction errors are about 2.00% for makespan and 1.86% for P95 E2E.
Median P95 ITL error is 1.31%. Some P95 TTFT strata remain `PARTIAL` above 10% error.

Idle/eligible witnesses confirm that one rank can finish while another retains runnable work.
Those witnesses do not establish that depth-aware admission can remove all remaining work.
The cost baseline is not proven strongest among all CDB-aware generic schedulers.
The optional async CDB path is not a validated comparator.

## Stage 3: oracle versus generic and length controls

{oracle_table}

These are modeled estimates from a best-known clairvoyant search.
C1/C2/C4 have zero additional solved-optimal headroom under the symmetric model.
Larger cases are not solved to global optimality.
The true-length heuristic sometimes loses to a generic control.
We therefore do not present its larger isolated gap as depth-specific gain.

The C64 refined estimate motivates a bounded physical hindsight-placement check.
The same pilot also supplies C16/C32 checks.

{physical_table}

These are measured live placements selected with future information.
They are not measured causal-policy speedups.
They reuse selection requests and allow batch-dependent output/depth differences.
The three-repeat uncertainty and natural K-token controls remain visible in the raw artifacts.
They do not prove untouched-cohort quality equivalence.

At C64, the 12.77% modeled P95 gain becomes a 4.41% live point estimate.
Its paired 95% CI is [-8.298, 13.341]%.
All three live comparison intervals include zero.
The changed placements also change outputs for many requests.
Therefore, this pilot does not establish statistically clear, quality-matched additional scheduling headroom.
On the observed C64 routes, the cost model predicts the live checks within about 1.5%.
Thus accurate conditional cost prediction does not establish portable depth traces across placements.
This is not proof of a specific numerical error or gate bug.

## Stage 4: admissible prediction

{pred_table}

At C64, the tested depth-only matched-length policy worsens modeled P95 latency by about 5.56%.
The joint depth/length predictor worsens it by about 7.96%.
The depth MAE is worse than the constant control.
Thus predictor accuracy and useful scheduling gain do not pass together.
No RL, complex predictor, or new engine is used to rescue the result.

## Gates and limitations

| Gate | Result |
|---|---|
| Stage 0R | `INCUMBENT_CONTINUE`. Online gate, physical skipping, and usable CDB verified. |
| Quality | `QUALITY_WARNING`. The old MATH drop and partial reference fidelity remain. |
| Heterogeneity | `RUN`: physical variation remains beyond output-length controls. |
| Cost and DP4 | `RUN`: measured DP4, exact burst transition counts, validated primary latency model. |
| Scheduling oracle | `NOT_CONFIRMED`: conditional modeled headroom does not establish a statistically clear quality-matched live gain. |
| Causal predictor | `PREDICTION_BLOCKED` for the tested prompt-only models. |
| Prefill | `PREFILL_FULL_DEPTH`. No new prefill runtime. |
| RL | `NOT_RUN`. |

The pilot is small and heavily capped. It does not establish broad natural-EOS scheduling gains.
The oracle CI excludes cost-model and search-selection uncertainty.
Batch-dependent numerical/output changes limit quality-portability claims.
The full non-burst concurrency/rate matrix is not measured.
The generic cost controls can still be strengthened.

## Next action

Do not start RL or a new multi-GPU architecture now.
First measure logits and exit decisions with teacher-aligned prefixes at B1/B4/B16.
This small control can test the portability of depth traces across batch shapes.
Keep the same gate and KV policy. Do not train a new gate.
Any later scheduling study also needs a stronger CDB-aware generic cost baseline and fresh natural-EOS requests.

## Publication and reproduction

Branch: `flashvep/decode-first-depth-aware-dp-v3`.
Parent evidence commit: `{BASE_COMMIT}`.
The published Git HEAD identifies this code/evidence snapshot.
The final user response records its commit SHA.
Only the new v3 directory is part of this commit.
Old dirty/staged work, reports, and raw evidence remain unchanged.
Large per-rank JSON traces are published as lossless gzip copies.
Their original uncompressed files remain on the server.

See `reproduce.md`, `timing_contract.md`, and the stage reports for exact commands and definitions.

## Figures

- [Online depth and cost](../figures/depth_heterogeneity.png)
- [Live DP4 scaling](../figures/live_dp4_scaling.png)
- [Cost-model validation](../figures/cost_model_validation.png)
- [Modeled oracle headroom](../figures/oracle_headroom.png)
- [Measured hindsight checks](../figures/live_hindsight_confirmation.png)
- [Prompt-only predictor](../figures/causal_predictor_scheduling.png)
''')

if __name__=='__main__':main()
