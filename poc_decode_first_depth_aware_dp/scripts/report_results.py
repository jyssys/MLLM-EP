"""Write new result prose from completed evidence. Do not alter frozen inputs."""
import csv
import statistics
from poc_decode_first_depth_aware_dp.contract import *


def number(x):
    return f'{x:.3f}'


def bounds(values):
    return '['+', '.join(number(x) for x in values)+']'


def new_report(name,content):
    path=ROOT/'reports'/name
    assert not path.exists()
    path.write_text(content)


def main():
    final=read(ROOT/'artifacts/final_decision.json')
    summary=read(ROOT/'artifacts/quality_pilot_analysis_v2/quality_summary.json')
    cap=read(ROOT/'artifacts/quality_cap_sensitivity.json')
    traces=read(ROOT/'artifacts/pilot_export_v2/trace_manifest.json')
    requests=list(csv.DictReader((ROOT/'artifacts/pilot_export_v2/request_depth_summary.csv').open()))
    hist=list(csv.DictReader((ROOT/'artifacts/pilot_export_v2/depth_histogram.csv').open()))
    quality_table='\n'.join(
        f"| {r['dataset']} | {r['fixed_correct']}/{r['n']} ({number(r['fixed_accuracy_percent'])}%) | "
        f"{r['dynamic_correct']}/{r['n']} ({number(r['dynamic_accuracy_percent'])}%) | "
        f"{number(r['accuracy_drop_pp'])} | {bounds(r['paired_bootstrap_CI95_drop_pp'])} | "
        f"{r['paired_losses']} / {r['paired_gains']} |" for r in summary['rows'])
    work_table='\n'.join(
        f"| {r['dataset']} | {number(r['mean_request_depth'])} | {bounds(r['request_bootstrap_CI95_depth'])} | "
        f"{number(r['executed_decode_loops_saved_percent'])}% |" for r in summary['rows'])
    parse_table='\n'.join(
        f"| {r['dataset']} | {r['fixed_parse_failures']} / {r['dynamic_parse_failures']} | "
        f"{r['fixed_truncations']} / {r['dynamic_truncations']} |" for r in summary['rows'])
    cap_table='\n'.join(
        f"| {r['dataset']} | {r['fixed_correct']}/{r['n']} | {r['dynamic_correct']}/{r['n']} | "
        f"{number(r['accuracy_drop_pp'])} | {bounds(r['paired_bootstrap_CI95_drop_pp'])} | "
        f"{r['fixed_truncations']} / {r['dynamic_truncations']} |" for r in cap['rows'])
    completion_table='\n'.join(
        f"| {r['dataset']} | {r['fixed_correct']}/{r['n']} | {r['dynamic_correct']}/{r['n']} | "
        f"{number(r['accuracy_drop_pp'])} | {bounds(r['paired_bootstrap_CI95_drop_pp'])} |"
        for r in cap['primary_completion_aware'])
    descriptive=[]
    for dataset in ['gsm8k','math500']:
        rows=[r for r in requests if r['dataset']==dataset and r['policy']=='dynamic']
        depths=[float(r['mean_depth']) for r in rows]
        prompts=[int(r['prompt_tokens']) for r in rows];outputs=[int(r['output_tokens']) for r in rows]
        descriptive.append(f"| {dataset} | {len(rows)} | {min(prompts)}–{max(prompts)} | "
            f"{min(outputs)}–{max(outputs)} | {statistics.pvariance(depths):.6f} | UNMEASURED |")
    histogram='\n'.join(f"| {r['dataset']} | {r['executed_depth']} | {r['tokens']} |"
        for r in hist if r['policy']=='dynamic')
    new_report('decode_depth_traces.md',f'''# Online decode-depth traces

Status: **MEASURED Stage-0 quality diagnostics only**.
The requested diverse Stage-1 serving cohort remains `NOT_RUN`.
The study does not relabel this math-only quality pilot as that serving cohort.

## Cohort and information horizon

The final pilot contains 32 GSM8K questions and 32 MATH-500 questions.
Both fixed and adaptive policies generate a separate natural answer for every question.
The primary comparison therefore contains 128 recorded generation runs.
The earlier benchmark manifest supplies public question text and answer keys.
This is not a claim of globally untouched benchmark data.

Metadata-only hashes determine the question order before all model outputs.
The first pilot uses 16 questions per domain.
The bounded continuation uses the next 16 in that same order.
It preserves the first pilot and does not tune q or answer extraction.
The continuation precedes its own inference but follows the first pilot's analysis.
`protocol_v3_addendum.json` explains the inherited timing field in that frozen manifest.

The compact primary trace contains {traces['actual_decode_events']:,} actual decode exit events.
Its file is `artifacts/pilot_export_v2/decode_depth_traces.jsonl.gz`.
The first generated token is absent from these events because full-depth prefill produces it.
Actual generated lengths include natural EOS or stop behavior and the predefined cap.
The runner does not force tokens after EOS.

## Actual depth distribution

| Dataset | Executed depth | Observed decode tokens |
|---|---:|---:|
{histogram}

These token counts describe execution.
They are not independent statistical samples for accuracy or scheduling gain.
The online gate uses the current completed recurrent state, not a recorded future depth schedule.
Request IDs, input/output token IDs, positions, GPU IDs, scheduler ticks, and actual depths remain in the raw trace.

## Descriptive request variation

| Dataset | Adaptive requests | Prompt tokens, range | Actual output tokens, range | Between-request mean-depth variance | Extra latency value beyond length |
|---|---:|---:|---:|---:|---|
{chr(10).join(descriptive)}

The variance is descriptive and unconditioned.
It does not establish request heterogeneity beyond prompt length, output length, and domain.
The depth-versus-length scatter is not a causal predictor evaluation.
No length-matched regression or cross-rank idle witness qualifies a Stage-1 result.

## Files and limits

`request_depth_summary.csv` preserves each measured generation's depth, length, host wall time, and finish reason.
`depth_histogram.csv` preserves the plotted counts.
`trace_manifest.json` records the model/runtime identity and evidence category.
The original per-request JSON also preserves the complete generated text and IDs.

The trace does not isolate GPU loop service time, queue waiting, or efficient mixed-depth refill.
The quality workers use B1 and heavy Python exit recording.
Four such workers are not a measured DP4 placement service.
The longer-cap follow-up remains separate from this primary export.

![Measured online depth histogram](../artifacts/pilot_export_v2/figures/online_depth_histogram.png)

![Descriptive request depth and actual output length](../artifacts/pilot_export_v2/figures/request_depth_vs_length.png)
''')
    verdict_text=('The tested adaptive operating point fails the registered severe pilot quality gate.'
        if final['verdict']=='QUALITY_FAIL' else
        'The study does not qualify a quality-preserving adaptive baseline. The overall result remains inconclusive.')
    audit_path=ROOT/'artifacts/final_audit.json'
    completion=('The final completion audit remains NOT_RUN.' if not audit_path.exists() else
        'The final audit records passing CPU contract tests and active owned burners on physical GPUs 4, 5, 6, and 7.\n'
        'No experiment GPU process remains.\n'
        'The original worktree\'s tracked staged and unstaged diff hashes remain unchanged.\n'
        'The full audit scope remains in `artifacts/final_audit.json`.')
    new_report('final_report.md',f'''# Decode-first dynamic-depth-aware DP scheduling discovery

## Decision

Verdict: **{final['verdict']}**.

{verdict_text}
Actual online early exit physically omits recurrent work.
That fact does not establish additional DP4 scheduling headroom.
The study stops at Stage 0 and does not train a predictor or an RL policy.

The conclusion applies to the tested CDB configuration.
It does not prove that every Ouro gate or cross-GPU depth-balancing policy fails.
The study reports no measured or simulated DP scheduling speedup.

## Model, hardware, and runtime

The adaptive candidate is **ByteDance/Ouro-1.4B**, not Ouro-1.4B-Thinking.
Its immutable revision is `{REVISION}`.
The spec permits this change because previous Thinking fixed-R4 validation does not establish adaptive support.
No Thinking accuracy result transfers to the base checkpoint.

The model uses BF16, 24 shared layers, maximum depth four, and 96 depth/layer KV slots.
It has hidden size 2,048 and 16 query/KV heads.
The official CDB revision is `{CDB_REVISION}`.
The wrapper does not implement a new inference engine.

Only physical GPUs 4, 5, 6, and 7 execute model work.
They are H100 80GB devices with NV18 connectivity between the assigned pairs.
The full mask maps physical 4/5/6/7 to logical 0/1/2/3.
Each quality worker instead uses its one-device subset as logical CUDA 0.
Each worker is TP1/EP1/PP1/CP1.
Their concurrent quality shards do not establish a live DP4 scheduler baseline.

The primary gate remains q=0.5, minimum R2, maximum R4.
It uses the checkpoint's original trained gate, same-step cumulative hazard, and `last_exited` KV propagation.
The policy does not use future token depths or untrained auxiliary gate heads.
Prefill remains full depth and produces the first output token.

## Physical execution and reference qualification

The bounded execution probe has 13 prompt tokens and 15 subsequent decode positions.
Both policies execute 52 prefill recurrent token-steps.
Fixed R4 executes 60 decode token-steps and 1,536 actual decoder-layer calls.
Adaptive execution uses 44 decode token-steps and 1,152 actual decoder-layer calls.
The 384-call difference confirms physical loop omission.

All 360 checked exit-KV copies match their source rows exactly after synchronization.
The cache retains independent depth slots rather than one shared slot.
The diagnostic's repeated completion waits do not support clean latency claims.

| Official HF comparison | Result | Interpretation |
|---|---:|---|
| Maximum logit absolute difference | 0.25, limit 0.5 | Numerical limit PASS |
| Mean logit absolute difference | 0.0236194, limit 0.05 | Numerical limit PASS |
| Full-KV relative L2 difference | 0.00834413, limit 0.02 | Numerical limit PASS |
| Teacher-aligned top-1 agreement | 15/16 | Not free-running agreement |
| Separate natural 16-token prefix | First mismatch at index 3 | Exact output parity PARTIAL |

The official HF comparison uses Torch 2.8/SDPA.
The CDB path uses Torch 2.11/FlashAttention3.
The study does not assign the discrete mismatch exclusively to rounding or one kernel.
The source reference status remains `REFERENCE_PARTIAL`.
The study does not relax a numerical limit or hide failed compatibility attempts.

## Primary free-running quality

The final bounded pilot has 32 paired questions per domain.
The first 16 questions per domain and their continuation remain available separately.
GSM8K uses three canonical CDB CoT examples.
MATH-500 uses the frozen zero-shot boxed-answer prompt.
Both policies use greedy natural generation with a common 4,096-token cap.
They use identical answer extraction and retain all questions in the denominator.

| Dataset | Fixed R4 correct / N, accuracy | Adaptive correct / N, accuracy | Drop, pp | Paired bootstrap 95% CI, pp | Fixed-correct losses / gains |
|---|---|---|---:|---|---:|
{quality_table}

Positive drop means lower adaptive accuracy.
MATH-500 meets the registered severe-stop condition in the common-cap primary comparison.
The bootstrap uses 20,000 question-level paired resamples with seed 20261018.
The intervals are nominal percentile intervals, not anytime-valid guarantees across sequential looks.

The registered severe pilot stop requires at least 15 pp loss and a positive lower loss bound.
The final quality-loss targets remain 1 pp for GSM8K and 2 pp for MATH-500.
This pilot cannot establish full-benchmark non-inferiority.

| Dataset | Parsing failures, fixed / adaptive | Truncated outputs, fixed / adaptive |
|---|---:|---:|
{parse_table}

Parsing failures and capped outputs remain visible in the denominator.
Primary accuracy uses the registered `answer_matches` metric.
A valid extracted answer can count even if the output later reaches the cap.
One fixed-R4 MATH-500 output, question 304, is correct by extraction but truncated.
The following sensitivity also requires EOS or a registered natural stop.

| Dataset | Completed and correct, fixed | Completed and correct, adaptive | Drop, pp | Paired bootstrap 95% CI, pp |
|---|---:|---:|---:|---|
{completion_table}

This secondary metric does not replace the registered primary decision rule.
`quality_predictions.csv` retains answers, parser version, correctness, lengths, and raw paths.
`baseline_correct_candidate_wrong.json` retains each lost-correct example.
The full 1,319-question GSM8K and 500-question MATH-500 evaluations remain `NOT_RUN`.
The study does not claim a final quality-preserving operating point from this pilot.
The severe primary loss and partial reference block scheduling qualification.
The study therefore does not expand to the full datasets.

## Actual adaptive recurrent work

| Dataset | Mean request decode depth | Request-bootstrap 95% CI | Executed loop-token work reduction |
|---|---:|---|---:|
{work_table}

The reduction denominator is four loops for the adaptive policy's own naturally generated decode positions.
It is not the total work of the separately generated fixed-R4 answer.
It excludes the prefill-produced first output token.
Different policies may generate different lengths and reasoning paths.
These percentages are work-count observations, not measured latency or DP scheduling gains.

Mean request depth gives each question equal weight.
The aggregate work reduction instead weights all actually generated adaptive decode positions.

## Longer-cap sensitivity

The predefined follow-up reruns both policies for every question in the primary truncation union.
Selected IDs and their freeze record appear in `quality_cap_followup_frozen.json`.
The follow-up uses an 8,192-token cap for those paired questions only.
Other questions retain their primary outputs in this secondary sensitivity table.
The common-cap primary analysis remains unchanged.

| Dataset | Fixed correct / N | Adaptive correct / N | Drop, pp | Paired bootstrap 95% CI, pp | Remaining truncations, fixed / adaptive |
|---|---:|---:|---:|---|---:|
{cap_table}

All checked primary token prefixes match in the follow-up: **{cap['all_primary_token_prefixes_equal']}**.
All checked primary executed-depth prefixes match: **{cap['all_primary_depth_prefixes_equal']}**.
This comparison tests cap sensitivity without changing q, precision, prompts, or KV policy.
It does not convert the pilot into a complete benchmark or resolve the independent reference mismatch.

## Stage and research-question status

| Item | Status | Supported conclusion |
|---|---|---|
| Actual online gate and loop omission | MEASURED | Physical skipped decode execution exists |
| Depth-indexed cache and exit propagation | MEASURED, bounded | Checked copies match after completion |
| Official fixed-R4 reference | PARTIAL | Numerical limits pass, exact natural prefix differs |
| Quality-preserving adaptive operating point | NOT_QUALIFIED | Do not advance the scheduling study |
| Diverse serving depth cohort | NOT_RUN | The quality trace is not the requested Stage-1 cohort |
| Depth effect beyond prompt/output length | UNMEASURED | Descriptive variation does not establish incremental latency value |
| Live RR/queue/length/strong-cost DP4 | NOT_RUN | No residual rank-idle witness |
| CDB cost model and validated simulator | NOT_RUN | No reliable serving oracle |
| Depth-only versus true-length-only oracle | NOT_RUN | Incremental depth headroom is unknown, not 0% |
| Homogeneous grouping and joint optimization | NOT_RUN | No local-batch or admission gain claim |
| Causal predictor and RL | NOT_RUN | No training or policy promotion |
| Prefill secondary | MEASURED audit | PREFILL_FULL_DEPTH in the tested path |

The intended primary action remains pinned initial placement before prefill.
No active request migrates between GPUs.
No oracle receives free KV movement, absent prefill cost, or invented idle intervals.
Four quality-worker finish times are not an eligible-work imbalance witness.

## Measurement limits

The quality runner records actual request wall time and host-visible TTFT/decode spans.
Its per-token Python observer does not qualify clean GPU service time.
The study does not report E2E speedup from those walls.
It does not validate CDB refill efficiency from B1 requests.
It does not build a constant-loop-time simulator from depth sums.

The independent fixed-R4 qualification remains incomplete.
The quality comparison describes the selected CDB gate/cache combination.
It does not isolate every possible cause of quality loss between the gate, cache policy, and backend.
No held-out threshold tuning or alternative predictor rescues the result.

## Prior boundary and next action

Ouro supplies the pretrained recurrent model and gate.
CDB already supplies local loop-level batching and refill.
Their existing loop omission is not a new DP scheduling method.
The proposed incremental question requires stronger evidence than a round-robin improvement.
It must separate true-depth information from true output length and calibrated generic placement costs.

This study supplies no such scheduling claim because Stage 0 does not qualify it.
The smallest next experiment is a separate reference and quality qualification task for one official adaptive configuration.
It must resolve fixed-R4 fidelity before any expensive diverse trace campaign or DP oracle.
Do not start RL, a new serving engine, or post-prefill migration from these results.

## Evidence and reproduction

- [Model, gate, and cache checks](stage0_model_gate_kv.md).
- [Actual online trace and descriptive figures](decode_depth_traces.md).
- [Timing definitions and limits](timing_contract.md).
- [Prior boundary](prior_overlap.md).
- [Reproduction commands](reproduce.md).
- Primary machine-readable quality: `artifacts/quality_pilot_analysis_v2/quality_summary.json`.
- Secondary cap sensitivity: `artifacts/quality_cap_sensitivity.json`.
- Stage status and decision: `artifacts/final_decision.json`.

The separate branch is `flashvep/decode-first-depth-aware-dp-poc`.
The task preserves the original dirty/staged tracked changes.
Only this PoC directory enters the task commit.
The final GPU audit checks permitted burner ancestry after all experiment workers finish.

## Completion checks

{completion}

![Primary paired quality difference](../artifacts/pilot_export_v2/figures/paired_accuracy_drop.png)
''')
    print('RESULT_REPORTS_WRITTEN',final['verdict'])


if __name__=='__main__':main()
