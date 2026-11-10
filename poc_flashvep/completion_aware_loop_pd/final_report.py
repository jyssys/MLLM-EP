"""Write the final narrative from audited measurements and explicit limitations."""
from collections import Counter
from .common import *


def fmt(value):
    if value is None:return 'NOT_MEASURED'
    if isinstance(value,bool):return 'PASS' if value else 'FAIL'
    if isinstance(value,float):return f'{value:.3f}'
    return str(value)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])


def result_table(items):
    return table(['Model','Arm','Arrival seed','TTFT p95, ms','ITL p99, ms','Max ITL, ms',
        'TPOT p95, ms','E2E p95, ms','Requests/s','SLO-goodput','Offloaded requests','Full Loops','Decode screen'],
        [[r.get('model'),r.get('secondary_arm',r['name'].split('_')[2]),r.get('arrival_seed'),
            *[r.get(k) for k in ('TTFT_p95_ms','ITL_p99_ms','maximum_ITL_ms','TPOT_p95_ms','E2E_p95_ms',
                'completed_QPS','SLO_goodput','offloaded_requests','completed_full_prompt_loops','Decode_SLO_screen')]]
            if r.get('complete') else [r.get('model'),r.get('secondary_arm',r['name'].split('_')[2]),r.get('arrival_seed'),
                *([r.get('status','NOT_RUN')]*10)] for r in items])


def ratios_table(rows):
    chosen=[r for r in rows if r['arm']=='loop' and r['reference'] in ('native','token')
        and r['metric'] in ('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','SLO_goodput')]
    return table(['Model','Comparison','Metric','Paired starts','Geometric ratio','Start-bootstrap 95% CI'],
        [[r['model'],r['arm']+'/'+r['reference'],r['metric'],r['paired_starts'],r['geometric_ratio'],
            '['+', '.join(fmt(v) for v in r['bootstrap_95_CI'])+']' if r['bootstrap_95_CI'] else 'NOT_ESTABLISHED'] for r in chosen])


def main():
    configure()
    primary=json.loads((ROOT/'serving_results.json').read_text())
    recovery_all=json.loads((ROOT/'recovery_serving_results.json').read_text())
    recovery=[r for r in recovery_all if r['cohort']=='N128_CONFIRMATION']
    ratios=json.loads((ROOT/'recovery_paired_start_ratios.json').read_text())
    completion=json.loads((ROOT/'recovery_completion_audit.json').read_text())
    systems=json.loads((ROOT/'systems_accounting.json').read_text())
    by_name={r['name']:r for r in systems}
    coverage=[]
    for model in ('ouro','huginn'):
        for arm in ('native','loop','token'):
            chosen=[r for r in recovery if r.get('complete') and r['model']==model and r['secondary_arm']==arm]
            counts=Counter()
            for r in chosen:counts.update({int(k):v for k,v in r['accepted_loop_counts'].items()})
            coverage.append([model,arm,len(chosen),sum(r['count'] for r in chosen),
                sum(r['source_armed_requests'] for r in chosen),sum(r['offloaded_requests'] for r in chosen),
                sum(r['completed_full_prompt_loops'] for r in chosen),
                sum(r['actual_D_prompt_recurrent_rows'] for r in chosen),str(dict(sorted(counts.items()))),
                sum(r['completion_reservation_misses'] for r in chosen)])
    timing=[]
    confirmation_names=set(json.loads((ROOT/'recovery_plan.json').read_text())['jobs'])
    for model in ('ouro','huginn'):
        chosen=[r for r in completion if r['model']==model and not r['token_split'] and r['name'] in confirmation_names]
        if chosen:
            from .analyze import percentile
            timing.append([model,len(chosen),
                *[percentile([r['committed_to_first_output_host_ms'] for r in chosen],q) for q in (.5,.95,.99)],
                max(r['committed_to_first_output_host_ms'] for r in chosen),
                max((r['largest_inter_quantum_wait_ms'] or 0 for r in chosen),default=None)])
    wire=[]
    for r in recovery:
        if not r.get('complete') or r['secondary_arm']=='native':continue
        s=by_name[r['name']]
        wire.append([r['model'],r['secondary_arm'],r['arrival_seed'],s['migration_KV_bytes'],s['migration_hidden_bytes'],
            s['migration_fragments'],s['migration_descriptors'],s['commit_to_native_ACK_host_ms']['p95'],
            s['P_prompt_distinct_host_submissions']])
    zero=json.loads((ROOT/'zero_control_results.json').read_text())
    zero_ratios=json.loads((ROOT/'zero_control_ratios.json').read_text())
    matched=json.loads((ROOT/'matched_native_path_ratios.json').read_text())
    late=json.loads((ROOT/'late_ABA_results.json').read_text())
    text=f'''# Completion-aware Adaptive Loop-Split PD: implementation and validation

## 1. TO-BE outcome

The campaign reproduced part of the TO-BE execution mechanism, but not its common serving benefit.
It implemented real boundary negotiation and exact P-to-D continuation.
Recovery moved 6/1536 Ouro Loops and 853/12288 Huginn Loops.
The corresponding request counts were 3/384 and 142/384.
Both models passed all three absolute Decode screens.
Huginn did not preserve Native-relative Decode tails near the 5% target.
The provisioned Native path without any migration also outperformed the Loop path.
It also removed one cause of the old multi-second starvation.
However, successful state movement did not establish useful large-scale P/D load balancing.
Physical capacity remains a separate measurement uncertainty.

The primary and recovery experiments remain separate.
They use different policy revisions and, for Huginn, different KV pool sizes.
The tables below must not become one combined treatment estimate.

## 2. Implementation and previous failure

The source directory is `poc_flashvep/completion_aware_loop_pd/`.
The detailed analyses are `root_cause_report.md`, `reference_code_review.md`, and `implementation_design.md` in the evidence directory.
The optional worker extension preserves the external vLLM-RLT checkout.
It reuses the Native-path connector, exact state frontiers, NIXL, and existing model kernels.
No model parameter, precision, recurrent depth, or Attention approximation changed.

The old source armed only 8/384 Ouro requests and 15/384 Huginn requests.
False initial arming could permanently remove later boundary opportunities.
One accepted continuation could occupy the only slot for most later arrivals.
Persistent admission suspension also prevented execution of already committed work.
The frozen host gaps were 23.207262 s and 85.872055 s.
Those gaps were missing submissions, not measured GPU idle.

The redesign separates source monitoring, completion admission, and quantum execution.
It retries legal Loop boundaries after transient rejection.
It prices complete indivisible quanta, transfer, accepted work, and repeated Decode reserves.
Committed work can progress under live deadlines even after new admission stops.
Eligible owned work uses completion-deadline priority.
A miss stops new borrowing but does not discard state or duplicate computation.

The reservation is an empirical forecast, not a guaranteed future GPU timeslot.
The implementation does not provide exact D-to-P recovery after commit.
It does not guarantee starvation freedom under every future arrival trace.
The direct P-native versus D-completion counterfactual remains incomplete.

## 3. Exactness and ownership

Twenty-six new CPU tests passed.
Thirteen inherited frontier and epoch tests also passed.
The final source compile check passed.
N8 and N32 compare exact output, valid prompt KV hashes, hidden hashes, full recurrent work, and Native lifetimes.
Their reference is an identified frozen Native GPU reference, not a new Native run.
These checks establish internal exactness, not independent original-HF fidelity or task accuracy.

Recovery N32 Loop diagnostics selected 1 and 3 Loops for Ouro.
Huginn's historical-cut diagnostic selected 2, 4, and 8 Loops.
The expanded Huginn diagnostic selected 4, 8, 11, and 16 Loops.
That expanded cohort committed 13/32 requests and completed 157 full-prompt Loop equivalents.
The Scheduler made these decisions online.
They were not a preassigned forced-cut cohort.

N32 means simultaneous submission of 32 requests.
It does not mean 32 simultaneous resident GPU requests.
State diagnostics use observer settings and longer completion deadlines.
They are not clean Decode SLO evidence.

Marker-free N128 runs check exact output, full work, ACK-before-retirement, final credit release, and empty Native allocations.
They do not hash every intermediate tensor during clean timing.
The separate N8/N32 tests qualify those state paths.
GPU error injection, post-commit cancellation recovery, and reverse migration remain `NOT_MEASURED`.
CPU stale-epoch checks are not substitutes for those GPU fault tests.

## 4. Frozen workload and fairness

Ouro uses original BF16 R4, 4096/32, QPS 6.356, and slots 2/8.
Huginn uses original BF16 R32, 2048/16, QPS 1.971, and slots 2/8.
The scientific pairs are 0/1 and 4/5.
Only one pair runs in the clean timing lane.
All approved Burners stop during that lane.

All arms retain 128 distinct controlled ShareGPT/LongBench prompt IDs.
Prefix caching is disabled.
Output caps ignore EOS.
Arrivals are paired open-loop Poisson traces.
These overloaded finite cohorts are not sustainable-capacity measurements.
They are not workloads where every request satisfies TTFT SLO.

The original confirmation uses arrival seeds 2026110911/12/13.
Recovery selection uses 2026110920.
Recovery confirmation uses 2026110931/32/33.
The latter arrivals are held out from recovery calibration and selection.
The texts are not held out.

The frozen absolute Decode SLOs remain unchanged.

| Model | TTFT limit, ms | Request TPOT limit, ms | Additional-work TPOT p95 limit, ms | ITL p99 limit, ms |
| --- | ---: | ---: | ---: | ---: |
| Ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| Huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

Native-relative ITL p99 non-inferiority remains a separate 5% design target.
Passing an absolute limit does not prove that target.

## 5. Original primary confirmation, before recovery

{result_table(primary)}

Ouro Token start 3 did not run after an earlier tail failure.
Its second start was already in progress when the explicit stop record arrived.
The completed Token results also include cold measured graph captures.
They do not establish an intrinsic Loop advantage.

The primary Loop moved 3/1536 Ouro Loops and 22/12288 Huginn Loops.
The corresponding offloaded request counts were 2/384 and 5/384.
Ouro's TTFT p95 ratio was 1.2692, with a paired 95% interval of [1.0821, 1.4432].
Huginn's ratio was 0.9285, with [0.8524, 1.0902].
The Huginn interval does not establish improvement.

## 6. Recovery changes and matched controls

The recovery uses the same full Decode-cycle reserve for Token and Loop.
It protects the next cumulative request-average TPOT with already delivered token counts.
It includes known accepted continuations' future Decode load.
It does not know future request arrivals or output times.

The source filters clearly infeasible offers before a relay.
Loop requests retain later legal boundary opportunities.
Token chooses a feasible initial prefix and revalidates it at the actual boundary.
After a rejection, Token computes its remaining suffix rather than create many new small checkpoints.
This is a bounded single-offer Token adaptation, not the complete official FluidToken algorithm.
It can miss later Token opportunities.
The report does not claim an optimal Token-only Scheduler.

All recovery arms prohibit cold packed-prefill capture during measurement.
They use cached graphs or the qualified eager path.
The common cache budget and warm-up policy apply to Native, Token, and Loop.
This policy does not control every Decode graph in the runtime.
Graph selections and eager fallbacks remain visible in the audit.

All recovery Huginn arms use 24832 D KV blocks, rather than the primary pool's 23552 blocks.
The larger pool admits six requests instead of five, despite eight logical slots.
Startup requires at least 3 GiB of free HBM.
The matched Native control receives the same memory change.
A same-arrival Native pool comparison changed completed QPS by only about 0.3%.
It does not establish KV admission as the unique throughput bottleneck.

The expanded Huginn candidate set passed separate GPU exactness tests.
Its N32 selection passed the absolute Decode screen and reduced TTFT p95 from 4964.819 to 4557.848 ms versus historical-cut Loop.
That selection remained slower than Native and worsened Decode tails.
Only selection data determined the expanded candidate choice.
The confirmation policy then remained fixed.

## 7. Recovery marker-free N128 confirmation

{result_table(recovery)}

`Service TTFT` and exact GPU-start Queue Latency are `NOT_MEASURED` in marker-free confirmation.
They are not replaced by coordinator admission timestamps.
The report does not add component percentiles to create a total percentile.
Separate CUDA Event boundary diagnostics retain their own scope.

All p50/p95/p99 latency values, input/output token rates, offered/realized QPS, and attainment counts remain in `recovery_serving_results.csv`.
Maximum ITL and threshold exceedances remain separate from the p99 screen.
Completed requests/s and goodput use the full cohort drain window.

## 8. Independent paired-start uncertainty

{ratios_table(ratios)}

Ratios use paired start-level metrics, not independent-token samples.
Three starts provide limited uncertainty coverage.
Missing starts do not become zero values.
A Token arm with no actual offload is a Native-fallback control, not evidence of Token redistribution benefit.

### Native-path zero-offload control

The original Native arm uses its original connector configuration.
Loop and Token enable Native-path continuation buffers and the qualified packed wire format.
This implementation difference can affect even a request with no offload.
Later controls enable that path without boundary negotiation or computation movement.
They pass separate N8/N32 state qualification before clean N128 execution.
Their later execution order remains a temporal limitation.
They do not provide a new randomized policy comparison.

{result_table(zero)}

{table(['Model','Metric','Zero / original Native','Paired starts','95% interval'],
    [[r['model'],r['metric'],r['geometric_ratio'],r['paired_starts'],
      '['+', '.join(fmt(v) for v in r['bootstrap_95_CI'])+']' if r['bootstrap_95_CI'] else 'NOT_ESTABLISHED']
        for r in zero_ratios if r['metric'] in ('TTFT_p95_ms','ITL_p99_ms','completed_QPS','SLO_goodput')])}

This control separates a provisioned Native path from the monitored Loop fallback path.
It does not prove that every enabled zero-offload Loop request has Native cost.

### Matched Native-path comparison

{table(['Model','Comparison','Metric','Geometric ratio','95% interval'],
    [[r['model'],r['arm']+'/zero',r['metric'],r['geometric_ratio'],
      '['+', '.join(fmt(v) for v in r['bootstrap_95_CI'])+']' if r['bootstrap_95_CI'] else 'NOT_ESTABLISHED']
        for r in matched if r['arm']=='loop' and r['metric'] in ('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','completed_QPS','SLO_goodput')])}

The later A/B/A control uses the same Huginn arrival seed throughout.
Its order is original Native, provisioned zero-offload, original Native.
It tests a simple temporal-drift explanation for the combined path difference.
It does not isolate wire packing, buffer provisioning, or scheduling as the unique cause.

{result_table(late['runs'])}

The complete A/B/A ratios remain in `late_ABA_results.json`.
The B path reduced TTFT p95 by 36.686% versus the mean of the two A controls.
Its ITL p99 was 14.546% lower.
The A controls themselves differed by 9.493% in TTFT p95.
One A control also had a 621.386 ms Maximum ITL.
These observations support a combined path effect without claiming stable Maximum ITL or a unique component cause.

## 9. Actual completed offloading

{table(['Model','Arm','Starts','Requests','Source armed','Actual offloaded requests','Full Loops','D recurrent rows','Loop-count distribution','Logged Prefill-phase misses'],coverage)}

Token recurrent rows are not reported as completed full-prompt Loops.
Each Loop count requires all prompt tokens at that depth.
Recurrent rows are structural work accounting, not FLOPs.
Attention work, Prelude/Coda, and tensor shapes differ between Token and Loop.
Actual migrated FLOPs remain `NOT_MEASURED`.

Huginn exceeded the original 500-Loop exploration target.
Ouro did not reach that target.
Huginn completed 490, 333, and 30 Loops in its three starts.
Its larger follow-up target of 4000 Loops was not reached.
The 853 completed Loops represent 6.942% of Huginn's total prompt recurrent Loops.
Ouro's 6 completed Loops represent 0.391%.
These fractions do not equal FLOP fractions.

Each Ouro N128 start accepted only one request.
Its selected counts were 1, 3, and 2 Loops across independent starts.
Within-run mixed-cut correctness passed in N32, not in these sparse Ouro N128 starts.
Huginn's N128 starts did contain actual online mixed cuts and Native fallbacks.

## 10. Completion latency and transfer

{table(['Model','Loop loans','Commit-to-output p50, ms','p95, ms','p99, ms','Max, ms','Max inter-quantum host wait, ms'],timing)}

These are host observations.
They include transfer, import, scheduling, computation, and output generation.
They are not CUDA kernel occupancy.
The complete prediction audit is `recovery_completion_audit.csv`.
Its direct first-output audit found four Huginn deadline overruns among 142 Loop loans.
The Runtime logged three misses in its receiving/Prefill phases.
The difference identifies an incomplete end-to-end completion guard.
The report does not equate a missing Runtime miss event with an on-time first output.
The largest inter-quantum host wait was 124.874 ms for Huginn and 0.287 ms for Ouro.
The corresponding maximum commit-to-first-output intervals were 2413.390 ms and 181.087 ms.

{table(['Model','Arm','Arrival seed','Offload KV bytes','Hidden bytes','Fragments','Descriptors','Commit-to-ACK p95 host ms','P host submissions'],wire)}

Payload bytes count successful Native transport regions for committed offload seeds.
They exclude protocol headers and control messages.
Native PD also transfers KV.
The offload payload is not entirely incremental communication over a zero-transfer baseline.
Commit-to-ACK is not DMA latency.
It includes source preparation, transfer progress, destination processing, and ACK observation.

One primary Huginn loan predicted 459.735 ms but completed 1277.638 ms after commit.
Its import-to-output interval was 404.793 ms.
The roughly 873 ms before first continuation submission was therefore not recurrent kernel execution.
The cut moved 4930928640 KV bytes and 43253760 hidden bytes.
It used 75 fragments and 7243 descriptors.
The existing dedicated copy stream was already enabled.
The campaign did not isolate a unique cause for the remaining handoff delay.

## 11. P/D execution and remaining overhead

The source must expose a completed boundary before it can negotiate ownership.
The current atomic graph does not expose earlier hidden state retrospectively.
Checkpoint-capable zero-offload requests therefore still pay extra submission and boundary costs.
The feature-disabled path remains Native Atomic PD.
This distinction prevents a claim of free zero-offload execution.

In the first primary Ouro start, Native recorded 128 source prompt spans.
The new Loop path recorded 510 spans.
Huginn recorded 128 versus 634 spans.
These are host request-span observations, not additional recurrent work.
Full-work checks show no duplicate or missing recurrent rows.

`systems_accounting.csv` reports source submissions, queue samples, graph selections, wire bytes, ACK intervals, and HBM inventories.
Exclusive P compute saved, exclusive D compute added, and GPU critical-path saving remain `NOT_MEASURED` in clean confirmation.
Serial calibration and structural rows cannot supply those missing online quantities.

## 12. Physical headroom and target scale

The separate Oracle report is `ouro_huginn_pd_idle_slack_loop_offloading_oracle_final_summary.md`.
Full D-only captures completed but failed the 5% interference gate.
Both-stage captures also failed progress.
Their exact interaction remains `ROOT_CAUSE_INCONCLUSIVE`.
The campaign does not treat those traces as clean Native physical capacity.

Diagnostic D idle was 32.492% for Ouro and 57.089% for Huginn.
Those values describe perturbed executions only.
Clean Native D idle, Decode-safe capacity, and completion-feasible capacity remain `NOT_MEASURED`.
This limitation prevents attribution of sparse acceptance solely to absent physical slack.

The historical cut ceilings across 384 requests are 1152 Ouro Loops and 3072 Huginn Loops.
The targets are 500 and 4000 Loops.
The historical Huginn set cannot reach its target structurally.
The newly qualified R16 frontier raises its structural ceiling to 6144 Loops.
Neither ceiling is measured serving capacity.
The campaign does not claim either target achieved.

## 13. Figures and interactive comparison

The figure manifest is `figure_manifest.json`.
The primary comparison is `figures/primary_serving_comparison.png`.
The matched recovery comparison is `figures/recovery_serving_comparison.png`.
The actual budget/cut scatter is `figures/recovery_budget_vs_actual_loop_count.png`.
The prediction-error plot is `figures/recovery_completion_prediction_error.png`.
The serial cost plot is `figures/full_prompt_continuation_costs.png`.
The intrusive idle histogram marks its failed gate explicitly.

`figures/native_old_new_observed_timeline.html` compares the same primary arrival seed.
`figures/recovery_native_zero_loop_token_timeline.html` compares the same recovery arrival seed.
Both timelines distinguish source work, destination work, handoff, and output delivery.
Their bars are host enclosures.
Blank space is not measured GPU idle.
Separate GPU-clock diagrams also identify enclosures rather than exclusive kernels.

## 14. Reference reuse and novelty

[FluidPD](https://arxiv.org/html/2610.06917v1) motivates pressure, KV-aware admission, and suffix completion checks.
The inspected sources did not establish an official public implementation for reuse.
Our Token arm remains a FluidToken-style adaptation.
It is not the original FluidPD system.

The reviewed [SGLang decode path](https://github.com/sgl-project/sglang/blob/fe30d1e0fd4ede090b48a4147d23e23571b327f2/python/sglang/srt/disaggregation/decode.py) separates preallocation, transfer pending state, and runnable Decode.
The redesign uses that lifecycle principle without transplanting incompatible request or KV objects.
The local `ref_loopserve` review covered its scheduler, recurrent planner, ownership, graph controller, and grouped execution.
Its Early Exit path and incompatible KV semantics remain disabled.

Generic logical-layer scheduling can reproduce the same recurrent boundaries and state transfers.
This case is `EQUIVALENT_MECHANISM`.
The campaign does not establish a Loop-specific novelty claim from boundary names or recurrent-row counts.
It also does not combine generic deflection and depth scheduling into an unsupported new-method claim.

## 15. Failure scope and next decision

The desired useful large-scale TO-BE remains unconfirmed against the provisioned Native control.
The current implementation still has checkpoint overhead and imperfect completion predictions.
No result establishes Loop superiority over a strong optimal Token-only Scheduler.
The physical-capacity verdict remains `INCONCLUSIVE_MEASUREMENT`.
The implementation verdict is `NO_GO_NET_BENEFIT` for Ouro and Huginn under these tested conditions.
This verdict does not mean `NO_GO_PHYSICAL_HEADROOM`.
The physical Oracle remains `INCONCLUSIVE_MEASUREMENT`.

| Model | Tested serving verdict | Physical-capacity verdict | Main reason |
| --- | --- | --- | --- |
| Ouro | NO_GO_NET_BENEFIT | INCONCLUSIVE_MEASUREMENT | Only 6 Loops moved. TTFT and goodput lose to the provisioned zero-offload path. |
| Huginn | NO_GO_NET_BENEFIT | INCONCLUSIVE_MEASUREMENT | Real 853-Loop movement does not beat the provisioned path. Decode p99 also rises. |

Huginn improved finite-cohort joint goodput versus the original Native path.
Its mean paired gain was 0.066788 requests/s, with a 95% interval of [0.022551, 0.119581].
One original Native start had zero goodput.
The analysis therefore reports a difference rather than an undefined geometric ratio.
The gain did not establish superiority over the provisioned zero-offload path.
Huginn's ITL p99 ratio to original Native was 1.1773, with [1.0226, 1.3061].
It therefore did not establish the requested Native-relative tail protection.

The common Loop-specific research verdict is no GO from this campaign.
The exact continuation mechanism works.
The measured Runtime-path improvement without migration is the stronger practical result.
It should become the no-offload control before further Dynamic Scheduler work.

The run inventory records complete, failed, and not-run cells with source and configuration hashes.
`configuration_inventory.csv` also identifies prepared configurations that never executed.
One Huginn seed-31 Loop start overlapped a 4.79-second host audit near measurement start.
The campaign excluded that start before inspecting its performance.
It retained the raw files and repeated the same policy and arrivals in a new namespace.
The clean replacement contributes to the final table.
The inherited control auditor initially failed on a null partial-cut field.
The Runtime correctly resolves that field to full depth.
A scoped analysis view applies the same resolution without changing the immutable configuration.
The retained GPU outputs then passed the unchanged state and lifetime gates.
Invalid full traces, earlier failed qualifications, and stopped Token starts remain preserved.
Low/medium load recovery comparisons, a clean N512 persistence test, and GPU fault-recovery qualification remain unexecuted.
A separate adaptive hybrid logical-frontier comparison also remains unexecuted.
Token-tiled execution of an accepted Loop suffix is not that separate hybrid comparison.
No sustainable-capacity or general-text claim follows from N128 finite cohorts.

The next step is not a larger Dynamic campaign by default.
It first requires a lower-cost checkpoint path or a demonstrated completion advantage over continued P execution.
It also requires reliable physical measurement if the claim concerns unused hardware capacity.
Increasing continuation slots alone does not resolve either requirement.

## 16. Reproduction, publication, and resource restoration

The evidence directory is `ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation/`.
It records source snapshots, configuration hashes, GPU maps, arrival seeds, request inventories, and qualification results.
The lossless bundle preserves raw evidence without changing the original files.
`evidence_bundle --verify` checks reconstructed file hashes.

The publication branch is `{BRANCH}`.
The publication parent is `{BASE}`.
The publication uses a temporary Git index and leaves the active checkout unchanged.
The staged-entry hash and external runtime diff hash receive separate preservation checks.
The publication receipt records the remote verification.
The final Burner receipt checks actual engine ownership, generation progress, and utilization on all eight GPUs.
The live Burner processes continue after the final response.
'''
    path=REPO/'poc_flashvep/reports/ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation_final_summary.md'
    path.write_text(text)
    print(str(path))


if __name__=='__main__':
    main()
