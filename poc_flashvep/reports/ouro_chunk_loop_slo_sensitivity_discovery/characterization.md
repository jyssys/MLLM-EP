# C/G characterization

## Definitions

C is configured prompt-token chunk capacity.
G is the number of prompt recurrent passes before a legal scheduling boundary.
N is client concurrency in the continuity cohorts.
All model tokens still execute R4.
G is not an early-exit setting.

The new continuity data contains 316 measured cohorts.
The complete 17-configuration grid covers P≈2K/8K and N4/8 with three cohorts per cell.
Eight shortlisted configurations receive ten cohorts at P≈2K/N4 and P≈8K/N8.
Those shortlisted comparisons span six process blocks.
Each block compares policies on the same GPU.
The historical prompt families provide continuity, not a newly held-out dataset.
P denotes the new long prompt, not every resident context.
Its actual length is 2,033 or 8,177 tokens.
Background prompts contain 288–385 tokens in the frozen continuity cohort.
For example, trial 2 contains 303/315/305/325/306/336/310-token background prompts.
The natural article/conversation lane uses its separately frozen prompt distribution.

## C-only: more boundaries can produce worse TBT

At P≈8K/N8, Atomic G4 issues 64/32/16/8 core-containing prefills for C128/256/512/1024.
This is an actual submission count, not a token-depth reduction.
All four settings issue four recurrent passes for every prompt token.

Smaller C does not monotonically reduce observed output-token gaps.
It increases the number of repeated core invocations and scheduling transitions.
The C128 long-TTFT mean is 8093.8 ms across three cohorts.
The C1024 mean is 971.3 ms across ten cohorts.
These have different cohort counts and are descriptive values.
Use `paired_comparisons.csv` for matched comparisons.

Repeated core entry can increase kernel-launch, metadata, and scheduling costs.
This study does not infer physical weight traffic from checkpoint size.
It does not measure achieved HBM bandwidth or GEMM FLOP efficiency from GPU utilization.
Separate heavy traces constrain the physical interpretation.

## The native fairness boundary is a recurrent batch

The source is [`SchedulingPolicy.record_batch`](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/scheduling_policy.py#L31).
The fairness counter increments for scheduled prefill phases.
A recurrent decode batch resets it.
The counter does not wait for a completed output token.
Static decode needs four recurrent passes before that token completes.

Consequently, one output-token interval can contain several prefills.
Reducing one prefill quantum does not guarantee a shorter complete token interval.
The measured descriptors expose these accumulated interruptions.
`token_prefill_exposure.csv` records their counts and host interval overlap.
Host overlap is not a GPU critical-path attribution.

Atomic includes its prelude inside a core-containing quantum.
Grouped/Split can submit a separate staged prelude.
The response surface reports both core-containing and all-prefill submission counts.
The per-token exposure analysis includes the separate prelude.
Do not compare only one policy's core count with another policy's total count.

## G-only: shorter severe stalls, more moderate stalls

At C1024/P≈8K/N8, Atomic and Grouped issue 8 and 16 core-containing prefills respectively.
Grouped also executes separate staged-prelude work.
The complete TTFT includes those boundary operations.

Atomic's mean cohort-p95 background ITL is 38.845 ms.
Grouped G2's mean is 130.377 ms.
Their mean cohort-p99 values are 388.869 ms and 189.721 ms respectively.
The paired p99 reduction is 51.212%, with 95% CI [45.036%, 59.925%].
This is a distributional trade-off, not an across-the-board latency gain.
The paired TTFT change does not exclude zero.
Its reduction estimate is −5.836%, with 95% CI [−15.182%, 9.141%].
The paired drained-cohort reduction is −2.159%, with 95% CI [−5.375%, 3.174%].

P95 and P99 inspect different parts of the same distribution.
Few large stalls can fall beyond P95 but affect P99.
More frequent moderate stalls can move the P95 boundary upward.
FigureA and `ITL_stall_counts.csv` show the complete measured distribution and event denominators.
No token interval serves as an independent statistical sample.

## Staging is a separate mechanism

Staged G4 keeps four-loop groups while using the staged path.
At P≈8K/N8/C1024, its mean cohort-p99 is 273.991 ms.
That differs materially from Atomic's 388.869 ms despite the same G.
Thus the entire Atomic-to-G2 tail difference cannot be attributed solely to G.

G2 reduces p99 by 30.756% relative to Staged G4.
The 95% paired interval is [21.428%, 43.868%].
Their drained-cohort change remains uncertain: 0.447%, CI [−3.082%, 6.902%].
These controls separate an observed tail benefit from an unproved E2E gain.

At C1024, staged hidden allocation is 32 MiB for N8.
At C128/256/512, it is 4/8/16 MiB.
That buffer is not the entire HBM or graph-workspace overhead.
The common physical KV reserve is 54 GiB.
Peak allocated memory remains a separate measured field.

## C×G interaction and existing Mixed

C×G alone does not describe client-visible latency.
The number of chunks, prelude boundaries, context lengths, and batch composition also matter.
Equal C×G values do not imply equal total submission counts or TTFT.
The separate GPU quantum trace does not automatically define a validated online cost predictor.

Existing Mixed is a strong baseline, not merely another pure-G setting.
It combines compatible prefill and decode rows.
Its actual prompt rows can be below configured C under the shared token budget.
The trace retains those actual counts.

Mixed C512 has mean TTFT 1715.895 ms at P≈8K/N8.
Its 110.277 ms value is the mean cohort-p99.
G2 C1024 has TTFT 1027.939 ms and p99 189.721 ms in this comparison.
G2 favors prompt responsiveness in this pair.
Mixed favors decode-tail latency.
Neither point establishes universal superiority or a new joint controller.

## Scope of the conclusion

The measured C/G effects are specific to this pinned RLT execution path.
They do not establish the same gains in optimized native-vLLM.
Existing Atomic, Split, and Mixed mechanisms are prior functionality.
The old G2 adapter is not a new fundamental parallelism technique.
Full-answer quality, online adaptive controls, and qualified capacity require separate evidence.
