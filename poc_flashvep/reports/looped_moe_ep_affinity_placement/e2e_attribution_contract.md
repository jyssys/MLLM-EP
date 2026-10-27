# Full E2E attribution in percentages

## Requested addition

The user requested an additive E2E breakdown on 2026-10-26.
The breakdown must show components as percentages of the whole measured E2E interval.
It must support further bottleneck research.
It must not substitute a sum of kernel service times for E2E latency.

## Primary measurement

The primary observer uses CUDA events around actual execution regions.
It observes a complete 32-token greedy rollout.
The rollout contains prefill, every subsequent autoregressive forward, sampling, and output completion.
The model still recomputes each full prefix.
No per-forward component is multiplied by 32 to manufacture an E2E breakdown.

The measurement uses the output rank's critical CUDA stream.
The observer records the entire rollout's CUDA-event span and CPU wall duration.
It also records an independently timed clean rollout on the same input.
The report always identifies the observer's own denominator.
This observer starts before input cloning.
The earlier hardware rollout starts after input cloning.
The report keeps their measured E2E values separate.

The initial target covers the three models at native, EP4, and EP8.
Each eligible configuration expands to B1/B2/B4/B8/B16 and prefix 128/512/2,048.
The generic source-owner control also receives selected B1/B16 and prefix-128/2,048 attribution checks.
The pilot uses three traced rollouts and three clean rollouts.
Other matrix cells initially use one traced rollout and three clean rollouts.
A single trace does not receive an invented confidence interval.

## Additive component schema

The primary table contains these regions:

1. Input staging.
2. EP metadata and count synchronization.
3. Dispatch, including rank waiting.
4. Expert execution and local tensor layout.
5. Combine, including rank waiting.
6. Reweighting and restoration, including rank waiting.
7. Attention, including QKV and RoPE.
8. Router execution.
9. Block normalization, residual, and auxiliary work where separately identified.
10. LM head and embedding where separately identified.
11. Sampling, output, and completion synchronization.
12. Other model and scheduling work.
13. Unattributed time.

The raw table retains finer regions where the timing gate permits them.
The coarse observer combines adjacent regions to reduce marker overhead.
The common primary schema keeps those coarse and fine runs comparable.
A zero attributed region does not prove that the model performs no corresponding arithmetic.
An enclosing region can contain work that lacks a separate marker.

## Exclusive accounting

Model regions contain attention, router, and expert regions.
Those nested regions must not count twice.
The interval sweep selects the innermost active region at every point in time.
Explicit call-nesting depth resolves equal GPU timestamps.
The remaining intervals enter `Unattributed`.
The additive component durations equal the complete observed E2E span.
Their unrounded percentages sum to 100%.

The first completed output defines a separate phase boundary.
The phase table separates prefill-to-first-output from the remaining output tokens.
Each phase row identifies both denominators: its own phase and the whole E2E interval.
The first phase includes sampling. It is not a pure prefill-kernel measurement.

The report does not add percentages from different ranks.
The report does not add component maxima from different ranks into a fictitious timeline.
Separate rank traces remain useful for straggler analysis.

## Interpretation limits

These values describe API execution spans on the endpoint's critical stream.
An expert span includes kernel execution and possible launch gaps.
It is not a pure GEMM service-time measurement.
An attention span likewise includes the native attention execution path and its submission gaps.

NCCL regions include communication and rank waiting.
A late remote expert can increase the endpoint's combine wait.
The report must not call that entire interval network transfer time.
The report must not infer removable bandwidth cost from that percentage alone.

The previous CUPTI observers supply separate kernel-level evidence.
They do not provide a clean full-rollout E2E percentage partition.
The report keeps kernel service time, API spans, and clean latency separate.

## Timing gates

The first pilot checks mean observer overhead against a 5% limit.
It also checks CUDA-event span versus CPU wall duration against a 1% alignment limit.
Both limits were selected before the first attribution pilot.
The coarse observer is the fallback when fine markers exceed the overhead limit.

A passing gate indicates low overhead.
It does not prove an unbiased decomposition of the uninstrumented run.
The report therefore always calls the shares observed E2E shares.
It shows clean latency and observer latency in separate columns.

A failed gate blocks a clean-E2E interpretation.
The report can retain the observer result with an explicit timing warning.
It must not rescale observer components to resemble measured clean components.
Failed fine pilots and passing coarse pilots remain separate in the CSV.
The summary selects one observer variant for each model and workload.
It does not average incompatible observer variants.

## Output files

- `e2e_breakdown.md` presents completed percentages and timing warnings.
- `e2e_component_percentages.csv` contains additive component summaries.
- `e2e_component_percentages_raw.csv` retains individual traces and evidence paths.
- `e2e_component_phase_percentages.csv` contains phase-specific and whole-E2E percentages.
- `figures/e2e_component_percentages.svg` compares the B1, prefix-128 observer results.

The campaign refreshes these derived artifacts after each completed attribution run.
Before the first completed run, the summary explicitly reports `PENDING_MEASUREMENT`.

## Correctness and isolation

The marker patch changes no tensor expression, weight, router score, or loop count.
Every traced rollout must match the uninstrumented greedy token IDs.
Marker regions never read a future router or hidden state.
No proposed loop-aware runtime enters this measurement.

The attribution campaign waits until the existing GPU campaigns complete.
It does not interrupt active quality or performance experiments.
Only verified owned burners can yield devices.
Unused authorized devices retain owned burners.
