# Dense AR Decode-safe Elastic Layer PD: experiment specification

## Goal and immutable anchors

First qualify standard Continuous Colocated DP2.
Then test smaller safe layer cuts and matched migration controls.
Do not assume that Dynamic wins or that Colocated harms Decode.
The publication parent is `96414d7baac14dea2d697198b2c595850ae83661`.
The branch is `flashvep/dense-ar-decode-safe-elastic-layer-pd-20261113`.
Use a new source directory and evidence namespace.
Preserve the existing checkout, staged index, external runtime, and prior evidence.

Use Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
Keep original BF16, exact Attention, all 36 layers, greedy sampling, and disabled thinking.
Use vLLM0.26.0+cu129, PyTorch2.11.0+cu129, and NIXL1.4.1.
Primary timing uses GPU0/1 and 42 GiB KV per GPU.
Disable prefix caching.
Do not quantize, skip layers, or change model semantics.

## Phase1: continuous correctness

Keep the independent full Native reference and previous token records.
First replay the archived mismatch.
Then reproduce the mismatch in bounded GPU runs.
Start with native Continuous Batching, Round-robin, and 8192 batched tokens.
Do not use a custom cohort barrier in the primary Colocated arm.
Check query coverage, positions, KV layout, mixed batches, Attention, graph mode, sampling, and identity separately.
Compare per-token logits and intermediate state when needed.
Test N8 before N32.
Use unchanged state tolerances `atol=0.02`, `rtol=0.01` where bitwise equality is unavailable.
Require exact greedy tokens.
Do not relax a failed gate or silently replace its reference.
Apply a necessary numerical runtime change to all compared methods.
Report its generic runtime effect separately from layer scheduling.
If no continuous arm qualifies, label the blocker explicitly.
Do not substitute a cohort arm as the strongest standard baseline.

## Diagnosis hypotheses

The archived Chunk512 replay mismatches8/64 requests after their first output token.
This observation motivates four falsifiable hypotheses.

1. Mixed Attention changes a BF16 reduction path when the global query length changes.
2. Chunk position or valid KV state differs from full Native state.
3. CUDA Graph or compilation changes the execution path.
4. Sampling or request lifecycle changes an otherwise identical output.

Change one diagnostic factor per run.
A direct Attention probe isolates a kernel property, not end-to-end model correctness.
Preserve every failed configuration and raw result.

## Phase2/3: revised selection and uncertainty

Audit each previous candidate and rejection reason before changing policy.
Evaluate Last2/4/8 even when Last12 fails.
Select by expected SLO-qualified completion benefit, not maximum transferred layers alone.
Keep Last0 when no candidate meets safety and benefit requirements.

Separate an admission completion budget from the immediate execution budget.
Use current and past state only.
Include known active/pending Decode, accepted continuations, KV credits, handoff, and host costs.
Do not read future arrival offsets inside the policy.
Do not treat `t2-t1` as a measured physical idle guarantee.
Use batch-aware measured costs where the shapes are legal.
Measure prediction errors on tuning data and freeze uncertainty bounds before confirmation.
Do not describe a small-sample p99 as WCET.
Roofline provides structural bounds, not online available capacity.

Choose a cut before submitting its P prefix.
Reserve generation-specific D credits before commit.
Keep Native fallback legal before commit.
Retain source buffers until ACK and destination buffers until CUDA consumption completes.
Execute each prefix and suffix once.
Do not assume kernel preemption or discard a committed prefix.

## Phase4/5: serving and matched controls

Qualify Last0/2/4/8/12, KV/Hidden/Residual, actual layer visits, Full36 Decode, and normal retirement.
Then run N64/N128 serving.
Compare Native, strongest qualified Continuous DP2, validation-selected Static, and Revised Dynamic.
Retain previous IDLE, DEADLINE, and COST as ablations where useful.
Compare Static Last12 at 31.25% and 50%.
Include controls with the same migrated request and completed layer budgets.
Freeze control budgets from tuning data for primary confirmation.
If exact realized-budget matching uses future outcomes, label it offline or exploratory.
Do not claim an online state advantage from an unmatched work amount.

Start with Input8000 and Output64/16/200.
Use measured capacities to select underload, transition, and overload QPS points.
Output32/128 and bursty arrivals are additional tests when useful.
Use the same prompt IDs, arrival trace, sampling, hardware, and SLO within each paired comparison.
Distinguish finite-cohort drain rates from sustainable capacity.
Report offered, observed arrival, submitted, and completed QPS separately.

## Data and SLO freeze

Keep previous Discovery64 as tuning data.
Treat the previous Held-out128 as previously viewed data, not a fresh test set.
Freeze new prompt IDs or clearly separated new arrival seeds before confirmation.
Preserve source document IDs, token lengths, hashes, and truncation rules.
Do not repeat meaningless filler tokens.
Use Ignore EOS for fixed-output tests.
Report Natural EOS separately if executed.

Keep the frozen SLO file unchanged.
Its SHA256 is `c8aa94213e4f4ce54821d01ba985a8e9e88a20566a3e8d1f3cc8060d1b3926d9`.
Report TTFT-priority, Balanced, and Decode-priority regimes.
Include first ITL and separate first/steady distributions.
Service TTFT remains `NOT_MEASURED` unless its required boundary exists.
Require both Native-relative95% confidence upper bounds at most 1.05 for Decode non-inferiority.
Use independent server starts as bootstrap units.
Do not replace independent starts with request or token counts.

## Figures and verdict

Produce QPS versus TTFT p95, TPOT p95, and output tokens/s for the four main arms.
Separate output lengths.
Show independent-start uncertainty and only qualified primary baselines.
Add ITL, goodput, cut distribution, prediction error, and timeline evidence where measured.
Distinguish kernel intervals, CUDA event enclosures, host observations, modeled budgets, and offline oracles.
Never draw unmeasured physical idle or synthetic speedup.

GO requires exact execution, meaningful TTFT benefit, Decode non-inferiority, matched-work benefit, and a qualified Continuous comparison.
Otherwise report the failed gate and tested envelope.
Do not claim structural impossibility from a runtime or measurement failure.

## Operation and publication

Use the shared quiet-lane lock for isolated timing.
Stop only approved owned burners.
Restore all 8 burners between measurement batches and after GPU work.
Check fresh generation and process ownership on every GPU before final handoff.
Give each GPU job startup, serving, and process-group timeouts.
Preserve immutable configurations, commands, source archives, hashes, GPU UUIDs, seeds, errors, and resume status.
Publish only this campaign through a private index.
Check Remote SHA after push.
