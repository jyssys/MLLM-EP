# Two-GPU Loop-versus-Token PD: Correctness and Protocol Contract

Status: FINITE_MIGRATION_GATES_PASS. Final controlled-serving results follow below.
Campaign start: 2026-10-30 17:37 UTC, or 2026-10-31 02:37 KST.
The hard closeout deadline is 2026-10-31 05:37 UTC, or 14:37 KST.
No long GPU job may start after 04:37 UTC.

## Scope and provenance

The campaign uses Ouro Base 1.4B R4 and Huginn-0125 R32.
It retains the checkpoint revisions from release `6c0a597c4ee58d54bde738307e05ac39f008fec0`.
It retains the `down_norm` and `down_norm_adapter` numerical corrections.
BF16 weights and KV remain unchanged.
Every prompt and generated token must execute its full fixed depth.
Original-HF equivalence and downstream accuracy remain separate, unproven claims.

The branch is `flashvep/adaptive-loop-token-pd-2gpu-w1-w4`.
The installed RLT and all inherited PoC files remain unchanged.
Only process-local overlays and independent evidence paths are new.
`adaptive_loop_token_pd_2gpu/configs/initial_contract.json` records source and index hashes.
Every job archives its actual overlay source before startup.

## Partition mechanisms

Loop migration reuses the qualified absolute prompt-state staging mechanism.
P completes a committed recurrent cut across every prompt token.
D receives the valid depth-specific KV and all intermediate carrier rows.
Huginn's carrier includes injection state.

Token migration stops P at a legal token-prefix boundary.
P completes every recurrent loop for that prefix.
The prefix includes valid Prelude, recurrent, and Coda KV where applicable.
D resumes the suffix at Prelude with full-depth prefix KV already valid.
The final prompt Coda row remains excluded until its normal execution.
An intermediate token-prefix carrier is transported but does not replace suffix Prelude.
For Huginn, migration also carries the canonical initial states for every suffix token.
The qualified engine initializes these states once for the entire prompt.
Reinitializing only the suffix changes recurrent KV despite exact transfer of the prefix.
The receiver installs the original suffix states and the matching RNG position.
CUDA events order those writes before Prelude and recurrent consumers.

Both connectors retain source buffers until transfer acknowledgement.
An online boundary is provisional before export.
P pauses at that boundary while the coordinator checks current D admission again.
If D refuses, P retains ownership and continues to full Prefill.
No remote snapshot writes occur before this decision.
The committed import plan is sent before D activates the transferred request.
The receiver rejects invalid transfer generations and waits for every completion.

## Online information

The initial rule uses observed P backlog, active Decode requests, and Decode age.
It also uses D's reserved KV credit and a disjoint-calibration cost coefficient.
The same rule and information apply to Loop and Token decisions.
It does not use future arrivals, actual future output length, or held-out outcomes.
The remaining-computation estimate is deliberately conservative.
It is not an SLO guarantee or a validated GPU critical-path predictor.

## Scheduling and transport controls

All main arms disable shared-GEMM Prefill/Decode co-batching.
DP uses two independent colocated replicas with atomic full-prompt Prefill.
DP_Chunked uses two replicas with separate-phase chunk interleaving.
The atomic arm uses a 4,096-token budget and no wavefront Prefill.
Chunked and partitioned arms initially use a common 256-token chunk and budget.
Kernel tuning is limited. These arms do not represent fully optimized production Mixed serving.

PD_G1 retains native full-depth token-chunk streaming.
Loop and Token migration use acknowledged snapshots.
The full-cut snapshot is a separate protocol control, not ordinary PD_G1.
Any unmatched protocol comparison must retain that limitation.

The common KV pool uses 8,192 blocks for Ouro and 18,432 for Huginn.
Both stages and both DP replicas use the same per-model pool configuration.
Long Huginn requests can exhaust KV credit before the four compute slots are occupied.
Admission therefore checks actual block credit rather than assuming four active requests.

## Initial failures and regression tests

The first Token prototype left Prelude ready after its prefix boundary.
The scheduler could select that phase without producing a legal span.
The exact failure was `scheduler made no progress`.
A CPU regression test failed before the boundary-window correction.

The next prototype initialized full-prefix KV validity twice.
The exact failure was `KV prefix was already initialized`.
A second CPU regression test failed before the single-initialization correction.
An online Token fallback also needs a new Prelude queue entry for the remaining suffix.
The corresponding regression test failed before the queue transition correction.
Huginn initially failed internal state parity despite matching eight short outputs.
The first differing KV plane was recurrent depth zero, physical layer two.
Both Prelude KV layers remained exact.
The missing suffix initialization state explained that pattern.
After its transfer was added, the eight-request Huginn Token gate passed output and state checks.
All original failed job namespaces and logs remain preserved.
Passing a CPU regression test does not replace GPU migration qualification.

## Acceptance sequence

1. Compare eight requests at prompt lengths 128, 512, and 1,536.
2. Check exact output IDs, valid Prompt KV, and first-Coda hidden state.
3. Check producer/receiver byte hashes and fixed-depth issued row counts.
4. Change chunk size, request order, batch capacity, and arrival seed.
5. Complete independent 32-request qualification for each eligible arm.
6. Verify zero, intermediate, and full partition endpoints.
7. Expand only qualified arms to real full-text W1–W4 serving.

Heavy byte and state tracing remains separate from clean latency.
Early diagnostic jobs can run on independent GPU pairs beside owned burners.
Their latency is not clean performance evidence.
Clean serving runs lease all eight devices and stop only verified owned burners.
One two-GPU server configuration runs at a time in that lane.
The resource manager restores approved burners after each completed lease.

## Workloads and statistical limits

The main manifests contain complete ShareGPT dialogue prompts.
They do not truncate prompt IDs to produce the target length bins.
Calibration and test prompt hashes are disjoint.
No original Azure prompt content is claimed.
The diagnostic gates alone reuse frozen token-prefix fixtures.

Main outputs use `ignore_eos=True` and declared controlled targets.
These runs are not Natural-EOS evaluation.
Huginn's prompt plus output target never exceeds 4,096 tokens.
W1's 32-request integer bin counts are 11, 13, and 8.
The corresponding percentages are 34.375%, 40.625%, and 25%.

One 24–32-request start cannot establish stable capacity or precise population p99.
Timeouts and request failures must remain in SLO denominators.
Independent starts, not individual tokens, define a qualifying paired uncertainty estimate.
Missing repeats or rate points must remain explicit.

## Finite acceptance results

Both models passed the independent 32-request qualification for all seven arms.
Each candidate arm has 32 exact output and Prompt-state comparisons against PD_G1.
The six non-reference arms therefore have 192/192 candidate comparisons per model.
The native reference's self-comparisons are not included in that denominator.
Every arm also matched its prescribed recurrent work accounting.

| Model | Non-reference 32-request output comparisons | Prompt KV and hidden comparisons | Zero/full endpoints | Real long-prompt migration |
|---|---:|---:|---|---|
| Ouro R4 | 192/192 exact | 192/192 exact | PASS | Loop and Token PASS |
| Huginn R32 | 192/192 exact | 192/192 exact | PASS | Loop and Token PASS |

Evidence class: actual GPU correctness execution with heavy tracing.
These requests use a finite prompt fixture and repeated prompt content.
They are not 192 distinct benchmark problems.
The 32-request fixture changes seed, request order, capacity, and chunk size.
It uses eight frozen prompt fixtures at lengths 128, 512, and 1,536.
The long gate adds eight complete dialogue prompts, including the 2.8–3.5K stratum.
Each long candidate has eight exact output, KV, and hidden-state comparisons.
Each nonzero migration check includes producer/receiver byte identity.
The byte check includes Huginn's canonical suffix initialization payload.
Zero-cut D-local requests do not transfer bytes.

Five failed development job namespaces remain preserved.
Completed prototypes with rejected internal state parity also remain preserved.
The acceptance results refer to the corrected implementations, not those prototypes.
UCX context-destruction warnings remain in completed-run logs.
The acceptance tests use actual payload, state, output, and work checks independently of those warnings.

The serving harness checks full controlled output sequences against the matched PD_G1 result.
An unexplained output mismatch stops that candidate cell and forbids further serving expansion.
Controlled-generation parity is summarized below for the cells that actually completed.
The finite gates do not establish original-HF equivalence or downstream accuracy.

## Final controlled-serving status

| Model | Accepted candidate cells | Matched controlled outputs | Incomplete/rejected candidate cells | Model verdict |
| --- | --- | --- | --- | --- |
| ouro | 24 | 672 | 0 | CONDITIONAL |
| huginn | 24 | 672 | 0 | CONDITIONAL |

Each accepted candidate sequence matches the same-start PD_G1 reference exactly.
These counts are request instances, not benchmark accuracy questions.
Clean serving does not repeat the heavy per-plane byte hashes on every request.
The finite byte/state gates, actual transfer completions, output checks, and work counters remain separate evidence.
Original-HF fidelity and downstream accuracy remain NOT_ESTABLISHED.

## Full-cut transport-only controls

Both cases keep all Prefill computation on P. They use the same W1 requests, targets, seed, and GPU pair.
The native case streams completed chunks. The snapshot case exports a completed full-prompt snapshot.
The table has one independent start per protocol. It does not establish a confidence interval.

| Model | Status | Full P cut | N | Native TTFT p95 ms | Snapshot TTFT p95 ms | Native TPOT p95 ms | Snapshot TPOT p95 ms | Snapshot p95 E2E speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | COMPLETE | 4 | 32 | 3449.61 | 3198.94 | 27.98 | 33.34 | 0.97 |
| huginn | COMPLETE | 32 | 32 | 37659.22 | 22970.29 | 131.50 | 117.17 | 1.26 |

The CSV also retains total payloads, NIXL completion counts, and goodput. These are measured protocol effects, not recurrence-split effects.

## Source and verification

Measured-data release: `d8a0209a0d5ab205308db798b9f04c6b413ad327`.
The final CPU audit checked 1,920 non-reference clean-serving output sequences.
All matched the corresponding native sequence exactly.
This result does not establish original-HF equivalence or downstream accuracy.
