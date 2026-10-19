# Lagged FFN contribution: source and dependency screen

## Decision

Reject this proposal before a quality run. Add no candidate card.
The proposal combines known MLP-output caching with asynchronous component execution and attention/FFN disaggregation.
This screen identifies no separate ownership benefit that requires Ouro recurrence.
This conclusion does not establish that every remote FFN layout loses.

No mechanism, model, GPU job, or quality test ran in this segment.
All performance statements below describe dependencies or structural costs.
No statement claims measured speedup or acceptable task quality.

## Changed assumption and exactness

The earlier `QUALITY_FAIL` concerns whole-stage stale/delta hidden substitution in a Thinking-1.4B pilot.
That pilot does not directly test this narrower proposal on dense Base models.
This proposal changes only the FFN contribution at each layer.
It recomputes attention, QKV, RoPE, and current-token KV from the current causal hidden state.
It retains the current attention residual instead of substituting an entire stage output.
See [the predecessor ledger](scope_and_prior_evidence.md).

“Current” does not mean baseline-exact after the first substitution.
A stale FFN contribution changes the next layer's input.
The next layer then computes current QKV and KV from that changed input.
Thus fresh execution does not preserve baseline attention inputs or baseline KV values.
If the proposal requires baseline-exact inputs at every consumer, it needs current FFNs or a valid repair procedure.

The exactness class is **unverified component approximation**.
No error bound, exact certificate, or task-quality result exists for this proposal.
The previous pilot's quality failure remains relevant negative evidence, not a proof of this proposal's quality failure.

## Direct primary priors

| Source | Actual mechanism | Transfer limit |
|---|---|---|
| [FORA, §3.2](https://arxiv.org/html/2407.01425v1#S3.SS2) | Cache each layer's attention and MLP outputs. Recompute at constant intervals. Reuse between refreshes. | Diffusion feature similarity does not establish similarity across four Ouro passes. |
| [PAB, §2.3 and Appendix A.2.2/B.2](https://arxiv.org/html/2408.12588v2#A1.SS2.SSS2) | Reuse selected MLP residual contributions across denoising steps. Select intervals and layers empirically. | The paper reports nonuniform FFN differences. Its selected diffusion intervals do not supply an Ouro schedule. |
| [AsyncDiff, §3.2](https://arxiv.org/html/2406.06911v1#S3.SS2) | Use warmup, prior-step features, component owners, and broadcasts to remove current-step dependencies approximately. | Its diffusion sampling trajectory and quality evidence do not transfer to autoregressive R4. |
| [Official vLLM AFD proxy](https://github.com/vllm-project/afd-plugin/blob/0fff229b592bc489c8857cdabc39f247687acaf5/afd_plugin/model_executor/models/deepseek_v2.py#L117-L158) | Send the current FFN input. Yield other microbatch work. Receive that FFN result before returning. | This inspected proxy preserves the current-result dependency. It is not a lagged-FFN implementation for Ouro. |

The source code makes the component boundary explicit.
[FORA `models.py:145–152`](https://github.com/pratheba/FORA/blob/acde628eb1c9605015542c91040c401919369487/src/models.py#L145-L152) adds a cached MLP result to the current residual stream.
[PAB `open_sora_transformer_3d.py:242–284`](https://github.com/NUS-HPC-AI-Lab/VideoSys/blob/4cce477878b0683b5c05cd001ce45ac0e2fc3034/videosys/models/transformers/open_sora_transformer_3d.py#L242-L284) selects cached FFN contributions before residual addition.
[PAB `pab_mgr.py:108–174`](https://github.com/NUS-HPC-AI-Lab/VideoSys/blob/4cce477878b0683b5c05cd001ce45ac0e2fc3034/videosys/core/pab/pab_mgr.py#L108-L174) tracks timestep/block keys and removes expired entries.

[AsyncDiff `async_sd3.py:15–40`](https://github.com/czg1225/AsyncDiff/blob/47cae2b021ec14a75eac8c37b14f69baf7f5cd07/asyncdiff/async_sd3.py#L15-L40) implements warmup, cached module results, and broadcasts.
These sources do not contain this exact combined Ouro proposal.
They do establish the main approximation and asynchronous execution mechanisms.
The targeted recurrent-component searches found no closer primary implementation during this bounded screen.
That search result does not establish absence of prior work.

## Actual Ouro boundary

Use one-based pass `r` and layer `l` below.
`N1` through `N4` are the four distinct RMSNorm modules in one physical layer.
The branch cache stores the **normalized FFN contribution**, not the full layer output.

```text
z[r,l] = N1(x[r,l])
a[r,l] = Wo(Attention(RoPE(Wq(z)), depth-specific KV from Wk(z), Wv(z)))
y[r,l] = x[r,l] + N2(a[r,l])
u[r,l] = N3(y[r,l])
b[r,l] = N4(Wdown(SiLU(Wgate(u[r,l])) * Wup(u[r,l])))

warmup:  x[1,l+1] = y[1,l] + b[1,l]
lagged:  x[r,l+1] = y[r,l] + b[r-1,l], for r=2,3,4
```

The last layer feeds the shared loop norm before the next pass, gate, or LM head.
The notation describes causal dependencies, not a bitwise rewrite of native fused residual kernels.
The native code carries branch and residual tensors separately until a fused norm consumes them.

Primary local locators are HF `modeling_ouro.py:401–423` and native `ouro.py:198–209,260–281,367–374`.
The inspected HF source hash is `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.
The inspected native source hash is `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff`.
Paths and additional source hashes appear in the structured artifact.

Caching raw MLP output before `N4` defines a different boundary.
It requires `N4` on the consumer and preserves that normalization cost.
Neither boundary makes a previous-pass contribution equal to the current contribution.
BF16 transport must preserve the stored words and the selected boundary.
Norm arithmetic, residual fusion, and reduction order still require explicit numerical qualification.

## Causal version and ownership

Each entry needs `(request, token position, physical layer, producer pass, input generation)`.
The input generation identifies the actual approximate `u[r,l]` that produced the entry.
A remote worker must retain an immutable input until its read completes.
It must publish an immutable result only after its write and return transfer complete.

R3 may consume only the specified R2 entry under a one-pass-lag contract.
If that entry is late, the consumer waits.
Using the “latest available” R1 entry silently changes the mechanism to variable lag.
An in-place replacement cannot change an entry that another stream still reads.
Two branch banks and completion events are one possible storage policy, not an implemented engine.

The attention owner retains current residuals, historical KV, logical-depth identities, the gate, and commit authority.
The FFN owner retains physical FFN weights and versioned input/output queues.
Placing `N4` there also requires its small norm weights.
No worker owns a future FFN input before current attention produces `u[r,l]`.

## Warmup1 R4 path

The following recurrence charges a single FIFO FFN owner.
Other ownership choices require their own resource queues.
Let `H[r,l]` be the time the front can start a layer.
Let `a` include its current attention, current KV operations, and work through `u`.
Let `s`, `f`, and `v` charge send, FFN plus `N4`, and return.
Let `i` charge cached-branch access and residual integration.
Let `Fprev` be the previous job's completion on that FFN owner.

```text
U[r,l]       = H[r,l] + a[r,l]
F[r,l]       = max(U[r,l] + s[r,l], Fprev) + f[r,l] + v[r,l]
H[1,l+1]     = max(U[1,l], F[1,l]) + i[1,l]
H[r,l+1]     = max(U[r,l], F[r-1,l]) + i[r,l], r=2,3,4
H[r+1,1]     = H[r,L+1] + loop_norm[r] + required_control_wait[r]
```

This dependency recurrence is exact only for the stated queue abstraction and specified service times.
It is not a calibrated latency model.
Real links, TP collectives, stream contention, copies, snapshots, and graph overhead can add dependencies.

Warmup1 executes one complete current-FFN pass on the token's current history.
If historical KV already came from approximation, warmup does not restore baseline history.
Only R2, R3, and R4 can use lagged contributions after that warmup.
An ideal front-only floor removes their FFN waits, not their attention or integration work.
For constant costs, that floor is `L(a+s+f+v+i)+n + 3[L(a+i)+n]` before head and commit.
It assumes all replacement deadlines succeed and all required control waits vanish.
It is not an achievable speedup claim.

There are three different terminal contracts:

1. **Lagged R4 commit.** The head consumes R4 with R3 FFN contributions. R4 refresh results do not affect this output.
2. **Drain unused replacements.** Buffer reuse or mandatory completion waits until `max(front_done, max_l F[4,l])`. This wait does not repair the output.
3. **Current-FFN R4.** Every R4 layer waits for `F[4,l]`. This restores R4's FFN chain but leaves earlier approximate state and history.

Skipping unused R4 refresh jobs reduces FFN evaluations from `4L` to `3L` for lagged R4 commit.
This is generic approximate cache reuse, not asynchronous exact replacement.
Applying all late R4 results without replay is invalid because later layer inputs already depend on older contributions.
Exact repair needs a valid verifier or recomputation of affected attention, KV, norms, and gates.
This screen supplies neither.

## Work supply, memory, and matched controls

At C1, a current attention front can overlap a ready FFN refresh from an earlier visited layer.
The overlap becomes legal only because the front accepts a stale contribution.
At C2/C4/C8, at most 2/4/8 request fronts independently expose current layer work.
Additional queued FFNs do not imply those jobs meet their next-pass deadlines.
Exact attention/FFN disaggregation can also overlap different ready requests or microbatches without stale contributions.
No ready-work or utilization measurement ran here.

Both Base models use `D=2048`, `I=5632`, and 16 MHA heads.
Base-1.4B uses `L=24`. Base-2.6B uses `L=48`.
One BF16 branch row occupies 4096 bytes.

| Structural payload | Base-1.4B | Base-2.6B |
|---|---:|---:|
| One branch bank per in-flight token | 96 KiB | 192 KiB |
| Two branch banks at C1/C2/C4/C8 | 192/384/768/1536 KiB | 384/768/1536/3072 KiB |
| One complete bank of immutable FFN inputs | 96 KiB | 192 KiB |
| Full physical FFN weights, BF16, excluding norms | 1584 MiB | 3168 MiB |
| Input plus returned branch per remote layer evaluation | 8192 bytes | 8192 bytes |
| Remote payload for `4L` evaluations per token | 768 KiB | 1536 KiB |

These counts exclude descriptors, allocator padding, events, TP traffic, and duplicated work.
They do not measure HBM or link traffic.
Every evaluated FFN still performs its two input projections, SwiGLU, and output projection.
One layer's 66 MiB of FFN matrices exceeds the observed 50 MiB L2 capacity.
The existing [component artifact](../outputs/components_base14_P2_graph_v1/components.json) records those capacities.
No residency or bandwidth gain follows from the weight-sharing fact alone.

For P2/P4/P8, compare equal total device counts and explicit attention/FFN splits.
P8 remains projected, not measured.
The controls must include native asynchronous TP with graph replay and generic exact AFD where applicable.
An approximate comparison also needs generic one-pass-lag FFN caching with the same ownership and queue policy.
Native TP already shards one aggregate copy of physical weights.
An attention/FFN split cannot claim a fourfold weight-residency reduction against that control.

The prior remote-MLP losses do not resolve every possible split or batching policy.
Conversely, this proposal identifies no different placement optimum from C02's KV-stationary remote FFN pool.
It adds approximation to that ownership pattern.
The alias-equivalence caution remains valid: generic graph expressibility alone does not reject a concrete ownership optimum.
The rejection here instead uses direct component-cache priors and the absence of a separate ownership discriminator.

## Original-gate adaptive semantics

Apply the unchanged original gate to each current loop-normalized hidden state.
Preserve BF16 sigmoid before FP32 conversion and the original recurrent-order hazard accumulation.
CDB `exit_policy.py:161–166` shows this order.
CDB `adapters/ouro.py:262–264` places the gate after the loop norm.

The unchanged gate receives approximate hidden states after the first lagged contribution.
It can therefore select different depths from the exact baseline.
Keeping its weights, threshold, and minimum depth does not preserve baseline exit decisions.
The selected policy still uses q=0.5, minimum R2, maximum R4, and `last_exited`.
R1's hazard still contributes despite the minimum-R2 constraint.

An R2 or R3 exit makes some queued refreshes unused.
Charge their completion or safe cancellation before request buffers return to the allocator.
Publish the selected pass's KV rows through the unchanged terminal policy.
Do not let unused refreshes change gate history, selected hidden state, or terminal KV.
No FFN replacement can retroactively repair already-written attention KV without a repair protocol.

## Cheapest discriminator and stopping reason

The source-level discriminator already fails the campaign's novelty screen.
PAB exposes a branch cache before residual addition, and AFD exposes current-input remote FFN ownership.
The proposed overlap removes the current FFN dependency through the same stale-contribution assumption.
No independent ownership or memory advantage survives against their matched combination in this derivation.
This is not a theorem that approximate component caching can never improve latency.

No quality run or GPU proposal follows from this screen.
Reopening it requires a concrete ownership difference from generic lagged FFN caching plus AFD.
It also requires an explicit quality or verification contract for changed hidden states, KV history, and online exits.
This screen allocates no new candidate number.

## Actual work and source record

Start: `2026-10-19 15:36:26 UTC`.
End: `2026-10-19 15:44:42 UTC`.
Observed segment: 496 seconds, or 8 minutes 16 seconds.
The time excludes this final timestamp insertion and handoff.
The segment contains source searches, pinned code inspection, dependency derivation, and report checks.
No GPU execution, model execution, or mechanism construction occurred.
Concurrent agent intervals do not multiply campaign elapsed time.

The structured record is [lagged_ffn_component_source_screen_v1.json](../artifacts/lagged_ffn_component_source_screen_v1.json).
It contains the actual queries, pinned revisions, hashes, locators, and bounded conclusion.
JSON parsing, 19 structural arithmetic checks, and the report lint passed.
