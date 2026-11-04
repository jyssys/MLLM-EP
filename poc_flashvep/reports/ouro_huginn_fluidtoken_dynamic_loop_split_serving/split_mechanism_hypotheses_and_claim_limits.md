# Split mechanisms and claim limits

## Evidence status

This note states structural hypotheses before the clean method comparison.
It does not report measured Token-versus-Loop speedup.
The final result must use the accepted serving measurements.

## Equal rows do not imply equal Attention work

For a dense causal prompt of length `T`, each Attention layer has `T(T+1)/2` valid query-key pairs.
A Loop cut moves this complete triangle for each remaining loop.
A Token suffix contains later queries with longer prior context.
Its valid pair count is `[T(T+1)-k(k+1)]/2` per loop, where `k` is the Prefix length.

The following counts exclude Prelude and Coda.
They also exclude head count, head dimension, and physical layer multiplicity.
They are not measured FLOPs or execution times.

| Model and matched boundaries | D recurrent rows | Loop causal pairs per physical Attention layer | Token causal pairs per physical Attention layer |
|---|---:|---:|---:|
| Ouro R4, Loop cut 3, Token Prefix 3072/4096 | 4096 | 8390656 | 14682112 |
| Huginn R32, Loop cut 28, Token Prefix 1792/2048 | 8192 | 8392704 | 15732736 |

Token has 1.749817 times the structural pair count in the Ouro example.
Token has 1.874573 times the structural pair count in the Huginn example.
These ratios do not predict equal ratios of total GPU time.
The MLP, projections, state movement, and scheduling also contribute.
Moving more Attention work can also remove more work from P.
Therefore, this counting argument does not establish that Loop Split is better.

## Recurrent execution granularity

The Ouro example moves one full-prompt loop instead of four suffix-depth passes.
The Huginn example moves four full-prompt loops instead of 32 suffix-depth passes.
The two axes have equal recurrent token rows in each example.
Their physical GEMM shapes and possible kernel counts can still differ.

Large prompt rows may improve weight reuse and reduce submission overhead.
This remains a hypothesis until actual kernel and latency measurements support it.
An implementation can batch or fuse Token work, which can reduce the difference.
Token Split must retain its large, efficient Prefix and Suffix options.
It must not receive an artificial C256 limit.

## Huginn boundary work can favor Token Split

The pinned Huginn checkpoint has two Prelude layers, four Core layers, and two Coda layers.
The native runner builds prompt Coda KV after the final recurrent step.
It writes valid prompt positions before the final prompt token.
Loop continuation must perform that boundary work on D.
Token continuation performs only the remaining Suffix boundary work on D.

| Huginn split | D Core block-token rows | D additional Prelude block-token rows | D prompt Coda-KV block-token rows |
|---|---:|---:|---:|
| Loop31, T2048 | 8192 | 0 | 4094 |
| Token Prefix1984, T2048 | 8192 | 128 | 126 |
| Loop28, T2048 | 32768 | 0 | 4094 |
| Token Prefix1792, T2048 | 32768 | 512 | 510 |

These counts exclude the common first-token Coda execution and other operations.
They do not imply equal compute cost for all block-token rows.
They show why equal recurrent rows alone are an incomplete matching criterion.
In this component, Loop Split transfers more boundary work to D.
The measured D continuation cost must include this work.
It must not be omitted to make Loop Split appear cheaper.

The source is the pinned checkpoint `config.json` and the native `ModelRunner._execute_batch` boundary-KV path.
No model parameter or checkpoint file changes.

## Projection geometry control

The qualified numerical runtime retains its fixed projection geometry.
Huginn's 64-token Suffix can incur more padded work than its logical row count suggests.
The Loop28 versus Prefix1792 comparison uses a 256-token Suffix.
It separates that small-row padding artifact from the primary boundary comparison.
No projection change is allowed for only one method.

## State transfer trade-off

Equal recurrent rows can produce comparable recurrent KV counts.
Loop migration can still require a larger intermediate hidden state.
Token migration can require different Prelude, Coda, and injection state.
Only the actual connector records determine transferred payloads and fragments.

Neither payload size nor cumulative transport polling time is exclusive DMA latency.
Exposed handoff must come from the actual request timeline.
Native streaming and snapshot controls remain separate.

## Dynamic policy scope

The new policy uses isolated measured service costs.
It does not subtract elapsed reservation time as though that time were GPU execution.
It uses a native completion receipt when Prefill computation is known to be complete.
Its remaining-service prediction is otherwise conservative.

Both axes use the same observed-state selector and frontier recheck.
An empty Decode batch has no existing token interval to protect.
The policy can therefore consider larger work when that batch is empty.
It must recheck the batch and reservation at the actual frontier.
The frozen evaluation SLOs remain unchanged.

Forced acceptance proposals and forced rejection tests are correctness diagnostics.
They are not online-policy performance measurements.
No future arrival or actual future output length enters the online policy.

## Novelty boundary

A generic logical-layer split can select the same recurrent boundary.
With identical placement, state, and execution, the mechanisms are equivalent.
Token-versus-Loop improvement alone cannot establish exclusive Loop-specific novelty.
The final report must retain this limitation even if Loop Split improves serving.
