# Ouro execution semantics audit

## Scope and result

This audit inspects source code and checkpoint metadata. It runs no model or GPU workload.
Source support does not establish numerical parity, quality, or speedup.

Both Base models use the same published Python implementation.
Ouro-1.4B has 24 physical layers. Ouro-2.6B has 48 physical layers.
Each model applies its physical stack four times.
The resulting fixed-depth counts are 96 and 192 logical block executions per position.

The expected local HF cache contains Base 1.4B weights.
It does not contain a Base 2.6B directory at the audited location.
The local 2.6B-Thinking checkpoint is a different checkpoint.
Its fixed-depth results do not check adaptive Base 2.6B.

The pinned CDB engine expresses both layer counts without a new model engine.
Its generic support remains a source-backed finding.
Adaptive Base 2.6B remains `ADAPTIVE_2P6B_UNVALIDATED` until the actual checkpoint passes runtime checks.
The historical wrapper contains 1.4B-specific assertions that require a scoped change.

## Source pins

| Source | Pin | Verification |
|---|---|---|
| Cached Base Ouro-1.4B | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | Local config, modeling source, and safetensors header |
| Published Base Ouro-2.6B | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | Live official API, pinned config, and pinned Python source |
| CDB | `abdfae0adbdd12be8c6123ccb34eee98a38afb39` | Local Git HEAD and source |
| Installed native vLLM Ouro | SHA-256 `cf0792c6c8024ded45dba10553b5f0d24b7b6542dd8f8f0e0e6899807ec0c4ff` | Local installed source only |

The two Base modeling files share SHA-256 `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.
Their configuration classes also share SHA-256 `950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb`.
The CDB modeling file differs. Its SHA-256 is `47c21a0e5f33992ccc40464eff8f41eee414e16413f247caabdb7f975b04b538`.

The official Base 2.6B API lists one `model.safetensors` file.
This audit retrieves no weight file.
Its published inventory and access details appear in `artifacts/semantics_manifest.json`.

Primary sources: [Base 1.4B source](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py), [Base 2.6B config](https://huggingface.co/ByteDance/Ouro-2.6B/blob/1ed04250da1a9936042725d302e81c8fa2ab5abd/config.json), and [Base 2.6B source](https://huggingface.co/ByteDance/Ouro-2.6B/blob/1ed04250da1a9936042725d302e81c8fa2ab5abd/modeling_ouro.py).

## Dimensions and structural accounting

| Field | Base 1.4B | Base 2.6B |
|---|---:|---:|
| Physical layers L | 24 | 48 |
| Recurrent passes R | 4 | 4 |
| Hidden width H | 2048 | 2048 |
| MLP width I | 5632 | 5632 |
| Query / KV heads | 16 / 16 | 16 / 16 |
| Head dimension | 128 | 128 |
| Vocabulary | 49152 | 49152 |
| Maximum configured positions | 65536 | 65536 |
| RoPE theta | 1000000.0 | 1000000.0 |
| RMSNorm epsilon | 0.000001 | 0.000001 |
| Checkpoint dtype | BF16 | BF16 |
| BOS / EOS IDs | 0 / 0 | 0 / 0 |
| Fixed depth-indexed KV pairs | 96 | 192 |
| BF16 KV bytes per cached token | 786432 | 1572864 |
| B1 full-depth KV at 1024 positions | 0.75 GiB | 1.50 GiB |
| B1 full-depth KV at 16384 positions | 12.00 GiB | 24.00 GiB |

These capacity values use `2 * L * R * N * 16 * 128 * 2` bytes.
They exclude allocator padding, block tables, graph buffers, and scratch memory.
They are structural accounting, not measured HBM traffic or latency.

Each Q, K, V, and output matrix has shape `[2048, 2048]`.
Each MLP gate/up matrix has shape `[5632, 2048]`.
The MLP down matrix has shape `[2048, 5632]`.
The embedding and LM-head matrices each have shape `[49152, 2048]`.
The config disables embedding/head weight tying.

The local Base 1.4B header contains 269 tensors and 1434652673 elements.
Their BF16 payload occupies 2869305346 bytes, or 2.6722488421946764 GiB.
The header includes `model.early_exit_gate.weight` with shape `[1, 2048]`.
It includes a separate gate bias with shape `[1]`.
No new runtime allocation or pointer measurement occurs here.

## Exact block dependencies

Use zero-based recurrent depth `r`, physical layer `l`, and token position `t`.
Let `N1`, `N2`, `N3`, and `N4` denote the block's four separate RMSNorm modules.

```text
x = input(request, r, l, t)
z = N1(x)
Q, K, V = Wq(z), Wk(z), Wv(z)
Q, K = RoPE(Q, K, position=t)
commit/read KV(request, r, l, positions<=t)
a = Wo(attention(Q, KV, causal_mask, scale=128**-0.5))
y = x + N2(a)
u = N3(y)
m = Wdown(SiLU(Wgate(u)) * Wup(u))
output = y + N4(m)
```

This notation describes the mathematical dependencies. It does not assert bitwise equivalence across kernels.
The source evaluates `N3(y)` once before the MLP call.

The official locations are `OuroMLP.forward:65–78`, `OuroAttention.forward:292–344`, and `OuroDecoderLayer.forward:401–423`.
`OuroRMSNorm.forward:357–362` computes the mean square in FP32.
The eager attention path computes softmax in FP32 at lines 246–248.
Other attention backends can reorder floating-point operations.

After the last physical layer, `OuroModel.forward:606–608` applies the shared loop-boundary RMSNorm.
The normalized result becomes the next pass's input.
The same normalized result supplies the original scalar gate.
The next pass cannot execute before this result exists.

Feature sharding does not permit an unreduced partial sum through these normalization steps.
The RMSNorm denominator depends on all hidden coordinates.
Attention softmax also depends on the complete relevant key set within each head.
Any distributed replacement must retain these dependencies or declare an approximation.

## Weight identity and KV identity

`OuroModel.__init__:502–508` creates one physical `ModuleList`.
`OuroModel.forward:592–604` reuses its modules across passes.
This source establishes module reuse. It does not measure cache residency or HBM reuse.

`OuroAttention.forward:317–321` indexes KV as `r * L + l`.
The same weights therefore act on different activations and different depth-indexed state.
Weight identity does not establish activation equality or KV equality.

The official cache stores keys and values as `[batch, heads, positions, head_dim]`.
`UniversalTransformerCache.update:164–178` stores the first input tensor directly.
Later updates create concatenated prefix tensors with `torch.cat(..., dim=2)`.
The initial assignment can retain an input tensor alias.
The source does not alias separate recurrent-depth cache entries.
An exact speculative branch must isolate cache writes before its acceptance check.

The CDB cache instead allocates one key tensor and one value tensor per cache layer.
Each paged tensor has shape `[physical_rows, 16, 128]` before any external head sharding.
`continuous_batching/cache.py:125–144` performs these allocations separately.
These paged allocations avoid the official cache's repeated full-prefix concatenation.
That cache implementation difference is generic and cannot establish a new recurrence-specific architecture.

## Fixed output selection and adaptive execution

The official HF implementation always executes all configured passes first.
Its threshold path later selects the hidden state used by the LM head.
See `OuroModel.forward:592–608` and `OuroForCausalLM.forward:774–800`.
An early selected output is not physical early skipping in this implementation.
`exit_at_step=3` selects R4 because this argument uses zero-based indices.

For the original gate, `lambda_r = sigmoid(gate(h_r))`.
The exit mass is `lambda_r * survival_r` before the last pass.
The last pass receives the remaining mass.
See `OuroForCausalLM.forward:722–733`.
The official threshold path selects the first cumulative mass at least q.
It does not implement the historical minimum-R2 policy by itself.

CDB separates embedding, recurrent computation, and the LM head.
`adapters/ouro.py:199–264` applies one physical-stack pass per recurrent launch.
`exit_policy.py:150–169` updates the token-local cumulative hazard and applies the minimum depth.
The R1 hazard still contributes when minimum depth prevents an R1 exit.
At maximum depth, the runtime exits regardless of a later gate result.

The historical selected path uses Base 1.4B, q=0.5, minimum R2, maximum R4, and `last_exited`.
It uses full-R4 prefill, synchronous routing, and online exits.
It disables delayed gate consumption and synthetic replay.
See `/home/esjung/MLLM-EP-depth-cdb-factorial/poc_decode_first_depth_aware_dp/runtime.py:28–59`.
That loader checks the loaded original gate weight and bias against the checkpoint.
This audit reads the check. It does not rerun it.

The CDB model also creates lookahead and preloop heads.
The selected adapter binds `decoder.early_exit_gate`, not either extra head.
See `adapters/ouro.py:266–299`.
`configure_ouro_exit_gate:94–99` rejects untrained optional heads without a compatible gate checkpoint.
The primary original-gate path requires no new trained component.

## `last_exited` state transitions

This policy preserves four physical slots per layer.
It does not reduce full-depth cache allocation.
`kv_cache_policy.py:53–74` derives the slot count.
Its cache index remains `slot * L + l` at lines 106–128.

When a token exits at depth e, the runtime records a copy backlog.
Before a later recurrent launch can read deeper slots, the runtime flushes that backlog.
See `continuous_api.py:1190–1230`.
For each physical layer, the runtime copies that token's K/V row from e into every deeper slot.
These are equal-by-copy rows, not aliases.
`continuous_depth_batching/cache.py:105–113` uses `index_select` and `index_copy_`.
Full-depth prompt KV remains depth-specific.

The copy protocol assumes at most one recurrent token per request in flight.
Its source states this condition at `continuous_api.py:1210–1215`.
A branch or parallel-token runtime must replace that assumption with versioned ownership.

The current paged launch accepts one explicit KV slot.
`model_runner.py:756–771` rejects a recurrent batch with multiple slots.
`scheduler.py:322–339` groups ready work by slot.
Thus the selected `last_exited` path cannot combine arbitrary depths in one core batch.
This is a current representation constraint, not a proof that Ouro forbids such batching.

An exact indirect reader could potentially represent copied terminal rows without physical duplication.
It would need per-token depth mapping and a compatible attention reader.
No such reader executes in this audit. Its memory and latency benefit remain untested.

## State ownership contract

| State | Creation and owner | Readers | Mutation and lifetime |
|---|---|---|---|
| Physical weights | Model loader, model worker | All passes on that worker | Read-only during inference, checkpoint lifetime |
| Prompt KV | Full-R4 prefill, request cache owner | Later tokens at matching depth/layer | Immutable after commit, request lifetime |
| Current token KV | Attention projection, request cache owner | Current attention and later tokens | The current runtime writes directly during the pass |
| Terminal KV copies | Exit-copy flush, request cache owner | Later tokens in deeper slots | Written before later reads, request lifetime |
| Hidden state | Predecessor block, compute worker or hidden bank | Next block, loop norm, gate, head | Replaced at each scheduled pass |
| Gate state | Original gate and token-local policy | Local scheduler | CDF/survival update for this token only |
| Branch scratch | Proposed branch owner | Branch-local readers | Must not change authoritative KV before its acceptance check |

At B1, one BF16 hidden vector contains 4096 bytes.
One physical layer's K/V row contains 8192 bytes.
Historical KV grows with context length, depth, and layer count.
Neither remote compute nor request migration can omit the required state reads.

## Both-size runtime support and gaps

| Path | Base 1.4B | Base 2.6B |
|---|---|---|
| Official fixed R4 source | Cached config/source/weights available | Published source/config verified, local Base weights unavailable at audit |
| Official selected adaptive logits | Source supports gate selection after all R4 work | Identical source, no new execution test |
| CDB original-gate staged execution | Prior runtime exists with partial reference qualification | Source supports L48, execution and quality unvalidated |
| Historical wrapper | Checks 24 layers / 96 slots | Hardcoded assertions require generalization |
| Installed native vLLM | Config-driven fixed R4 source | Config-driven fixed R4 source, prior Thinking measurements only |

CDB `OuroCDBAdapter.recurrent_step:249–264` uses `config.num_hidden_layers`.
`kv_cache_layout:186–197` derives all core layers from the same config.
The allocator uses the derived geometry at `continuous_api.py:303–320`.
The inspected engine surfaces contain no 24-layer or 48-layer branch.
This supports a configuration-driven L48 path, not a runtime PASS.

The historical wrapper asserts exactly 96 cache entries at `runtime.py:59`.
Its factorial worker also asserts a 24-layer call formula at `scripts/worker.py:224`.
Changing only the model path would therefore fail the wrapper's integrity checks.

The installed native vLLM source partitions Q/K/V heads and MLP columns across TP ranks.
It uses row-parallel output and down projections.
It creates separate attention-cache objects for each recurrent depth.
See installed `ouro.py:120–196` and `ouro.py:68–101`.
Its forward resets the fused residual representation at each pass boundary.
It applies the boundary norm after every pass at `ouro.py:367–374`.
The forward never calls its declared exit gate. This inspected path supports fixed depth only.

## Prior evidence and limits

The async PP report at commit `458c1fe9b43f08863308d3b661a4163a854cb9aa` reports `QUALITY_FAIL`.
Its checkpoint is 1.4B-Thinking, not Base.
The source freezes stale/delta stage inputs before each round at `emulator.py:81–107`.
It preserves candidate KV into later tokens at `emulator.py:139–155`.
Its serial emulation does not supply measured PP4 performance.
The report gives exact, stale, delta, and W0 GSM8K counts of 15/16, 8/16, 7/16, and 6/16.
The corresponding MATH counts are 14/16, 3/16, 3/16, and 1/16.
This audit does not repeat those quality measurements.

The asymmetry report at commit `eb7d6a400f794d16952cb15889e8d567a5eb0a62` reports `GENERIC_ONLY`.
Its prefill DAG maps to generic causal PP.
Its native/HF and smaller-chunk numerical checks retain partial qualification.
Its footprint and component times do not establish Base 2.6B speedup.

The TP report at commit `2481ef0f77dd029a9ac905ee749e9f0317becf02` reports `CONTROL_BLOCKED`.
Its Ouro checkpoint is 2.6B-Thinking at `f1edd81e7ac41355db670500ceaf204e0f73af68`.
`tied_tape.py:5–18` preserves shared physical modules and per-pass normalization.
The report retains an unexplained natural mismatch in its retained-graph lane.
The ordinary-graph follow-up passes its selected checks.
Neither result proves universal parity or a Base adaptive result.

No source-backed dependency permits concurrent exact R1–R4 computation for the same token without additional work or approximation.
Prefill receives all prompt token IDs. Their intermediate causal KV remains subject to block dependencies.
Decode token t+1 becomes available only after the preceding output decision.
Independent DP replicas need no artificial global token barrier.

## Required next checks

1. Acquire or locate the pinned Base 2.6B checkpoint through the coordinator's controlled workflow.
2. Generalize only the wrapper's model path and layer/slot integrity formulas.
3. Check loaded original gate values against that checkpoint.
4. Check all 192 slot addresses, full-depth prefill, terminal copies, and online exit traces.
5. Retain numerical-order caveats across TP degrees, batch shapes, and backend changes.
6. Compare changed adaptive execution against the same original-gate/KV policy.

All new performance, free-running quality, and Base 2.6B runtime results remain `NOT_RUN` in this audit.
