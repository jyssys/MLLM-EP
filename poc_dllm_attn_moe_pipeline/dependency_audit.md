# Native dependency source audit

Pinned bridge: dInfer `modeling_llada2_moe_sglang.py`, `forward_deepep`;
native `LLaDA2Block.forward`, `LLaDA2Attention.forward`, ModelRunner;
installed SGLang0.5.3 `LayerCommunicator` and DeepEP normal dispatcher.
Recheck source SHA in environment.json; external dirty sources are not edited.

| Operator | Input → output layout | Ready unit / stream | Collective / mutable state | Legal early use |
|---|---|---|---|---|
| source-SP narrow | replicated dense `[B,N,2048]` → contiguous localN/4 rows | view only | all ranks deterministic contiguous ownership | not a computed result |
| router/top8 | local rows → IDs/weights | whole call/current stream | fresh native sigmoid+bias, scale2.5 later | no future truth |
| dispatch | source rows → owner-local received rows/IDs | public complete receive tensor; DeepEP event ordering | actualEP4 A2A, dispatcher handle state | no per-row release exposed |
| fused expert apply/local reduction | owner received rows → reduced rows | grouped kernel/current stream |64 unique expert weights/rank; all8 paths | individual expert completion UNKNOWN |
| combine+wait | owner reduced rows → source-row routed sum | whole-buffer event, current_stream_wait | DeepEP reverse traffic; reset dispatcher stage/handle | routed output only, not final hidden |
| shared expert/add | local input → local final MoE rows | full shared projection+add/current stream | shared weights replicated (TP1 within shared path) | local publication after add event only |
| source all-gather | local final rows → replicated all rows | whole output on NCCL completion/current stream | TP group contains the4 EP ranks | next head-shard consumer before gather is illegal |
| layer epilogue | complete MoE rows + residual | native communicator | actual scatter modes recorded physically | preserve residual order |
| next prepare_attn | hidden+residual → normalized rows+updated residual | fused residual/norm / native communicator | TP/SP mode matches actual flags | after complete rows only |
| QKV / QK norm / RoPE | complete rows → per-head TP shards | local operators/current stream | positions/request/cache version unchanged | token-local after TP visibility |
| immutable prefix partial | query group + prior-block KV → softmax state | complete kernel | valid prefix ONLY; original mask | can precede fresh C completion if prefix proven |
| current-block KV | every current row → fresh K/V | join across groups | **accepted current positions still mutable** | all fresh current-block keys required |
| current partial+merge | global normalized attention output | FP32 max/sum/weighted-sum | no sum of separate normalized outputs | no future-refinement state |
| output projection | attention head shards → native output | whole row | native TP reduction/residual | final synchronization charged |

Natural interface publishes all local routed rows together, then all gathered
rows together. Individual-row arithmetic completion is **unobservable**, not
inferred from rank-local stage latency. Any engineered2-group run is a separate
lane with real extra dispatch/expert/combine/shared/all-gather costs.

Cache audit: native attention uses `slice_scatter` of current K/V into the tail
of a padded cache. The actual selected hot snapshot passes `attention_mask=None`;
we preserve that unmasked native behavior, including zero-padded cache slots,
rather than assuming a causal/boolean mask. Padding is not real prefix tokens.
Prefix eligibility comes from immutable earlier blocks and the actual native
mask/cache contract, never token acceptance. Direct per-invocation cache-version
logging was missing in the main capture, so that join remains unresolved.
Cold/no-cache work has no eligible prefix; cross-block two-block invocations
use their full fresh current span. No overwrite of native KV state is allowed.

Clock rule: only same-device event offsets/durations are comparable. Natural
group spread is0 at the exposed whole-buffer release by interface, not by
invented GPU clock alignment. Cross-device latent skew remains UNKNOWN unless
a validated common-time uncertainty bracket exists. Use one cumulative rank
for component totals, never sum unrelated maxima.

Baseline ladder: native fused norms and unquantized Triton experts retained.
Manual dInfer DeepEP invokes host-variable receive-count dispatch and has no
supported whole-model CUDA graph in this pinned interface; capture is probed,
not assumed. TBO serving flag is not connected to the manual runner. Shared
expert dual-stream functional adaptation is measured separately on the
physical producer; its savings never count as new cross-layer prefix gains.
