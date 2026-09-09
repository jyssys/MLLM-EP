# CP × EP Direct Handoff PoC

## Final status

**NO_GO**

- Exact Streaming Handoff: **NO_GO_ACTUAL_GAIN_AND_CORRECTNESS**
- Exact Direct CP-to-Expert Transport: **ALGEBRAICALLY_BLOCKED**
- MLLM-specific effect: **NO**; the supported conclusion is a generic tested-regime result.

## Configuration and evidence scope

- Model: Qwen3-VL-30B-A3B-Instruct, snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`
- Model config: BF16; H=2048; 48 layers; 32 Q/4 KV heads; head dimension 128; 128 routed experts; top-8; no shared expert; MoE intermediate 768
- Hardware: physical GPUs 4,5,6,7; four H100 80GB HBM3; full NV18 connectivity
- Execution: exact Ulysses-style CP replay at CP1/2/4; DeepEP high-throughput EP4; actual Qwen attention/router/expert weights; actual captured Qwen3-VL hidden distributions
- Timing: same-device CUDA events, rank-critical median, warmup >=5, measured repetitions 30, randomized case order
- Scope: forward-only layer replay, not native request serving

## Runtime and prior-art audit

The local vLLM 0.20.0 has PCP topology and a PCP-aware MoE bridge, but its standard attention implementations do not advertise PCP support and the worker asserts this capability. Current SGLang source makes the relevant materialization explicit: CP-local token hidden states are gathered before Qwen MoE and sliced after it. Megatron-LM and TensorRT-LLM expose CP+EP composition but no audited exact long-prefill CP-output block streaming/direct route-aware handoff.

No direct prior-art collision was found in NanoCP, Helix/HOP-B, MoE Parallel Folding, Ulysses, or the audited runtime sources. The candidates fail on systems/algebraic merit rather than novelty.

## Gate 1 — CP crossover

Layer 24 attention-path critical-rank medians (ms):

| Context | CP1 | CP2 | CP4 | Best |
|---:|---:|---:|---:|---|
| 4K | 1.432 | 1.240 | 1.254 | CP2 |
| 8K | 2.917 | 1.922 | 1.450 | CP4 |
| 16K | 6.957 | 4.185 | 2.353 | CP4 |
| 32K | 19.275 | 10.859 | 5.695 | CP4 |
| 64K | 65.920 | 35.758 | 18.297 | CP4 |

CP becomes materially useful by 8K and is dominant at long context, so `NO_CP_REGIME` does not apply. This controlled replay scales the CP resource count and is not presented as a native-serving TP4-vs-CP fairness result.

## Gate 2 — CP→EP boundary mass

Layer 24 CP4+EP4 rank-critical medians:

| Context | Attention path | MoE path | Layer wall | Perfect block-overlap oracle | Return-A2A + dispatch share |
|---:|---:|---:|---:|---:|---:|
| 4K | 1.640 ms | 1.423 ms | 3.033 ms | 45.93% | 20.72% |
| 8K | 1.808 | 1.674 | 3.459 | 47.72% | 17.19% |
| 16K | 2.456 | 2.658 | 5.105 | 47.94% | 13.84% |
| 32K | 5.742 | 5.103 | 10.839 | 47.02% | 11.96% |
| 64K | 16.847 | 10.124 | 26.963 | 37.52% | 9.47% |

Early/middle/late 8K–32K layers show the same nominal 45–49% overlap window. The oracle is analytical and deliberately optimistic: it assumes block subdivision has zero incremental Router/DeepEP/expert cost.

## Gate 3A — Exact Streaming Handoff

The prototype computes every causal query block exactly, performs the inverse Ulysses A2A for each block, then runs its Router, real DeepEP dispatch/combine, and actual local expert weights. It overlaps one exact future attention block with current-block EP. A same-chunk/no-overlap variant isolates overlap from chunking.

### Observed actual layer result

Clean timing (30 repetitions, layer 24):

| Context | Block | Whole serial | Same chunk/no overlap | Streaming | Gain vs same chunk | Gain vs serial |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 256 | 3.504 ms | 19.366 | 19.444 | −0.40% | −454.9% |
| 8K | 1024 | 3.504 | 5.856 | 5.856 | −0.00% | −67.1% |
| 32K | 256 | 10.835 | 75.975 | 76.299 | −0.43% | −604.2% |
| 32K | 1024 | 10.835 | 22.271 | 22.285 | −0.07% | −105.7% |
| 64K | 256 | 26.721 | 154.461 | 155.669 | −0.78% | −482.6% |
| 64K | 1024 | 26.721 | 47.944 | 48.107 | −0.34% | −80.0% |

The complete detailed 64/128/256/512/1024 sweep likewise never recovers more than 1.04% over same-chunk/no-overlap and is 63–2134% slower than the whole layer. Layers 4 and 47 reproduce the result. Removing per-block detailed timing does not reveal a hidden benefit; clean timing is actually 9–24% slower than the instrumented counterpart in matched available points, so observer tax does not cause the negative result.

### Mechanism

Ulysses must first redistribute Q/K/V before any attention block. Block readiness then turns one efficient full-token EP invocation into 2–128 small invocations. Repeated route layout, dispatch/notify, combine, kernel launches, output preservation, and underfilled grouped GEMMs overwhelm the nominal window. Concurrent attention and EP also share H100 compute/memory/NVLink resources. The same-chunk negative control shows that scheduling those fragments concurrently recovers essentially none of their cost.

### Correctness

The block attention primitive alone passes: cosine 0.99999976 and relative L2 0.077%. Integrated BF16 layer decomposition does not: route agreement is approximately 96.6–99.54% and relative L2 approximately 2.2–3.0%, below the required >=99.9% route agreement and <=1% relative L2. No performance row is claimed as a correctness-qualified end-to-end speedup.

## Gate 3B — Exact Direct CP-to-Expert Transport

In Ulysses, each pre-return rank owns all sequence positions but only an attention-head shard. Qwen's dense `o_proj` mixes all heads into every output coordinate. Exact residual/RMSNorm and exact router top-k therefore require cross-rank assembly/reduction before the expert destination is known.

At H=2048 BF16 CP4, the mandatory Ulysses return carries 3,072 network bytes/token. Actual routes have mean fanout approximately 3.57 and mean remote destinations approximately 2.69, requiring another approximately 11.0 KB/token of full-hidden EP dispatch traffic. An exact counterfactual retains both transfers; distributed projection/norm/router only adds reductions, metadata, and assembly.

- Invalid magic-removal layer oracle: 11.96–17.19%
- Valid exact structural bytes saved: 0%
- Valid exact layer oracle: 0%
- Decision: `ALGEBRAICALLY_BLOCKED_ROUTE_DEPENDENCY`

The analytical oracle is below the implementation gate, so no direct-transport prototype was built.

## Matched MLLM analysis

At an exactly matched 32K execution volume, layer-24 wall time was 10.583 ms for mixed, 10.626 ms for Vision, and 10.568 ms for Text activation bases: a <=0.55% spread. There is no material modality-conditioned crossover or handoff opportunity. The result should be interpreted as generic to the tested MoE geometry, not MLLM-specific.

## Request-level gate

Request-level TTFT/E2E integration: **NOT RUN AFTER LAYER KILL GATE**.

The specification permits full-model integration only after >=10% actual layer gain or projected TTFT gain. Actual layer gain is negative and correctness fails, so projecting the 38–49% zero-overhead oracle as a request benefit would be invalid. Kimi validation was skipped for the same reason.

## Decision

The research premise has a large scheduling-only oracle but no realizable exact execution headroom on the tested hardware/runtime. Exact streaming destroys the efficient EP execution granularity and does not overlap meaningfully; exact direct transport cannot know destinations early enough to remove either mandatory communication term.

**Final: NO_GO. Do not proceed to native serving integration, Kimi, or a custom production kernel for these two candidates.**

## Artifacts

- `CP_CROSSOVER.csv`
- `HANDOFF_BREAKDOWN.csv`
- `STREAMING_ORACLE.csv`
- `STREAMING_CLEAN_TIMING.csv`
- `DIRECT_TRANSPORT_ORACLE.csv`
- Supporting audits, correctness notes, code, and raw rank JSON under the result root.
