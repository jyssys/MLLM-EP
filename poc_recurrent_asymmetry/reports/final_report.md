# Recurrent Memory–Compute Asymmetry Discovery

## Verdict

**GENERIC_ONLY. Do not construct a new recurrent parallelism engine from this evidence.**

The exact R4 reference passes its regression checks.
Tied weights remain fixed while computation and depth-indexed KV grow with recurrence.
However, neither tested track demonstrates an additional recurrent-specific physical latency advantage.
Full clean four-GPU critical-path attribution and cross-backend numerical qualification remain incomplete.
The study reports these limitations rather than converting structural opportunity into measured speedup.

The source contract is the spec at the exact path supplied by the user.
Its unchanged copy and SHA256 are recorded in `environment.md`.
All model experiments use physical GPUs 4–7 only.

## Measured result summary

| Item | Result | Evidence class |
|---|---|---|
| Model | Ouro-1.4B-Thinking, revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, BF16, exact R4 | Loaded configuration and source audit |
| Physical execution | 24 shared layers, 96 logical executions, 96 independent KV slots | Actual execution/cache checks |
| Reference numerical parity | 8/8 padded B1/B2 cached checks, bitwise logits and all KV | Measured |
| Natural reference parity | 251/251 and 772/772 tokens, including EOS | Measured |
| Small quality smoke | GSM8K 2/2, MATH-500 2/2 | Measured, not a benchmark accuracy estimate |
| Unique tied weight storage | 2.672249 GiB | Actual storage accounting |
| B1 logical KV at 1K / 16K | 0.75 / 12 GiB | Actual tensor shapes/storage |
| Valid decode attention-core interval | 4.869 / 6.544 ms at 1K / 16K | Separate single-family event lane |
| Infinite attention-core acceleration ceiling | 7.268% / 7.345% | Modeled from matched reference intervals |
| Physical conventional CP operator | Approximately 0.55 ms versus 0.055–0.102 ms native | Measured microbenchmark, not E2E |
| Exact prefill dependency graph | Legal concurrency, identical edges to matched generic unroll | Analytical control and toy tests |
| Loop-specific gain over identical generic schedule | 0% within the tested model | Analytical identity, not measured speedup |
| Smaller-chunk numerical qualification | 8/32 checkpoints pass, 32/32 top-1 match | Measured, REFERENCE_PARTIAL |
| Native TP1 / TP2 / TP4 | All physically measured, all REFERENCE_PARTIAL against HF | Hardware diagnostics only |
| New engine / measured candidate E2E gain | NOT_RUN / NOT_RUN | Scope and failed gates |

## What costs are visible?

The long-prefill reference has substantial attention and MLP intervals.
Their 16K intervals are 325.764 and 168.858 milliseconds per forward.
The matched clean prefill is 879.832 milliseconds.
These intervals correspond to 37.03% and 19.19% of that denominator.
They do not form an exclusive partition across independently instrumented runs.

Cached attention is smaller in the matched light lane.
At 16K, its interval is 6.544 ms against an 89.089 ms clean forward.
The reference KV append interval is larger, at 20.786 ms.
The source contains full-prefix `torch.cat` operations.
The optimized native cache already uses paged writes.
This generic cache-manager difference is not a new recurrent architecture contribution.

The eager trace also contains launch and idle/unresolved intervals.
Nsight overhead is +56.879% at 1K and +13.363% at 16K.
Both exceed the 10% validity limit.
Those intervals cannot establish a clean removable architecture budget.
Actual HBM traffic, L2 behavior, and occupancy remain unavailable because counter permission was denied.

## Prefill and decode results

The following exact-reference values average frozen request-group medians.
They contain 32 fixed-input cached forwards, not natural output generation.

| Context | B1 prefill, ms | B1 cached step, ms | Logical KV, GiB |
|---|---:|---:|---:|
| 1K | 85.037 | 71.950 | 0.75 |
| 4K | 177.383 | 63.147 | 3.00 |
| 8K | 376.285 | 66.704 | 6.00 |
| 16K | 880.956 | 65.498 | 12.00 |

The 1K group retains startup variation.
The broad sweep and light component lane have different warmup/allocation boundaries.
Their total latencies must not share a denominator.
The full B1/B2/B4/B8 matrix and request-group bootstrap intervals are in `profiling.md` and the CSV files.

The following optimized native TP4 results are hardware diagnostics.

| Context | TTFT, ms | Cached TPOT, ms |
|---|---:|---:|
| 1K | 62.870 | 7.747 |
| 4K | 73.037 | 8.125 |
| 8K | 107.446 | 8.813 |
| 16K | 206.178 | 9.717 |

These are not candidate speedups over HF.
The native path exceeds the frozen mean-logit tolerance despite matching two 64-token natural prefixes.
It remains `REFERENCE_PARTIAL` for exact cross-backend comparison.

## Independent track decisions

| Track | Question | Supported finding | Unsupported claim | Decision |
|---|---|---|---|---|
| A | Can recurrence-aware KV/attention distribution beat a matched generic baseline? | Exact conventional CP runs on all four GPUs. Its bounded implementation is slower | No recurrent-specific delta or integrated speedup | NO_GO, A_NO_BOTTLENECK for the measured decode core |
| B | Can exact depth–chunk scheduling exceed ordinary causal PP? | Legal concurrency exists. The fine DAG maps exactly to generic unrolled PP | No extra loop-specific schedule benefit. No qualified physical pipeline | GENERIC_ONLY, physical benefit INCONCLUSIVE |

Track A's ideal four-way-core reduction is approximately 5.5% in the valid decode lane.
Its measured-operator cost model is slower than that reference.
This result does not rule out every optimized CP implementation.
It does fail to justify a new recurrent-specific architecture here.

Track B has conditional generic prefill overlap in some modeled cases.
For example, 4K/1,024-token chunks model 104.244 ms with layer-stationary placement.
That is not a measured execution.
It does not beat the diagnostic TP4 TTFT of 73.037 milliseconds.
Smaller-chunk full-KV/logit checks also fail the frozen numerical gate.
The model therefore cannot support a pipeline performance claim.

## Communication and realizability

Actual P2P copies cover 12 directed paths and nine payload sizes.
The small activation sizes incur roughly 0.054–0.073 ms host-completion cost per transfer.
Large 256 MiB copies require roughly 0.730–0.749 ms.
These measurements include software and synchronization.
They are not pure NVLink bandwidth measurements.

An actual pretrained MLP plus 64 MiB peer-copy experiment saves about 9.7% versus isolated-time addition.
It does not approach free overlap.
Both architecture models retain measured communication, fill/drain, and chunk-shape costs.
Neither model supplies a measured full-request improvement.

## Questions, evidence, and actions

| Question | Evidence | Status | Next action |
|---|---|---|---|
| Does recurrence create storage/work asymmetry? | Fixed weights, R-scaled KV/executed layers | Supported | Keep separate memory and traffic accounting |
| Does that asymmetry prove an attention bottleneck? | Small decode-core interval, larger generic cache append | Unsupported as a new architecture premise | Do not build CP engine |
| Is exact prefill concurrency legal? | Dependency tests and generic edge isomorphism | Supported structurally | Treat as generic causal PP |
| Does a loop-specific delta remain? | Same-resource graph/schedule identity | Not demonstrated | Do not build wavefront engine |
| Is clean exclusive critical-path attribution complete? | Heavy observer failure, partial native alignment | Inconclusive | Retain PARTIAL/UNRESOLVED labels |
| Are new execution paths numerically qualified? | Native and smaller-chunk tolerance failures | No | Resolve fidelity before a separate performance PoC |
| Is there a measured new E2E speedup? | No candidate engine | NOT_RUN | Make no speedup claim |

## Limitations and next research action

This is a calibration discovery study with eight frozen question IDs.
It has no untouched held-out confirmation or full benchmark quality campaign.
Long contexts use controlled token shapes, not independently checked long-context reasoning tasks.
The strongest exact same-four-GPU comparison remains unestablished.
The study therefore does not prove that every future recurrent architecture lacks headroom.

Do not continue either tested track as a new recurrent parallelism method.
If a follow-up is necessary, first isolate the native/HF numerical mismatch on the existing fixed probes.
That is a correctness diagnostic, not permission to construct a new engine.
Only a later qualified native profile could justify another architecture discovery question.

## Delivery

Branch: `flashvep/recurrent-memory-compute-asymmetry-poc`.
The final commit identifies itself in Git and in the final handoff message.
`reproduce.md` contains commands and artifact locations.
`metrics/completion_audit.json` records original-worktree preservation and final GPU burner state.
Only this PoC directory belongs to this change.
