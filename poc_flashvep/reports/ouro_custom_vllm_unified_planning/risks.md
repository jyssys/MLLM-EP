# Risk register and research value

## Highest-risk decision

The full eight-arm target is not a small model-file patch.
Linear Loop PP is a bounded candidate. Cyclic physical PP and Dynamic token microsteps can require substantial worker and runner changes.
Do not approve a replacement serving engine under the name of a vLLM adapter.

| ID | Risk and evidence | Impact | First cheap test | Mitigation / stop |
| --- | --- | --- | --- | --- |
| R1 | Static Loop PP equals contiguous PP over 96 tied logical layers | Novelty may disappear | Compare exact logical DAG and parameter ownership | Treat generic PP as a mandatory control. Do not claim a new partition |
| R2 | Native workers assume a forward chain and final-rank output | Physical PP wraparound may need extensive control changes | Two-rank cyclic reentry and cancellation test | Stop the arm if bounded native integration fails |
| R3 | Native runner requires sampling before another model call | Loop microsteps can corrupt scheduler token accounting | CPU state-machine trace, then one-token P2 test | Keep recurrence internal to one token completion |
| R4 | Native Ouro gate assumes parallel input while unused | Activating it can break TP shapes or bias semantics | Gate TP1/TP4 operator comparison | Audit input split, tuple return, reduction, and single bias |
| R5 | RLT and CDB differ in sigmoid dtype and probability arithmetic | Exit depth changes despite identical weights | CPU gate fixtures near q=.5, then bounded GPU probes | Use pinned CDB semantics. Preserve differences in reports |
| R6 | Dynamic survivors do not match whole-R4 graph execution | Reported early exit may save no GPU work | Count executed layer bodies and active rows | Depth-level graph scopes plus static graph-cut control |
| R7 | Last-exited pages use backend-specific strides | Full-page copy can corrupt historical KV | Row-repair fixture across a page boundary | Native row views or slot-mapped writes, no blind page replacement |
| R8 | An enqueued write is mistaken for completed publication | Stale or missing KV reaches attention | Inject delayed transfers and force dependent reads | Receiver-local completion events and valid-read checks |
| R9 | Source buffers or pages are reused during transfer | Intermittent corruption under load | Cancellation/preemption/reuse stress test | Allocation-generation keys and page leases |
| R10 | Strict no-refill barrier includes all repair ACKs | Async calls expose no useful overlap | Trace coda, wave release, repair completion, successor R1 | Separate barrier and copy factors. No API-only claim |
| R11 | Copy/compute contention removes predicted savings | Modeled overlap does not improve E2E | Serial, isolated, and concurrent transfer/compute tests | Include contention in the bound. Stop below practical gain |
| R12 | Early exits leave R3/R4 owners idle | Loop PP throughput can be worse than physical PP | Measure real per-rank kernels and queues | Report imbalance. Do not train a new gate or silently change placement |
| R13 | Dynamic exit coda is not on native final rank | Sampling/result routing can serialize execution | R2/R3 exit with natural successor | Explicit coda-owner plan and counted hidden/control transfer |
| R14 | Full-core replicas reduce free HBM | Logical KV split may not improve capacity | Per-rank parameter storage and page reserve audit | Measure safe frontier. Avoid theoretical capacity claims |
| R15 | PP B1 stages and native packed batches differ | Apparent architecture gain can be scheduler gain | Record active/padded rows and batch policy | Matched-admission lane plus strongest native serving lane |
| R16 | TP hidden sensitivity remains unresolved | Accuracy can run without full numerical qualification | Same-backend and FP32 diagnostic ladder | Keep exploratory label. Never erase the old FAIL |
| R17 | Heavy profiling perturbs latency | Kernel service sums misrepresent critical path | Paired clean/traced cohort | Use clean client clocks. Keep profiles explanatory |
| R18 | Small quality and tail cohorts lack power | False equivalence or unstable P99 | Bootstrap and sample-count report | Expand only necessary tests. Mark uncertainty |
| R19 | Native vLLM version is pinned to a removed model path | Upgrades can drop Ouro support | Validate v0.26.0 source and wheel identities | Isolated fork and environment. No global package update |
| R20 | Private RLT kernels and cache schema are copied wholesale | Native performance path is lost | Function-by-function portability review | Port semantics, not the runtime |
| R21 | Git contains extensive unrelated cleanup and staged work | A commit can include unrelated data loss | Compare staged fingerprints and commit path list | Commit only this new folder. No reset, clean, or force push |
| R22 | A single request's stage dependencies remain serial | PP may help throughput but not C1 latency | Measure C1 and C4 separately | Do not infer B1 speedup from cross-request kernel overlap |

## Relative scope and review ownership

| Work package | Relative size | Required review |
| --- | --- | --- |
| Native baseline harness and numerical attribution | M | Measurement and model semantics |
| Static linear PP adapter and ownership | L | Native PP and model semantics |
| Cyclic physical PP and worker reentry | XL risk | Distributed execution and scheduler |
| Native Dynamic gate and last-exited rows | L | Numerical policy, cache, and TP |
| Dynamic PP coda and repair protocol | L | Distributed lifecycle and cache safety |
| Async deferred publication | M prototype, L integration | Streams, transport completion, and timing |
| Main serving/quality evaluation | L | Benchmark methodology and statistics |

These are estimates of complexity, not a promise of calendar duration.
Do not begin all work packages in parallel before their gates.
One reviewed minimal patch should demonstrate each contract before the next patch.

## Research value and overlap with prior mechanisms

Native TP, PagedAttention, linear PP, and distributed softmax are existing mechanisms.
Reusing them is correct engineering, not new research by itself.
Static Loop PP maps exactly onto contiguous logical-depth PP in this R4 layout.
The experiment must acknowledge that equivalence even if the resulting deployment is useful.

CDB already addresses token-level variable-depth batching and cache management.
RLT provides recurrent scheduling and cache lifecycle designs in a separate runtime.
The proposed contribution cannot be the original gate, generic refill, or the term loop-level scheduling.

The narrower unresolved question is physical early-exit publication across depth owners.
Its potential value is exact overlap between repair of an older row and legally independent earlier-depth work.
This may still be generic dependency-aware overlap. Establish the measured Ouro-specific regime before claiming broader novelty.
Cross-GPU aliasing is a separate candidate with different memory and remote-read costs.

## Reasons to terminate the program

- Native TP qualification remains unexplained or unstable.
- Static PP requires an engine rewrite instead of bounded adapters.
- Static Loop PP lacks a useful region against native TP4 and generic contiguous PP.
- Dynamic rank imbalance or repair costs exceed any measured legal overlap opportunity.
- A new execution mechanism changes quality beyond its accepted margin.
- The only improvement comes from generic graph cleanup, different batching, or excluded preparation time.

Preserve any useful negative result. A justified stop is preferable to a large unqualified fork.
