# DeepEP wait-aware tail PoC specification

## Objective

Measure whether an unsafe/outstanding DeepEP dependency can be detected before
the next MoE dispatch and selectively drained, reducing extreme dispatch tails
without the p50/throughput cost of an unconditional device synchronization.
This is a bounded measurement/policy prototype, not a production scheduler.

## Environment

- `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical GPUs 1--4 only)
- Qwen3-VL-30B-A3B-Instruct, BF16
- vLLM 0.20 V1, TP2/DP2/EP4/PP1
- DeepEP high-throughput, Triton unquantized experts, linear placement
- DBO off, prefix cache off, eager path

## H0--H10

| ID | Hypothesis | Falsifiable prediction |
|---|---|---|
| H0 | A readiness signal cannot distinguish tails. | `P(>10 ms|ready)` is comparable to `P(>10 ms|not-ready)` and no monotonic wait relation exists. |
| H1 | `previous_event`/communication-stream backlog is causal. | Event-not-ready or wait duration predicts dispatch excess; relevant drain reduces tail rate. |
| H2 | Relevant-event wait is sufficient. | Event-only or communication-stream-only wait matches global sync tail reduction with lower p50 cost. |
| H3 | Reported dispatch improvement merely moves the wait earlier. | Dispatch shrinks but request/step latency and TTFT do not improve. |
| H4 | Readiness instrumentation perturbs the runtime. | Query-on/off paired runs change p50 or tail rate materially without intervention. |
| H5 | An actual expert/workspace kernel, not dependency state, causes tails. | Fixed-route expert replay remains tail-heavy and expert duration is the first divergence. |
| H6 | Combine/finalize is the causal wait point. | Combine is first divergence and combine-only intervention removes tails. |
| H7 | Global GPU interference explains the event. | Attention/non-MoE and unrelated CUDA activity co-tail; local DeepEP drain does not selectively help. |
| H8 | DP/EP peer synchronization amplifies the backlog. | Both TP ranks in the affected DP group co-tail and DeepEP peer notification/barrier kernels are present. |
| H9 | Tail is intrinsic to route/shape. | Exact route/input replay remains slow under a persistent worker. |
| H10 | DBO/async overlap changes the mechanism. | Tail/readiness behavior differs materially with DBO control; if DBO-off still tails, mechanism is not DBO-specific. |

## Instrumentation hierarchy

Every record carries run id, local step/invocation id, route id, layer, DP/TP/EP
rank, phase, M, and previous-state fields. CUDA events are recorded on the
same device. No cross-GPU timestamp subtraction is allowed.

At the dispatch boundary record whether a previous DeepEP event exists, a
nonblocking readiness result if available, event-to-ready duration (diagnostic
only), communication-stream proxy, previous dispatch/expert/combine/whole
duration, gap, and current shape. Instrumentation overhead is measured with
the observer disabled/enabled.

## Policy definitions

- **P0 STOCK**: unmodified validated vLLM/DeepEP path.
- **P1 ALWAYS_SYNC**: diagnostic `torch.cuda.synchronize()` before MoE.
- **P2 ORACLE_SELECTIVE_SYNC**: offline ground truth selects only invocations
  with measured dependency wait or known-not-ready state.
- **P3 ONLINE_SIMPLE_POLICY**: uses only state available before dispatch and a
  threshold fixed in a small calibration run; no ML or scheduler rewrite.

Prefer relevant event wait, communication-stream drain, or peer dependency
wait over global synchronization. Every candidate must report E2E latency,
TTFT/TPOT, throughput, p50/p99, and tail frequencies (>5/10/20/50/100 ms).

## Required controls and gates

Use decode-heavy continuous batching (thousands of M=1 observations), plus a
bounded mixed/prefill validation. Run each policy at least three independent
times with identical warmup and randomized order. Use fixed-route replay to
separate intrinsic cost from online state.

`STRONG_GO` requires >=80% reduction in >20 ms events, >=20% p99 improvement,
<=2% throughput loss, no meaningful p50 regression, and P3 recovering >=60%
of P2's benefit. `GO` requires >=50% extreme-tail reduction, >=10% p99
improvement, and <=3% throughput loss. If readiness does not predict tails,
if only global sync helps, or if E2E does not improve, declare the exact
NO_GO subtype rather than claiming a method win.

## Deliverables

Maintain `EVENT_READINESS_ANALYSIS.md`, `POLICY_COMPARISON.md`,
`NSIGHT_VALIDATION.md`, `PRIOR_ART_NOTES.md`, and `EXPERIMENT_LOG.md` in the
timestamped result directory. Preserve raw stage traces and write the final
report to `poc_flashvep/reports/deepep_wait_aware_tail_poc.md`.
