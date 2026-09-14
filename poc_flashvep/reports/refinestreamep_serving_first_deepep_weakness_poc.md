# RefineStreamEP: serving-first DeepEP weakness PoC

## Decision

**NO-SERVING-GAP.** On 4xH100, hidden 4096, 256 experts, top-k 8, BF16 EP4,
legacy DeepEP low-latency remains the strongest valid path for every measured
real LLaDA2 compacted refinement shape. The credible new-kernel serving oracle
is below the mandatory gate, so no CUDA kernel was implemented.

## Evidence boundary

GPU measurements cover DeepEP communication semantics and legal multi-wave
issuance on physical GPUs 4--7. Real routing comes from validated LLaDA2 true
EP4 traces. Online numbers are an EP-stage route-replay harness over real
chronological trajectories; they are not full-model online E2E. The real engine
validation was contingent on a promoted route-replay result and was skipped
after the gate failed.

## Main results

| Question | Result |
|---|---|
| Normal--LL crossover | between global M 4096 and 8192 |
| Crossover in real LLaDA2 range? | no; real compacted M is 1--1024 |
| Capacity mismatch | median 4.65x, p90 57.6x for peak cap 256 |
| Stable latency tax from capacity? | no monotonic effect; noise-scale |
| Capacity memory | cap256 ≈3.03 GiB/rank heap+double-recv |
| Q=16 real mixed routes | LL two-slot 11,705 waves/s; Normal 4,268 |
| Two-slot vs serial LL | +13.1% waves/s |
| Q=8→16 saturation slope | +1.17%, no collapse |
| Existing backend oracle | 0.00% over LL |
| O4 at Poisson 82.5% | +0.16% req/s, -4.53% p99 |
| O4 at Poisson 95% | +0.78% req/s, -6.09% p99 |
| O4 at closed-loop Q32 | +2.19% req/s, -2.14% p99 |

The M=16384 LL endpoint is environment-blocked by legacy V1's internal 32-bit
RDMA-buffer indexing assertion; it is not counted as a Normal win. M=8192 is a
successful measured stress point where Normal wins all route shapes.

## Answers to the required questions

1. Normal crosses LL between M=4096 and M=8192.
2. No, the measured real workload ends at M=1024.
3. Capacity changes memory linearly, but no stable monotonic latency dependence
   was found.
4. With cap256, actual source-rank mismatch is median 4.65x and p90 57.6x.
5. No throughput-limiting serialization was observed under the legal lifetime.
6. LL approaches a plateau around Q=8--16; throughput still rises 1.17%.
7. No valid concurrent real-route point lets Normal overtake LL.
8. Mixed M behaves between the homogeneous controls, without a new reversal.
9. Online route replay reveals no implementation-scale throughput/tail gap.
10. The small tail movement is queueing amplification, not a large backend tax.
11. Perfect existing switching recovers 0.00%.
12. The optimistic variable-capacity ceiling is 1% service-local and <=3.2%
    p99 in Poisson replay.
13. Unlimited-inflight service ceiling is 1.17%; its p99 effect is <=3.3%.
14. Combined O4 is <0.8% throughput and <6.1% p99 under Poisson loads.
15. No implementation gate is cleared.
16. No DeepEP contract becomes a decisive latency bottleneck; peak capacity is
    principally a memory reservation, and two slots sustain throughput.
17. A custom path could be exact, but correctness feasibility is not headroom.
18. Normal/naive gaps are weak-baseline effects relative to LL.
19. No distinct refinement-stream kernel problem was established.
20. No defensible paper-level kernel target remains on this substrate.

## Interpretation

The hypothesis was useful because it separated three often-conflated effects:
large-M crossover, capacity reservation, and in-flight lifetime. Only the
first and the memory portion of the second are real; neither lies on the target
serving critical path. Moreover, current [DeepEP V2](https://github.com/deepseek-ai/DeepEP)
already unifies high-throughput and low-latency operations with an
`ElasticBuffer`, while legacy [DeepEP](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)
documents two-micro-batch overlap. [StreamEP](https://github.com/evolutionaryscale/StreamEP)
and [FlashInfer MoE EP](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_architecture.md)
further raise the novelty bar for generic streaming or elastic-buffer claims.

## Reproducibility

Machine-readable results live under `poc_refinestreamep/`. Raw GPU records are
git-ignored but the aggregate CSVs, real stream, scripts, reports, and figures
are retained. Important results use three independent GPU process restarts;
online simulations use five seeds. All communication outputs pass the numerical
check (worst real-route relative L2 7.71e-4).

## Sources

- [DeepEP current upstream and V2 design](https://github.com/deepseek-ai/DeepEP)
- [DeepEP legacy V1 APIs and two-micro-batch overlap](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md)
- [StreamEP tile-streaming EP implementation](https://github.com/evolutionaryscale/StreamEP)
- [FlashInfer MoE EP architecture](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_architecture.md)
- [NCCL EP paper](https://arxiv.org/abs/2603.13606)
