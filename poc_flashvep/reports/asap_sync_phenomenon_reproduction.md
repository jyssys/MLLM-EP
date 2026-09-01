# ASAP-style DP→EP synchronization phenomenon reproduction

## Executive result

**FINAL STATUS: HOLD**

`MEASUREMENT_VALIDATION: PASS` for the calibrated positive control, but the
natural controlled workload did not reproduce a stable barrier penalty across
repetitions.  The first TP1/DP4/EP4 4096-token heterogeneous run was 27.5%
slower than its balanced control, while a repeat in the opposite run order was
30.9% faster and a reverse-order pair was 8.8% slower.  This direction reversal
means the attractive first result cannot be used as robust evidence of an
ASAP-style synchronization stall.  The bounded evidence supports a
**scale-limited/runtime-variable** interpretation, not a reproduced generic
barrier.

No scheduler, router, placement, kernel, or model change was made.

## Environment and topology

| item | value |
|---|---|
| model | Qwen3-VL-30B-A3B-Instruct (BF16) |
| vLLM / PyTorch / Triton | 0.20.0 / 2.11.0+cu129 / 3.6.0 |
| GPUs | 4× H100, physical 1,2,3,4 only |
| DBO / prefix cache | off / off |
| backend | DeepEP high-throughput + TritonExperts |
| execution | eager, V1 scheduler, async scheduling enabled |
| chunked prefill | on for the primary runs, `max_num_batched_tokens=8192` |
| topology A | TP2 / DP2 / EP4 / PP1; sequence-parallel MoE true |
| topology B | TP1 / DP4 / EP4 / PP1; sequence-parallel MoE false |

Runtime topology proofs (physical GPU, PID, TP/DP/EP groups and world sizes)
are in the final result directory.  In both topologies the EP group is all
four visible ranks; A has two-rank DP groups and B has four one-rank DP groups.

## ASAP reference and code status

The paper is *A Disaggregated and Asynchronous Inference System for MoE
Prefill* ([arXiv:2606.22541](https://arxiv.org/abs/2606.22541)).  Its baseline
characterization is a synchronous DP-attention/EP-MoE boundary where request
length variance produces DP progress imbalance and a synchronization bubble,
reported through TTFT/throughput decomposition.  The bounded search of the
author page ([sc2682cornell/sc2682cornell.github.io](https://github.com/sc2682cornell/sc2682cornell.github.io))
and public repositories found no official ASAP implementation:
`ASAP_CODE_STATUS: NOT_PUBLICLY_FOUND`.

| dimension | ASAP characterization | this PoC |
|---|---|---|
| model | paper workload/model | Qwen3-VL-30B-A3B-Instruct |
| hardware | paper system | 4× H100 |
| DP/EP | paper DP-attention + EP-MoE | TP2/DP2/EP4 and TP1/DP4/EP4 |
| length regime | heavy-tailed sequence lengths | controlled 4K/8K text compositions |
| prefill | synchronous baseline and chunked comparison | vLLM V1, chunked on; off only feasible at 16K budget |
| metric | TTFT/throughput and synchronization decomposition | wall prefill, CUDA stage spans, closest collective wait proxies |

## Source-level synchronization audit

The installed `deepep_ht.py` path and line-level notes are captured in
`results/..._final/source_audit.md`.  In summary, vLLM's
`DeepEPHTPrepareAndFinalize` captures a compute-stream `previous_event`, passes
it to `get_dispatch_layout` and `buffer.dispatch` on the communication stream,
then waits on the returned `EventOverlap` before consuming expert inputs.
Finalization passes a `previous_event` to `buffer.combine` and waits before
copying the output.  This is implicit collective + CUDA stream/event
dependency, not an explicit Python/global barrier in this class.  vLLM logs
also report asynchronous scheduling and disabled NCCL DP synchronization.

Thus a cross-rank duration spread is not by itself a direct synchronization
wait.  CUDA timestamps from different GPUs were not subtracted as a global
clock.  `event_wait_cuda_ms` is retained as an asynchronous wait-enqueue
diagnostic; `prepare_host_ms`, dispatch CUDA span, and `ep_entry_to_done_ms` are
the closest complete collective-span proxies available without changing the
execution path.

## Stage A — positive-control measurement validation

The calibrated H100 delay kernel measured 0.515, 1.020 and 2.030 ms for
requested 0.5, 1 and 2 ms.  At layer 24 in TP1/DP4/EP4, injected delay was
placed before the MoE entry on DP0.  The peer ranks' dispatch/prepare spans
rose with the injected delay (approximately 0.29/0.34 ms at 0 ms,
0.58–0.62/0.58–0.62 ms at 1 ms, and 1.54–1.60/1.59–1.64 ms at 2 ms).
This is a monotonic response and passes the measurement positive control.
The direct `EventOverlap.current_stream_wait()` enqueue duration stayed near
0.02–0.04 ms, confirming that it must not be labeled as a complete wall-clock
stall.  The full calibrated table and plot are in
`injected_skew_validation.csv` and `plot1_injected_skew_validation.png`.

## Stage B — controlled sequence-shape workload

The request schedule held nominal token volume per DP equal while changing
composition:

* balanced: four requests of 1,024 tokens for 4K (or four of 2,048 for 8K);
* heterogeneous: one request of 4,096 (or 8,192) on DP0 and eight requests of
  512 (or 1,024) on each other DP.

The scheduler traces show the realized prefill chunks, including the one-token
decode request: balanced 4K uses approximately 1,025 + 3,075 scheduled rows;
heterogeneous DP0 uses 4,097 while the other ranks use 513 + 3,591.  These
traces are preserved and indexed in the final result.

### End-to-end paired results

| topology / scale | balanced wall ms | heterogeneous wall ms | heterogeneous − balanced |
|---|---:|---:|---:|
| TP2/DP2, 4K | 3229.1 | 3306.0 | +2.38% |
| TP2/DP2, 8K | 3299.9 | 3386.7 | +2.63% |
| TP1/DP4, 4K (first) | 2834.4 | 3614.4 | **+27.52%** |
| TP1/DP4, 4K (same-order repeat) | 4287.8 | 2961.3 | **−30.94%** |
| TP1/DP4, 4K (reverse-order repeat) | 3335.4 | 3627.7 | +8.77% |
| TP1/DP4, 8K | 2941.2 | 3173.7 | +7.90% |

The first 4K DP4 result is an important candidate regime, but the two
repetitions show that it is not stable under the same fixed inputs and
configuration.  The run-level wall time is therefore reported as evidence of
runtime variance, not as a replicated causal barrier.

### CUDA stage and closest wait proxies

In the first DP4/4K pair, heterogeneous versus balanced medians were:

| stage | balanced ms | heterogeneous ms |
|---|---:|---:|
| pre-MoE CUDA | 1.070 | 1.440 |
| host prepare span | 0.343 | 0.803 |
| dispatch CUDA | 0.297 | 0.754 |
| expert CUDA | 0.382 | 0.396 |
| combine CUDA | 0.117 | 0.290 |
| EventOverlap enqueue wait | 0.015 | 0.206 |
| EP entry→done | 0.584 | 0.769 |

The elevated prepare/dispatch/combine spans and p95 rank spreads are
consistent with a possible synchronization/context effect in that run, while
expert median itself changes little.  However, the repeated runs show similar
large rank spreads in both conditions and reverse the wall ordering, so these
spans are not sufficient to claim a stable DP→EP barrier.

The DP2 pairs were only +2.38% and +2.63%, while DP4 was +7.90% at 8K and
highly variable at 4K.  This is suggestive of a DP-degree/scale interaction,
but not a robust monotonic DP4 amplification.

## Stage C — chunked-prefill ablation

Disabling chunked prefill at `max_num_batched_tokens=8192` is rejected by this
vLLM configuration (`max_num_batched_tokens` must not be below the 16K model
length when chunked prefill is disabled).  A successful bounded comparison at
the required 16K budget gave:

| DP4 heterogeneous 8K | wall ms | prepare spread median ms | dispatch spread median ms |
|---|---:|---:|---:|
| chunked OFF, max batch 16K | 3307.6 | 2.857 | 2.260 |
| chunked ON, max batch 16K | 3093.1 | 2.850 | 2.259 |

OFF was 6.93% slower in this pair and had larger event/compute tails, which is
consistent with chunking mitigating sequence-shape variance.  It is not an
apples-to-apples OFF-at-8K experiment because that setting is unsupported.

## Nsight Systems bounded capture

`nsys` was available (`2024.6.2.225-246235244400v3`).  The preserved small
capture contains DeepEP dispatch/combine kernels, including
`notify_dispatch<4>`, `cached_notify_combine<4>`, `dispatch<4,768,8192>`,
`combine<bf16,4,768,4096>`, and layout kernels.  No custom NVTX ranges were
added and the capture did not establish a reliable per-rank idle interval, so
it is used as backend/timeline proof rather than as a quantitative stall
measurement.

## Decision

`FINAL STATUS: HOLD` (case: **SCALE_LIMITED_OR_RUNTIME_VARIABLE**).

* Measurement validation: **PASS** for calibrated delay visibility.
* Natural ASAP-style reproduction: **not robust**.  The DP4/4K first-run
  +27.5% effect is contradicted by same-order and reverse-order repetitions.
* DP2: small +2–3% paired deltas.  DP4: +7.9% at 8K and one +27.5% 4K run,
  but high run-to-run variance.
* Direct global synchronization wait: **not directly measured**; the current
  vLLM/DeepEP HT path exposes asynchronous event dependencies rather than a
  simple blocking global barrier.
* `WHY_PREVIOUS_NO_GO`: earlier runs were low-load, chunked, used an indirect
  spread proxy, lacked a validated artificial-delay response, and did not
  separate asynchronous enqueue from complete collective wait.  The new
  positive control resolves the measurement concern, while the repeated
  natural runs expose runtime variability rather than confirming a stable
  barrier.

No modality sweep is recommended yet.  `NEXT: MODALITY STUDY NO` until a
generic DP→EP synchronization measurement is made direct and reproducible;
the next single action is a small controlled runtime experiment that captures
per-rank absolute arrival/collective completion with a synchronized clock or
explicit DeepEP completion marker, rather than another broad workload sweep.
