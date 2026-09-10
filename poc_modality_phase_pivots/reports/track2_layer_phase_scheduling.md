# Track 2 — Modality-Aware Layer/Phase Scheduling

**Decision: NO-GO.** The zero-contention oracle is large but physically false;
the contention-corrected request-chain oracle is only 0.18% of modeled
Attention+MoE makespan and about 0.13% of measured aggregate clean TTFT.

## Cost atlas

Both inputs had 2,363 actual LM tokens. Rank-critical medians were summed over
48 layers:

| Modality | Attention | Dispatch | Expert | Combine | full MoE | layer total |
|---|---:|---:|---:|---:|---:|---:|
| Text-heavy | 41.83 ms | 34.20 ms | 40.42 ms | 20.14 ms | 101.83 ms | 145.70 ms |
| Vision-heavy | 41.24 ms | 32.18 ms | 39.61 ms | 18.96 ms | 98.56 ms | 142.85 ms |

The per-layer natural execution units were `Attention_l` and `MoE_l`; no token
chunking or kernel fragmentation was introduced. A precedence-constrained DP
scheduled two complete request chains.

## Oracles

| Policy | Modeled makespan | Gain vs serial |
|---|---:|---:|
| FCFS/static serial | 283.467 ms | 0.000% |
| Perfect phase-aware oracle (`Txy=max(Tx,Ty)`) | 201.222 ms | 29.014% |
| Contention-corrected layer oracle (fresh measured eta) | 282.969 ms | 0.176% |
| Simple `eta>=0.2` pairing heuristic | 283.467 ms | 0.000% |

The perfect oracle removes 82.245 ms only by assuming away the contention that
the Track-1 experiment directly measured. The contention-corrected saving is
0.498 ms. Dividing by the two measured clean TTFTs (129.014 + 264.979 ms) gives
an Amdahl-style projected aggregate TTFT improvement of **0.126%**. This is an
analytical projection, not an observed request-level speedup.

## Modality value and trivial-fix attack

At equal actual token count, attention and MoE cost profiles differed by only a
few percent and the pairwise eta difference was at most 0.040. Consequently,
modality labels add no useful scheduling decision: the simple heuristic chooses
no overlap, which is also the best measured policy. There is no remaining
10–15% modality-aware oracle for a more complex implementation to recover.

Layer-granular execution is also directly adjacent to
[Layered Prefill](https://arxiv.org/abs/2510.08055), while phase disaggregation
and operation pipelines are covered by
[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
and [NanoFlow](https://arxiv.org/abs/2408.12757). Our intended incremental axis
was modality-dependent resource compatibility; that axis was not observed.

## Gate

- Kernel-fragmentation-free projected TTFT oracle >=10%: **FAIL** (0.126%).
- Measured/contention-corrected modeled gain >=10%: **FAIL** (0.176%).
- Zero-contention oracle: 29.014%, but **rejected as physically contradicted**.
- Correctness: **PASS**.
- Recommendation: do not build the scheduler.
