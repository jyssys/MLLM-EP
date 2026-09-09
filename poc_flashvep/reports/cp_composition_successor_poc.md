# Context Parallelism Composition Successor PoC

Date: 2026-09-09

Branch: `flashvep/cp-composition-successor-poc`

Result root: `poc_flashvep/deepep_revalidation/results/cp_composition_successor_poc_20260909_174613/`

## Executive decision

**FINAL STATUS: `PARTIAL_ENVIRONMENT_BLOCKED`**

No paper-level successor was found.

- **Track C (MoE-aware CP degree)** had the strongest valid request-level signal: the per-regime PCP lower envelope was 15.42% faster than the best static degree. It fails the research gate because the trivial rule “PCP4 for concurrency one, PCP1 otherwise” recovered 100% of the oracle. At matched 8K/16K token counts and concurrency two, natural/code/math/repetitive content all selected PCP2 with 28.65--38.94% separation. There is no evidence that MoE routing changes the choice.
- **Track A (DCP × speculative decoding)** showed 14.46% lower median E2E for combined DCP+spec than spec-only and a 4× KV-capacity effect, but cross-DCP greedy agreement failed. Those runs are excluded from positive evidence. The exact problem/strategy space is also already active upstream.
- **Track B (PCP perpendicular DCP / zero-transition)** is already structurally addressed by current PCP canonical global slot writes. The working PCP-only and DCP-only paths showed 0% clean first-decode transition excess. The PCP4+DCP4 combination stalled in warmup in the newest driver-compatible runtime, so the overall study is partially environment-blocked rather than reported as `ALL_THREE_NO_GO`.

## Scope and evidence quality

### Hardware safety

All task-owned GPU execution used `CUDA_VISIBLE_DEVICES=4,5,6,7`, mapping logical ranks 0--3 to physical H100s 4--7. Physical GPU UUIDs were recorded before execution. Processes on GPUs 0--3 were inspected and left untouched. The task left GPUs 4--7 empty before final artifact generation.

Logged live GPU time is 2,372 seconds (39.53 minutes), or 2.64 four-GPU hours. Early hard gates ended the study without manufacturing extra repetitions.

### Runtime versions

- Current source/novelty audit: vLLM main commit `5acd95906b7c7a54dde89396d2bb06fe28ebeed0` (2026-09-09 checkout).
- Main source's CUDA-13/PyTorch-2.13 binary stack is incompatible with this host's CUDA-12.8 driver.
- Decisive compatible execution: isolated vLLM 0.26.0+cu129, PyTorch 2.11.0+cu129.
- Initial compatibility reference: vLLM 0.20.0, PyTorch 2.11.0+cu129.
- Measurement model: cached `deepseek-ai/DeepSeek-V2-Lite-Chat`, BF16, MLA, 64 routed experts, top-6, EP off.

Qwen3-VL-30B-A3B-Instruct is not a faithful four-GPU CP target in current vLLM: its GQA has four KV heads, so TP4 cannot satisfy the requirement that TP be greater than KV-head count for DCP; current PCP is MLA-only and rejects multimodal inputs. No fake CP or production claim was substituted. This makes the study generic MoE rather than MLLM-specific.

## Current-upstream audit

The source audit and current upstream discussions establish:

1. PCP and DCP are separate axes with backend/model constraints, as documented in [vLLM's CP deployment guide](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/).
2. [RFC #50391](https://github.com/vllm-project/vllm/issues/50391) already states the DCP speculative-verification problem and exact strategies including virtual single-query rows, context/current split with LSE merge, and a DCP-aware varlen path. [Issue #45425](https://github.com/vllm-project/vllm/issues/45425) documents related silent output corruption, and [RFC #53673](https://github.com/vllm-project/vllm/issues/53673) addresses draft/target parallelism.
3. [RFC #46358](https://github.com/vllm-project/vllm/issues/46358) explicitly proposes PCP perpendicular DCP. [Issues #51429](https://github.com/vllm-project/vllm/issues/51429) and [#53573](https://github.com/vllm-project/vllm/issues/53573) cover validator and combined-MLA collective correctness.
4. Current `PCPManager` rejects multimodal and speculative workloads and supports MLA only. It computes DCP-aware global slot mappings and masks duplicate PCP writes, already expressing canonical final KV ownership.
5. An attention backend without `supports_dcp_with_varlen` forces the speculative multi-query batch away from the decode fast path by reducing the reorder threshold to one. DCP+spec is backend-specific, not a universal flag.

The full audit is in `UPSTREAM_STATUS.md`, `SUPPORT_MATRIX.csv`, and `PRIOR_ART_MATRIX.md`.

## Track A — DCP × speculative decoding

### Support/topology

Faithful fallback topology: TP4, PCP1, DCP1 or DCP4, EP off. N-gram draft length was three. The v0.26 runtime falls back to Model Runner V1 for n-gram.

### Clean request metrics

Eight measured requests after warmup:

| Path | E2E p50 | TTFT p50 | TPOT p50 | Throughput |
|---|---:|---:|---:|---:|
| Vanilla | 3575.92 ms | 159.55 ms | 54.23 ms | 143.86 tok/s |
| DCP4 only | 6291.71 ms | 284.88 ms | 95.35 ms | 81.60 tok/s |
| Spec only | 4561.33 ms | 846.32 ms | 59.26 ms | 110.90 tok/s |
| DCP4 + spec | 3901.70 ms | 133.27 ms | 60.20 ms | 128.85 tok/s |

Observed, not an oracle: combined was 37.99% faster than DCP-only and 14.46% faster than spec-only, but 9.11% slower than vanilla.

### Correctness

Cross-DCP agreement failed: vanilla/DCP-only agreed for 2/8 outputs; spec-only/combined agreed for 3/8. Within a fixed DCP degree, the corresponding no-spec/spec pair agreed 7/8. The failure is deterministic for duplicated prompts. It could be numerical sensitivity, but it violates the required greedy/token agreement, so none of these speed differences supports GO.

### Capacity and oracle

The v0.20 compatibility run directly logged 138,080 versus 552,320 KV tokens and 16.86× versus 67.42× nominal 8K concurrency: DCP supplied +300% capacity. The raw combined-versus-spec request gap is 14.46%, but there is no correctness-valid successor oracle.

### Prior art and gate

Direct collision: #50391 already owns the exact semantic issue and the portable exact solution space. **Gate: NO_GO** due to correctness failure, active upstream collision, and combined being slower than vanilla at the measured load.

## Track B — PCP perpendicular DCP / zero-transition

### Support/topology

- PCP-only: TP1/PCP4/DCP1.
- DCP-only: TP4/PCP1/DCP4.
- Combined: TP1/PCP4/DCP4, admitted by configuration validation but blocked in warmup.

### Correctness and clean request metrics

Exact 16K prompt plus 32 decode tokens, three measured repetitions after two warmups:

| Path | TTFT p50 | E2E p50 | First ITL | Steady ITL p50 | Transition excess |
|---|---:|---:|---:|---:|---:|
| PCP-only | 99.28 ms | 1950.19 ms | 41.03 ms | 59.55 ms | 0.00 ms |
| DCP-only | 103.08 ms | 2128.45 ms | 29.40 ms | 65.74 ms | 0.00 ms |

The two working paths produced identical greedy outputs. Large cold first-ITL spikes disappeared after warmup; they are not counted as removable transition work.

### Capacity, oracle, and blocked point

Current source has canonical global DCP-aware slot ownership and no obvious transition-copy byte mass. The direct clean transition share in both working paths is 0%. A 16K combined run and bounded 8K retry loaded weights but did not finish warmup; the latter was stopped after 180 seconds. This is a compatible-runtime integration block, not evidence of a method failure.

### Prior art and gate

PCP perpendicular DCP is active upstream (#46358), and the zero-transition slot principle already exists. **Gate: NO_GO for the successor premise; `ENVIRONMENT_BLOCKED` for combined runtime evidence.**

## Track C — MoE-aware CP degree

### Support/topology

Four-GPU fixed fleet, EP off:

- PCP1 = four independent one-GPU replicas.
- PCP2 = two independent two-GPU replicas.
- PCP4 = one four-GPU replica.

Each regime used exact prompt token IDs, matched arrival, two warmups, three measurements, and one-token greedy correctness. Outputs agreed across PCP degrees.

### Request-level screen

Median fleet wall, milliseconds:

| Actual tokens | Concurrency | PCP1 | PCP2 | PCP4 | Winner |
|---:|---:|---:|---:|---:|---:|
| 8192 | 1 | 119.60 | 87.10 | **86.71** | PCP4 |
| 8192 | 4 | **121.87** | 168.24 | 235.38 | PCP1 |
| 8192 | 16 | **457.34** | 572.85 | 750.75 | PCP1 |
| 16384 | 1 | 256.39 | 168.16 | **117.15** | PCP4 |
| 16384 | 4 | **259.78** | 315.84 | 421.82 | PCP1 |
| 16384 | 16 | **969.61** | 1225.81 | 1493.10 | PCP1 |
| 32768 | 1 | 600.11 | 377.97 | **249.30** | PCP4 |
| 32768 | 4 | **605.68** | 714.17 | 914.72 | PCP1 |

### Oracle and trivial-fix attack

- Best static PCP1 aggregate: 3390.38 ms.
- Perfect per-regime oracle: 2867.44 ms, **15.42%** gain.
- Best length-only selector: 3276.72 ms; oracle adds **12.49%**.
- Simple concurrency rule (`c=1 → PCP4`, otherwise PCP1): 2867.44 ms, **100% oracle recovery**.

The high raw headroom is therefore real deployment headroom but not a MoE-aware successor problem.

### Matched content causal control

At exact 8K/16K, concurrency two, natural/code/math/repetitive content all selected PCP2. PCP2 beat second place by 35.33--38.94% at 8K and 28.65--29.56% at 16K. Content-induced routing differences did not change the optimal degree. Router histograms were not instrumented because this decisive request-level kill made deeper attribution unwarranted.

### Capacity, prior art, and gate

PCP4 reduces isolated-request TTFT; PCP1 preserves fleet replication. That is the expected latency/capacity trade-off. No MLLM claim is possible because Qwen3-VL PCP is unsupported. **Gate: NO_GO** because an obvious existing deployment rule recovers all measured oracle value.

## Ranking

| Rank | Track | Score / 50 | Research status |
|---:|---|---:|---|
| 1 | C — MoE-aware CP degree | 31 | NO_GO: trivial concurrency rule |
| 2 | A — DCP × speculation | 22 | NO_GO: correctness + prior art |
| 3 | B — zero-transition | 17 | premise NO_GO; combined environment blocked |

No winner was deep-dived because none passed correctness, non-triviality, headroom, and novelty together.

## Limitations and conservative interpretation

- Current-main source and current issue state were audited, but live execution used the newest driver-compatible v0.26 runtime rather than current-main binaries.
- Strong comparisons would normally require three engine restarts. Hard kill gates appeared in the first restart, so extra restarts would add confidence to operational numbers but cannot repair correctness, triviality, or prior-art collision.
- Track C is generic MoE/MLA, not Qwen3-VL or modality evidence.
- Track B combined failure is not classified as the same mechanism as #53573 without a completed trace.
- SLO goodput was not derived; Track C reports fleet completion and prompt throughput for matched traces, while Tracks A/B report direct request metrics.

## Final recommendation

Do not pursue any of these three as a paper-level successor on the current four-GPU single-node platform. Operationally, use a simple concurrency-aware PCP deployment rule and treat DCP as a capacity lever. Revisit combined PCP+DCP only after upgrading to a driver-compatible current runtime—and then as upstream correctness/engineering validation, not as a zero-transition novelty claim.

## Reproduction index

- Scripts: `poc_flashvep/cp_composition_successor/run_track_a.py`, `run_track_b.py`, `run_track_c.py`, `analyze_results.py`.
- Aggregates: `analysis/track_a_request_matrix.csv`, `track_b_transition.csv`, `track_c_regimes.csv`, `analysis_summary.json` under the result root.
- Raw request outputs, runtime proofs, logs, environment installation logs, and all summaries are preserved under the result root.
