# RefineEP Deep PoC Specification
## Refinement-Aware Third Expert-Parallel Path for dLLM-MoE

### 0. Research question

Can a deliberately narrow, dLLM-specialized EP dispatch/combine path outperform existing generic DeepEP paths on the fresh-work shapes produced during block-diffusion refinement?

The target is not dynamic configuration alone. The target is a new third path, tentatively called **RefineEP**, specialized for:

- 4 x NVIDIA H100
- single node
- NVLink/NVSwitch only
- EP4
- LLaDA2.0-Flash
- hidden size 4096
- 256 routed experts
- top-k 8
- 64 experts per rank
- BF16
- forward inference only

The research story is:

1. AR inference has two familiar physical regimes:
   - prefill: many tokens, throughput/bandwidth oriented;
   - decode: very few tokens, latency oriented.
2. A block-diffusion model does not literally alternate between prefill and decode.
3. Instead, repeated refinement changes the amount and shape of **fresh routed expert work**.
4. Once liveness-aware compaction materializes that logical change physically, one diffusion block may traverse large, medium, and small EP work regimes.
5. Existing EP runtimes are generic and optimized around broad endpoint workloads.
6. RefineEP asks whether the medium/small changing refinement regime deserves a specialized intranode protocol.

No quality approximation is allowed.

---

## 1. Primary substrate

Use only:

`CUDA_VISIBLE_DEVICES=4,5,6,7`

Primary model:

`inclusionAI/LLaDA2.0-flash`

Primary model placement:

- dense TP4
- routed EP4
- DeepEP dispatch
- owner-local routed expert execution
- reverse combine

Record exact:
- model revision
- runtime branch/commit
- DeepEP commit
- CUDA
- NCCL
- NVSHMEM if present
- GPU UUID
- NVLink topology
- compile flags

Do not use GPUs 0,1,2,3.

---

## 2. Important terminology

Do not say that dLLM literally performs prefill and decode repeatedly.

Use:

- **prefill-like EP shape** = many fresh token rows / large payload
- **decode-like EP shape** = few fresh token rows / startup-sensitive
- **refinement regime** = the continuum between these shapes inside repeated diffusion refinement

The key physical variable is not semantic phase alone, but the actual compacted routed workload:

- fresh rows M
- token-expert pairs
- remote assignments
- remote bytes
- fanout
- rows per expert
- load skew

---

## 3. Relationship to Epoch-like liveness compaction

The current dense runtime may still execute all block rows even when logical decision liveness shrinks.

Therefore do not claim that raw LLaDA2 automatically exposes shrinking physical M.

The kernel opportunity must be studied on the physical worklist that a liveness-aware runtime would expose.

Preferred:
- use an actual Epoch/FreshLane-like compacted worklist if available.

Otherwise:
- derive/replay compacted fresh rows from existing LLaDA2 instrumentation;
- include live/new/refresh-required rows according to the chosen exact semantics;
- document assumptions.

The conceptual layering should be:

`Epoch-like system: what work is fresh?`
`RefineEP: how should that fresh work be physically dispatched/combined?`

---

## 4. Stage 0 — baseline truth

Reproduce the strongest valid LLaDA2.0-Flash EP4 baseline.

Record clean:
- BCT
- throughput
- NFE
- quality
- peak HBM

Verify:
- true remote routing
- 64 experts/rank
- hidden=4096
- top-k=8
- BF16
- GPUs 4-7 only

Create:
`reports/00_environment_and_substrate.md`

---

## 5. Stage 1 — refinement EP shape atlas

For every representative request, layer, and refinement iteration, record/derive:

- M_fresh
- token-expert pairs
- local assignments
- remote assignments
- remote bytes
- destination-rank fanout
- active experts
- rows/expert
- max rows/expert
- rank-load CV
- critical-rank load

Primary tasks:
- GSM8K
- HumanEval

Report:
- p10/p25/p50/p75/p90/p95/max
- early/middle/late refinement
- per-task distribution

Central question:

**What fraction of request time is spent in large, medium, and small compacted EP shapes?**

Create:
- `REFINEMENT_EP_SHAPES.csv`
- `reports/01_refinement_shape_atlas.md`

---

## 6. Stage 2 — replay dataset

Create a replay corpus from real routing traces.

Each case should encode enough information to reconstruct:

- M
- hidden=4096
- top-k=8
- top-k expert indices
- expert weights when needed
- source rank
- destination expert/rank
- per-expert counts

Also generate controlled variants:
- uniform load
- mild skew
- strong skew
- matched M with different fanout
- matched M with different remote fraction

Create:
- `EP_SHAPE_REPLAY.jsonl`
- `EP_SHAPE_SUMMARY.csv`

---

## 7. Stage 3 — existing kernel envelope

Benchmark all valid current paths on replay shapes.

Required:
- DeepEP normal dispatch/combine

If valid on the actual environment:
- DeepEP low-latency
- DeepEP V2/hybrid/direct variants

Do not force unsupported BF16/intranode configurations.

For each shape measure:
- layout/preparation
- dispatch
- existing expert GEMM
- combine
- whole routed-MoE

Use:
- CUDA events for broad sweeps
- Nsight Systems for selected representative cases
- Nsight Compute only for selected kernels

Report:
- median
- p25/p75
- CV
- effective bandwidth
- launch gaps
- SM use
- memory bandwidth

Create:
- `EXISTING_KERNEL_BENCH.csv`
- `reports/03_existing_kernel_envelope.md`

---

## 8. Stage 4 — runtime tax atlas

For representative shape bins:

- very small
- small
- medium-small
- medium
- large

decompose:

- useful payload transfer
- layout/count preparation
- buffer reset/setup
- launch overhead
- dispatch synchronization
- expert launch
- combine launch
- combine synchronization
- host/framework gaps
- idle/bubbles
- staging copies

The point is to distinguish:

**unavoidable physical work**
from
**generic runtime/protocol overhead that a fixed-contract third path could remove**

Create:
- `RUNTIME_TAX.csv`
- `reports/04_runtime_tax_atlas.md`

---

## 9. Stage 5 — kernel headroom oracle

A kernel headroom oracle estimates the maximum credible request-level benefit of a specialized kernel before implementing it.

### O0 — best-existing-path oracle

For each replay shape:

`T_O0 = min(valid existing path latency)`

Map the per-shape winner back to request BCT.

This measures the ceiling of simple dynamic mode switching.

### O1 — removable-control-cost oracle

Set profiling-supported removable overhead to zero:

- avoidable layout/setup
- repeated buffer lifecycle
- avoidable launch gaps
- avoidable staging
- avoidable synchronization

Keep:
- required bytes
- expert computation
- irreducible synchronization

Map to E2E.

### O2 — physical lower bound

Estimate conservatively from measured hardware envelopes:

`required transfer bytes / measured sustainable NVLink bandwidth`
plus
`measured expert compute lower envelope`
plus
`irreducible control`

Do not use theoretical peak bandwidth alone.

### O3 — credible RefineEP target

Choose a target above the physical lower bound, with realistic implementation margin.

Report:

`Task | baseline | O0 dynamic-existing | O1 control-removal | O2 physical LB | O3 credible RefineEP`

Create:
- `HEADROOM_ORACLE.csv`
- `reports/05_kernel_headroom_oracle.md`

---

## 10. Promotion gate before CUDA implementation

Use request-level credible O3 headroom:

- <5%: NO-KERNEL-HEADROOM
- 5-8%: characterization only
- 8-12%: HOLD
- 12-20%: implement real CUDA prototype
- >20%: very strong opportunity

Also require:
- signal on both GSM8K and HumanEval
- target shapes have meaningful frequency/mass
- not caused by one pathological configuration

Do not implement a complex kernel if the credible oracle fails.

---

## 11. RefineEP v0 contract

If gate passes, freeze the first CUDA prototype to:

- H100 only
- single node only
- NVLink/NVSwitch only
- EP4 only
- hidden=4096 only
- top-k=8 only
- experts=256
- experts/rank=64
- BF16 only
- forward only
- observed target M range only

No:
- multi-node
- RDMA
- backward
- arbitrary hidden
- arbitrary top-k
- arbitrary EP
- A100
- generic library API

The goal is research clarity, not production generality.

---

## 12. RefineEP design principle

The novelty must be the new path itself, not the selector.

Start by specializing dispatch/combine while keeping the existing expert GEMM.

Potential mechanisms should be chosen from profiling, not intuition.

Candidate mechanisms:

### P1. Fixed-capacity persistent buffers

Allocate once for the maximum supported target shape.

Avoid generic dynamic buffer lifecycle.

### P2. Fixed-contract layout/count specialization

Exploit:
- EP4
- top-k8
- 64 experts/rank
- hidden4096

Reduce generic metadata/index/layout overhead.

Fuse count/offset/packing preparation only if profiling supports it.

### P3. NVLink-only protocol

Remove multi-node/RDMA-domain generality.

Use only valid peer-access / NVLink-domain mechanisms for the node.

### P4. Medium/small-M mapping

Tune blocks/warps/data movement for the observed refinement shape distribution rather than huge prefill throughput.

### P5. Dispatch/combine first

Do not rewrite grouped expert GEMM initially.

### P6. Optional persistent refinement service

Only if launch/control overhead remains large after P1-P5.

A long-lived service kernel may reuse queue/buffer/control state across refinement invocations.

Do not force this approach if it causes SM contention.

---

## 13. CUDA learning / implementation order

This project assumes the researcher is new to CUDA.

Do not start from PTX or aggressive optimization.

### Step A
Understand DeepEP data contract:
- hidden input
- top-k indices
- token packing
- expert ownership
- receive layout
- combine reconstruction

### Step B
Implement the simplest correct intranode prototype for the frozen contract.

Speed does not matter yet.

### Step C
Match baseline output exactly/numerically.

### Step D
Profile and remove unnecessary copies / poor coalescing / serialization.

### Step E
Tune:
- threads
- warps
- vectorized access
- shared memory
- synchronization
- occupancy

### Step F
Only if justified:
- TMA
- async copies
- persistent service
- H100-specific instructions

Use existing DeepEP kernels as reference implementations.

---

## 14. Correctness contract

RefineEP must preserve:

- token count
- token identity
- expert destination
- expert weight
- expert ownership
- combine reconstruction

Compare to baseline:
- max abs error
- mean abs error
- relative L2

Small normal floating-point reduction-order drift is acceptable only if explained.

Then validate full trajectories:
- GSM8K
- HumanEval
- final sequence
- NFE
- task score

No semantic approximation is allowed.

Create:
`reports/07_refineep_correctness.md`

---

## 15. RefineEP microbenchmark

Benchmark RefineEP against every valid baseline on the real replay corpus.

Report by shape regime:
- very small
- small
- medium-small
- medium
- large

Metrics:
- dispatch
- combine
- full routed-MoE
- effective bandwidth
- CV
- SM use
- memory use

Desired result:

- large M may stay on normal DeepEP
- RefineEP wins materially on high-mass medium/small refinement shapes

Create:
- `REFINEEP_MICROBENCH.csv`
- `reports/08_refineep_microbench.md`

---

## 16. Runtime selector

Only after the third path is real.

Use the simplest selector possible.

Candidate inputs:
- M
- remote assignments
- remote bytes
- fanout
- rows/expert
- max expert rows

Start with a lookup table or thresholds from held-out microbench data.

Concept:

`if shape is in RefineEP winning region -> RefineEP`
`else -> normal DeepEP`

The selector is not the novelty.

Measure selector overhead.

---

## 17. End-to-end integration

Integrate the exact RefineEP path into the true LLaDA2 EP4 runtime.

Important:
the physical runtime must expose the compacted fresh worklist or a faithful equivalent.

Measure clean:
- BCT
- throughput
- NFE
- score
- peak HBM
- per-layer EP time
- percentage of invocations using RefineEP

Use multiple independent restarts.

Create:
- `E2E_RESULTS.csv`
- `reports/10_end_to_end_integration.md`

---

## 18. Required ablations

A1. Normal DeepEP only

A2. Best existing dynamic path selection only

A3. RefineEP forced on all supported shapes

A4. Normal + RefineEP selector

A5. Fixed persistent-buffer ablation if implemented

A6. Specialized/fused-layout ablation if implemented

A7. Persistent-service ablation if implemented

The final paper claim must identify which new kernel mechanism creates the gain.

---

## 19. Optional generality only after EP4 success

### EP8
Support 8×H100 / EP8.

Rebuild the shape atlas and thresholds.

### Second model
Support one additional block-diffusion MoE model only if it exhibits a comparable changing fresh-work regime.

Do not expand scope before the core result is strong.

---

## 20. Prior-art / novelty boundary

Compare with:
- DeepEP normal / low-latency / current V2 paths
- automatic normal-vs-low-latency selection
- Epoch/FreshLane
- UniEP
- generic fused/persistent MoE kernels such as FlashMoE/mKernel/StreamEP when relevant

Not novel:
- choosing an existing DeepEP mode dynamically
- changing thresholds
- using a generic small-M MoE kernel unchanged

Desired novelty:

**A fixed-contract intranode EP third path designed around the repeatedly changing compacted fresh-work shapes of block-diffusion MoE refinement.**

Strong framing:

`Epoch-like execution determines WHAT fresh work remains.`
`RefineEP determines HOW that changing fresh work should be physically executed.`

---

## 21. Failure recovery

### No changing physical shape
If compaction does not expose meaningful M variation, stop.

### Existing DeepEP already near physical lower bound
If credible E2E headroom <5%, stop.

### Dynamic existing kernels capture nearly all oracle
If O0 approximately equals O3, a third path is unjustified.

### Kernel wins only rare shapes
Frequency-weight the result; do not promote microbenchmark-only wins.

### Correctness difficulty
Reduce supported M range and implement dispatch-only first, then combine.

### Expert GEMM becomes dominant
Benchmark existing grouped GEMM first. Do not automatically rewrite it.

### Persistent service causes contention
Remove it if saved launch/control time is smaller than interference cost.

---

## 22. Required figures

1. fresh M vs refinement iteration
2. fresh remote assignments vs refinement
3. refinement shape histogram
4. existing path latency vs M
5. existing path latency vs bytes
6. normal/LL/V2 crossover map
7. Nsight large-shape timeline
8. Nsight medium-shape timeline
9. Nsight small-shape timeline
10. runtime tax breakdown
11. measured path vs physical lower bound
12. request-level kernel headroom oracle
13. RefineEP vs baseline latency by M
14. RefineEP vs baseline by skew/fanout
15. end-to-end BCT
16. kernel-mechanism ablations
17. fraction of invocations using RefineEP

---

## 23. Required reports

- reports/00_environment_and_substrate.md
- reports/01_refinement_shape_atlas.md
- reports/02_ep_replay_dataset.md
- reports/03_existing_kernel_envelope.md
- reports/04_runtime_tax_atlas.md
- reports/05_kernel_headroom_oracle.md
- reports/06_refineep_design.md
- reports/07_refineep_correctness.md
- reports/08_refineep_microbench.md
- reports/09_switching_policy.md
- reports/10_end_to_end_integration.md
- reports/11_ablations.md
- reports/12_prior_art_and_novelty.md
- reports/final_decision.md

Machine-readable:
- REFINEMENT_EP_SHAPES.csv
- EP_SHAPE_REPLAY.jsonl
- EXISTING_KERNEL_BENCH.csv
- RUNTIME_TAX.csv
- HEADROOM_ORACLE.csv
- REFINEEP_MICROBENCH.csv
- E2E_RESULTS.csv
- ATTEMPT_LOG.csv

---

## 24. Final labels

- NO-KERNEL-HEADROOM
- CHARACTERIZATION-SIGNAL
- HOLD-KERNEL
- STRONG-KERNEL-CANDIDATE
- VERY-STRONG-KERNEL-CANDIDATE

Definitions:

NO-KERNEL-HEADROOM:
credible request-level specialized-kernel headroom <5%.

CHARACTERIZATION-SIGNAL:
shape transition exists but useful headroom <8%.

HOLD-KERNEL:
credible headroom 8-12% or incomplete positive prototype.

STRONG-KERNEL-CANDIDATE:
credible oracle >=12% and RefineEP materially beats DeepEP on high-mass target shapes.

VERY-STRONG-KERNEL-CANDIDATE:
exact integrated RefineEP improves real request BCT/throughput >=10-15% over the strongest existing-path baseline and reproduces across tasks/restarts.

---

## 25. Final questions

1. What fresh-M distribution appears after liveness compaction?
2. Does it span genuinely different physical EP regimes?
3. Which existing DeepEP path wins each regime?
4. Is there a medium/small regime where all existing paths are inefficient?
5. How much request E2E mass lies there?
6. How much EP time is useful transfer vs generic runtime tax?
7. What is the best-existing dynamic oracle?
8. What is the credible specialized-kernel oracle?
9. Does the credible oracle exceed 12%?
10. Which physical overhead should RefineEP remove?
11. Can a fixed-contract CUDA kernel reproduce dispatch/combine correctly?
12. Does RefineEP beat normal DeepEP on high-mass target shapes?
13. Does it beat every valid existing low-latency/V2 alternative?
14. Is output exact enough for full trajectory equivalence?
15. Does the microbenchmark gain survive LLaDA2 integration?
16. Is speedup due to the new path rather than switching?
17. Which kernel mechanisms matter?
18. Is persistent block-lifetime execution necessary?
19. Does EP8 preserve the phenomenon if tested?
20. Is the method distinct enough from generic persistent/fused MoE kernels?

---

## 26. Final instruction

Do not begin with a complicated CUDA kernel.

First prove:

`refinement -> high-mass compacted EP shape regime`
`+ existing kernels leave substantial overhead`
`+ credible third-path target gives >=12% request-level E2E headroom`

Only then implement RefineEP.

The intended contribution is not:

`we switch DeepEP modes dynamically`

but:

`dLLM refinement exposes a distinct changing fresh-work EP regime, and a fixed-contract intranode third path removes generic dispatch/combine overhead for that regime while preserving exact model semantics.`
