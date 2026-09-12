# Fair vanilla Single / EP2 / EP4 characterization

## Measurement contract

- Model/checkpoint: SDAR-30B-A3B-Chat-b32, FP16.
- Harness: released vanilla block-diffusion generator; 4 GSM8K and 4
  HumanEval prompts; generation length 128; block/denoising length 32;
  threshold 1.0; greedy target rule; one warm-up request.
- Fixed-work control: EOS detection is disabled during compute so the complete
  output budget executes on every topology. Scoring may ignore special-token
  padding, but timing does not.
- Topology is the only execution variable: TP1/DP1 with EP1, EP2, or EP4.
  EP1 bypasses vacuous self-collectives; EP2/4 use true disjoint ownership.
- Clean comparisons use three independent engine/model restarts in balanced
  order. Tracing is separate and observer tax is reported.

## Correctness boundary

Direct one-layer checks against the released expert implementation achieved
cosine at least 0.99999988 and relative L2 at most 0.0545%, with identical
router logits. Fused group shape changes the floating-point accumulation path,
so full generated strings need not be bitwise identical. The performance
comparison therefore requires equal model-forward counts and reports bounded
task quality for every topology; a run with different work is excluded.

## Results

Three-restart results use the median restart as the statistical unit. The
bounded quality column combines the four GSM8K and four HumanEval examples; it
is a fidelity guard, not a publishable accuracy estimate.

| Topology | Bounded quality | Clean 8-request wall | Mean request | vs Single wall | Model forwards | Assignments | Remote assignments | Remote hidden bytes (dispatch + combine) | Median fanout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single | 62.5% | 161.511 s | 20.091 s | 1.000x | 1,166 | 14,487,552 | 0 | 0 | 1 |
| EP2 | 62.5% | 184.931 s | 23.230 s | 1.145x | 1,166 | 14,487,552 | 7,236,667 | 59.283 GB | 2 |
| EP4 | 62.5% | 192.096 s | 24.437 s | 1.189x | 1,166 | 14,487,552 | 10,856,501 | 88.936 GB | 4 |

All nine clean runs executed exactly the same 1,166 model forwards and
14,487,552 token-expert assignments. The median remote fractions were 49.95%
for EP2 and 74.94% for EP4, as expected for contiguous, balanced ownership.
EP2 was 14.5% slower and EP4 18.9% slower than Single on clean aggregate wall;
EP4 was 3.9% slower than EP2. This establishes a substantial distributed
runtime tax in this reference backend, but does **not** establish that a new
method can remove it.

Generated strings varied across restarts because the fused group shape and GPU
execution are not bitwise deterministic. The important paired controls held:
router logits were exact in the layer check, expert outputs met the numerical
gate, every topology performed identical logical work, and median bounded
quality was identical. Per-restart overall scores were Single
`[62.5, 62.5, 50.0]%`, EP2 `[62.5, 62.5, 50.0]%`, and EP4
`[50.0, 62.5, 62.5]%`.

## Interpretation

The reference EP transport deliberately prioritizes exactness and observability
over production optimization: four outbound A2As carry hidden rows and integer
metadata, one reverse A2A returns weighted branches, and a combined-tensor
broadcast realigns replicated dense state. Communication-only opportunities are
therefore stress-tested under faster-runtime sensitivity before promotion.
The slowdown itself is characterization rather than headroom: simply reverting
to one GPU is not a multi-GPU EP method and sacrifices the memory/capacity
contract that motivates EP.

## Stage localization

A separate two-request run added same-device CUDA events. Tensor-value capture
was disabled, but event/module hooks still cost 12.6% (EP1), 13.1% (EP2), and
67.3% (EP4), so these rows localize cost on their own instrumented timeline and
are not substituted for clean wall time. Critical-rank maxima for individual
sub-stages are not additive because the critical rank can differ by stage.

| Topology | Attention share | Router | Prepare | Dispatch | Expert | Combine | Whole MoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP1 | 32.09% | 9.61% | 10.25% | 0.36% | 15.54% | 2.76% | 40.31% |
| EP2 | 28.45% | 8.39% | 22.15% | 5.22% | 12.73% | 9.76% | 49.53% |
| EP4 | 29.75% | 8.07% | 38.82% | 5.94% | 13.27% | 16.57% | 66.90% |

The reference path's EP-degree penalty first appears in route preparation and
communication, not in reduced local expert compute. EP4 does not make expert
execution faster enough to amortize the additional collectives and host-visible
split preparation at the 32-position refinement shapes. Because this is a
reference NCCL transport with separate metadata collectives and dense-state
broadcast, every communication-derived oracle is also reported at 0.75×,
0.5×, and 0.25× runtime-cost sensitivity.
