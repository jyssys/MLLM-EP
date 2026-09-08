# Native author-runtime Qwen-VL bridge validation

This is a **post-vision prefill bridge**, not full online VL serving. Exact real
image encoder embeddings, MRoPE cos/sin and three DeepStack contributions are
captured from the full48-layer HF model. No synthetic embeddings substitute for
images. Native author planner, replication, copy streams and expert kernels stay
unchanged. Physical4–7; native four-rank AllGather runtime is NOT DeepEP.

## Two port bugs separated from method behavior

1. Raw VL checkpoint packs expert matrices as right-multiplication `[E,H,2I]`
   and `[E,I,H]`. Native Linear loaders require transposed separate gate/up/down
   matrices. Real-checkpoint CPU regression failed before, passes after explicit
   split/transpose; FP32 one-expert arithmetic max error6.56e-7.
2. Native first-layer residual aliases its embedding buffer and DP scatter writes
   to it. Stock embedding lookup is fresh every invocation. Reusing a captured
   embedding buffer corrupted subsequent forwards. CPU repeated-input regression
   failed before and passes after a fresh clone. No runtime algorithm changed.

The first diagnostic forward had exact layer0 input and attention-output relative
L2≈0.0024–0.0027 to HF. This localized neither missing vision embeddings nor a
first-layer weight-layout error. Repeated pre-fix logits were invalid; all such
timings are excluded, not presented as Libra failing on MLLM.

## First live corrected group

`online/libra_native_vl48_gqa0_lifetimefix`: four distinct real GQA images,
natural equal M287 per source,3warmups/10 randomized paired measured executions.
All native vanilla/Libra greedy first tokens match HF across repeats. Native
vanilla minimum HF logit cosine0.9977569; maximum relativeL2≈0.06735. Libra vs
vanilla minimum cosine0.9973943, maximum relativeL2≈0.07223. This is approximate
BF16 cross-backend numerical agreement, NOT bitwise parity or benchmark accuracy.

Native vanilla prefill median88.29ms; Libra185.48ms. This small-shape scaled result
does not reproduce the paper's large-model8H200 setting and is not itself an
MLLM-specific failure. No full-request E2E speedup is inferred.

## Oracle fidelity limitation and next control

Vanilla-trace future logits improve realized prediction but only93% mean top-k
recall in the first VL group; replicated reduction changes BF16 trajectories.
That diagnostic has effectively0% timing gain, but must NOT be called a100%
perfect-route upper bound. A separate controlled pair fixes actual logical
routes to the SAME recorded trace in both ordinary-prediction and oracle modes.
Only the oracle predictor receives future knowledge; its realized recall must
equal100%. Original predictor GEMM is retained in both. This pair is labeled
causal route replay, not model-correct online serving or a quality benchmark.

Remaining: other seven real groups, a large text control, source-cost decomposition
if prediction-only headroom survives. No successor method or Kimi promotion yet.
# Eight-group completed control — additional parity boundary

All eight source groups (32 real VL requests) now ran10 randomized measured
repetitions with the full48-layer author decoder. Native Libra and its native
vanilla baseline have identical first greedy tokens in320/320 rank/request
observations. Versus the captured HF reference,each matches310/320 observations,
or31/32 distinct requests. The mismatch is `gqa_5474`,present in native vanilla
as well as Libra,not a newly introduced Libra error. HF first token29898 versus
native67; HF puts0.634 probability on29898 and0.141 on67. Maximum HF→native
vanilla KL is about0.389 in this case. **Do not call this exact cross-backend
quality parity or dismiss the gap as numerically negligible.**

The bridge therefore provides approximate bounded functional fidelity,not a
full-generation quality certificate. No measured loss against HF is attributed
to Libra without the same native baseline control. The original first-group
sanity below remains valid for that group but must not be generalized to all32.

In the orthogonal current-route-frozen comparison,both variants execute the same
logical current expert routes,and the oracle's realized top-k prediction recall
is exactly1.0. Across eight group medians,the extra rank-critical prefill gain is
0.1464%,with group-bootstrap95% CI[-0.1019,1.4076]%;group range[-0.3117,1.7389]%.
This measures improving prediction at fixed predictor compute in this supplied
consumer. It is **not** an unrestricted future-runtime or full-request oracle.
