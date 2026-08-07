# FlashVEP DBO numerical divergence localization

## Final status

**FINAL STATUS: GO**

At the identical autoregressive prefix `2132,4977,1075`, layer 0 is bit-exact
through its input, attention output/residual, MoE dispatch input, MoE output,
and final residual. The first nonzero difference is consistently the **layer 1
attention output**. It is small (`max=4.8828125e-4`, cosine `0.99999404`) and
then accumulates gradually through later layers. This matches numerical
execution-order sensitivity, not abrupt state corruption.

The source-level Attention metadata and DeepStack correctness issues remain
closed. Bit-exact DBO-off/on generation parity is not claimed.

## Method

- Qwen3-VL-30B-A3B-Instruct, BF16, TP2/DP2/EP4/PP1
- DeepEP high-throughput, vLLM 0.20, enforce eager
- Existing Attention and DeepStack source fixes retained
- Only physical GPUs 4,5,6,7 exposed
- Original per-rank `[blue, red]` two-request wave
- Compared red-request decode state immediately before generated token 4,
  where both modes have prefix `2132,4977,1075`
- Captured FP32 copies of layer input, attention output, attention residual,
  dispatch input, MoE output, and layer-final residual
- Two independent DBO-off executions and three independent DBO-on executions;
  each engine invocation also generated the prompt three times

## First divergence

- First numerically divergent layer: **1**
- First divergent operator/stage: **attention output**
- DP0 and DP1, across all three independent DBO-on executions: identical
  location and magnitude
- Layer 1 attention-output metrics:
  - max absolute: `0.00048828125`
  - mean absolute: `0.00002700370`
  - RMSE: `0.00004833979`
  - relative L2: `0.00348207`
  - cosine similarity: `0.99999404`
  - norm off/on: `0.62824974 / 0.62796359`

Layer 0 is exactly equal at all six capture points. Layer 1 `dispatch_input`
therefore receives an already-different tensor after attention. Decode-stage
DeepEP dispatch, expert GEMM, and combine cannot be the first source. The most
specific supported localization is a tiny layer-1 attention/KV difference;
the capture does not distinguish whether its seed was written into layer-1 KV
cache during the earlier DBO prefill or arises in the decode attention kernel.

## Layer-by-layer final-output error

Representative DP0 comparison is shown below. Full metrics for every stage,
both ranks, and repeats are in `summary.json`.

| Layer | Max abs | Mean abs | RMSE | Rel L2 | Cosine | Norm off | Norm on |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 1.00000000 | 1.3758 | 1.3758 |
| 1 | .001953 | 6.74e-5 | .000126 | .00280 | .99999614 | 2.0399 | 2.0406 |
| 2 | .001953 | .000130 | .000205 | .00401 | .99999197 | 2.3119 | 2.3121 |
| 3 | .015625 | .000307 | .000554 | .00780 | .99996989 | 3.2153 | 3.2126 |
| 4 | .019531 | .000420 | .000715 | .00882 | .99996116 | 3.6673 | 3.6661 |
| 5 | .023438 | .000602 | .000950 | .01086 | .99994108 | 3.9591 | 3.9585 |
| 6 | .039062 | .000874 | .001409 | .01373 | .99991210 | 4.6424 | 4.6254 |
| 7 | .031250 | .000967 | .001432 | .01301 | .99991717 | 4.9842 | 4.9745 |
| 8 | .039062 | .001081 | .001677 | .01383 | .99990549 | 5.4856 | 5.4767 |
| 9 | .023438 | .001344 | .001833 | .01422 | .99989895 | 5.8308 | 5.8273 |
| 10 | .023438 | .003597 | .004551 | .03457 | .99940241 | 5.9578 | 5.9558 |
| 11 | .016602 | .003792 | .004769 | .03550 | .99937284 | 6.0798 | 6.0910 |
| 12 | .019531 | .003819 | .004837 | .03493 | .99939073 | 6.2668 | 6.2722 |
| 13 | .046875 | .006605 | .008356 | .05800 | .99831719 | 6.5197 | 6.5151 |
| 14 | .078125 | .006880 | .008843 | .05327 | .99861190 | 7.5126 | 7.5619 |
| 15 | .062500 | .006824 | .008821 | .05082 | .99873937 | 7.8547 | 7.9073 |
| 16 | .062500 | .006807 | .008769 | .04909 | .99883081 | 8.0830 | 8.1427 |
| 17 | .062500 | .006675 | .008620 | .04848 | .99885602 | 8.0471 | 8.1020 |
| 18 | .093750 | .006386 | .008446 | .04770 | .99887866 | 8.0125 | 8.0504 |
| 19 | .093750 | .006088 | .008133 | .04606 | .99895178 | 7.9909 | 8.0235 |
| 20 | .062500 | .006378 | .008328 | .04156 | .99914287 | 9.0675 | 9.0937 |
| 21 | .093750 | .006309 | .008405 | .03451 | .99940687 | 11.0209 | 10.9890 |
| 22 | .062500 | .006321 | .008343 | .03424 | .99941382 | 11.0281 | 11.0265 |
| 23 | .039062 | .006372 | .008344 | .03497 | .99939701 | 10.7971 | 10.8357 |
| 24 | .042969 | .006305 | .008339 | .03364 | .99944871 | 11.2186 | 11.2730 |
| 25 | .050781 | .006382 | .008406 | .03156 | .99951167 | 12.0525 | 12.1004 |
| 26 | .046875 | .006638 | .008602 | .03091 | .99953294 | 12.5926 | 12.6455 |
| 27 | .046875 | .006774 | .008715 | .02828 | .99960080 | 13.9446 | 13.9573 |
| 28 | .062500 | .006724 | .008804 | .02708 | .99963386 | 14.7113 | 14.6890 |
| 29 | .062500 | .006678 | .008737 | .02616 | .99965802 | 15.1146 | 15.0994 |
| 30 | .125000 | .007476 | .010037 | .02872 | .99958788 | 15.8145 | 15.7937 |
| 31 | .125000 | .007398 | .010027 | .02648 | .99965012 | 17.1352 | 17.1073 |
| 32 | .125000 | .007755 | .010684 | .02556 | .99967531 | 18.9153 | 18.8705 |
| 33 | .078125 | .013096 | .016575 | .03720 | .99930827 | 20.1651 | 20.1346 |
| 34 | .078125 | .013514 | .017218 | .03756 | .99929464 | 20.7473 | 20.7440 |
| 35 | .078125 | .013761 | .017576 | .03790 | .99928148 | 20.9844 | 20.9602 |
| 36 | .093750 | .013974 | .017930 | .03693 | .99931804 | 21.9741 | 21.9541 |
| 37 | .109375 | .014633 | .018851 | .03518 | .99938382 | 24.2465 | 24.2904 |
| 38 | .109375 | .015170 | .019575 | .03363 | .99943551 | 26.3396 | 26.3661 |
| 39 | .125000 | .016338 | .021018 | .03465 | .99940803 | 27.4483 | 27.5462 |
| 40 | .125000 | .018248 | .023053 | .03438 | .99941372 | 30.3435 | 30.4211 |
| 41 | .156250 | .020359 | .025877 | .03316 | .99945829 | 35.3171 | 35.4406 |
| 42 | .156250 | .023493 | .029561 | .03169 | .99950412 | 42.2133 | 42.3432 |
| 43 | .250000 | .025441 | .032418 | .03205 | .99949580 | 45.7769 | 45.9532 |
| 44 | .187500 | .029913 | .037590 | .03146 | .99950587 | 54.0760 | 54.1176 |
| 45 | .187500 | .032941 | .041520 | .03240 | .99947626 | 57.9901 | 58.0539 |
| 46 | .250000 | .042811 | .053886 | .03390 | .99942897 | 71.9419 | 72.0947 |
| 47 | 1.250000 | .046335 | .066044 | .03116 | .99952370 | 95.9049 | 95.4430 |

The absolute maximum grows with activation norm, but relative L2 remains about
3% in late layers and cosine stays above 0.998. There is no isolated structural
break or non-finite value.

## Router and D/E/C interpretation

Router top-k IDs/weights were not captured because this vLLM path performs the
router inside the fused `FusedMoE` kernel; recomputing it would have changed the
execution being diagnosed. This is a limitation, not evidence of mismatch.

At the earliest layer:

- layer 0 dispatch input, expert/combine output, and layer final: exact;
- layer 1 attention output: first small difference;
- layer 1 dispatch input: already different before dispatch;
- layer 1 MoE output: small downstream difference.

Therefore dispatch/expert/combine do not initiate the observed decode-stage
difference. Their internal boundaries cannot be separated by the safe module
hooks used here.

## Repeat stability and corruption audit

All three independent DBO-on executions and both DP ranks localize first at
layer 1 attention output with exactly the same metrics. In this instrumented
run, all DBO-on generated sequences were `2132,4977,1075,697`; DBO-off remained
`2132,4977,1075,498`. No Attention batch metadata mismatch, DeepStack payload,
shape error, NaN, or abrupt tensor corruption appeared.

## Final judgment

**GO: source-level correctness issues are closed; remaining divergence is
localized to numerical execution-order sensitivity.** The first observable
seed is a tiny layer-1 attention/KV difference, followed by gradual layer-wise
amplification. It is not first introduced by decode-stage DeepEP dispatch,
expert compute, or combine.

## Limitations

- The hook observes decode modules, not the earlier prefill operation that
  populated layer-1 KV cache; it cannot separate prefill-produced KV rounding
  from decode attention rounding.
- Fused router IDs and internal dispatch/expert/combine tensors were not exposed.
- Full `.npy` captures remain local and are intentionally not committed; the
  repository contains compact JSON metrics and raw generation/logit artifacts.
- The result is scoped to this prompt, configuration, and BF16 environment.

## One recommended next action

Add a bounded layer-1 KV-cache checksum immediately after prefill for the red
request, comparing DBO-off with each DBO ubatch, to separate prefill KV rounding
from decode attention rounding.

Raw summary:
`poc_flashvep/deepep_revalidation/results/dbo_numerical_localization_20260807_211800/summary.json`
