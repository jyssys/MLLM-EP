# Four-rank DeepEP replay

## Protocol

- Qwen3-VL-30B-A3B-Instruct BF16 captured hidden states, routes, router
  weights, and real expert weights.
- Physical GPUs 4,5,6,7 only; four NCCL ranks; DeepEP high-throughput;
  `num_sms=20`.
- Primary M=8177 is exactly 37 copies of a 221-token route; selector/layer
  controls use M=8192.  Each uses one large dispatch/expert/combine invocation,
  5 policy warmups, and 30 randomized/interleaved repetitions.
- Rank-critical same-device CUDA event time is reduced across ranks.
- Three images at layer 24 are the primary robustness set; camera layers
  4/24/44 provide the layer control.
- Correctness numbers measure the intended approximate output against stock;
  they are not required to be numerically exact.

The local vLLM installation has no tuned `E=32,N=768,H100` FusedMoE config and
therefore emits the default-config warning.  Every policy shares that path,
so only paired relative effects are used.

## Primary three-image median

| Policy | Vision drop | Max-rank reduction | MoE reduction | Projected TTFT reduction |
|---|---:|---:|---:|---:|
| Global semantic 10% | 11.48% | 9.98% | 6.89% | 4.28% |
| + EP refinement | 11.48% | 14.92% | 9.34% | 5.81% |
| Global semantic 20% | 19.90% | 16.35% | 12.21% | 7.60% |
| + EP refinement | 19.90% | 26.38% | 15.94% | 9.92% |
| Global semantic 30% | 29.46% | 24.44% | 16.96% | 10.55% |
| + EP refinement | 29.46% | 32.26% | **21.13%** | **13.14%** |

These primary rows use M=8177, exactly 37 copies of the 221-token captured
route, so semantic and refined policies have identical assignment counts.
The TTFT projection multiplies paired MoE reduction by the prior clean 16K
critical-MoE share, 62.191%.  It is intentionally labeled an Amdahl
projection, not observed request E2E.

## Selector and trivial controls at 30% (M=8192)

| Policy | Max-rank reduction | MoE reduction | Projected TTFT |
|---|---:|---:|---:|
| Fixed visual K=6 | 21.26% | 9.79% | 6.09% |
| Router mass 0.8 | 18.26% | 11.38% | 7.08% |
| Global semantic full | 24.43% | 17.06% | 10.61% |
| Global semantic + EP | 32.25% | 21.09% | 13.12% |
| Request-route router risk | 23.09% | 14.50% | 9.02% |
| Request-route router risk + EP | 36.62% | 20.43% | 12.71% |
| Actual-contribution risk | 24.89% | 15.27% | 9.50% |
| Actual-contribution risk + EP | 33.55% | 19.74% | 12.28% |

The contribution oracle does not dominate router risk.  More importantly,
the total 12–13% projection combines the already-known semantic skipping gain
with only 2.5–3.7 percentage points from the EP-specific refinement.

## Evidence boundary

No production ragged-K vLLM path was implemented.  The replay encodes omitted
suffix experts as invalid IDs and measures actual DeepEP/FusedMoE work in one
large invocation.  Selector construction, route-load reduction, and mask
construction overhead are absent, making the projected result optimistic.
Because the EP-incremental opportunity is already small and collides with
prior art, adding those costs cannot rescue the novelty gate.
