# TEAM positive control under true EP2

## Result

**Gate B: PASS.** TEAM's benefit survives semantics-preserving true EP2.

| Restart | Warm-up | Baseline | TEAM | Speedup |
|---:|---:|---:|---:|---:|
| 2 | no | 14.546 s | 11.284 s | 1.289x |
| 3 | no | 14.686 s | 11.589 s | 1.267x |
| 4 | yes | 12.022 s | 7.751 s | 1.551x |
| **Median** | — | — | — | **1.289x** |

The cold smoke pair (2.09x) is excluded from the restart median.  The warm
paired point is the clean headline for the reference path.  The spread shows
why this PoC should not be read as a production throughput benchmark.

## Work and communication

In matched observer-heavy reference traces, baseline made 1,152 MoE calls and
TEAM 768.  TEAM reduced call count by one third and active local-expert events
from 32,577 to 18,031, but increased token-expert assignments from 307,200 to
403,968 in that stochastic run.  Remote hidden payload similarly increased:
the algorithm traded fewer launches/round trips for larger speculative
messages.

The key result is not “TEAM always sends fewer bytes.” It is that TEAM's total
decoder policy still yields lower request latency under real expert ownership,
remote dispatch, local execution, and combine.

## Production-like trivial-fix control

Replacing the Python per-expert loop with vLLM's existing fused expert operator
changed the clean pair to:

| Backend | Baseline | TEAM | Speedup |
|---|---:|---:|---:|
| Python reference | 12.022 s | 7.751 s | 1.551x |
| vLLM fused local expert | 3.936 s | 3.338 s | 1.179x |

TEAM remains positive, but much of the apparent reference-path headroom was
ordinary expert-kernel fragmentation.  The 56.9% TEAM latency reduction from
using the already-available fused primitive is a successful adversarial
trivial-fix attack, not a new research result.

## Correctness qualification

For the same layer-0 FP16 hidden tensor and router, the Python-reference EP2
output reached cosine 0.99999994 and relative L2 0.0505% against the original
full-expert implementation; fused EP2 reached cosine 1.0 and relative L2
0.0553%. Router logits matched exactly in both. Long diffusion strings can
diverge under this small numerical reordering, so the fused control is used to
bound systems cost, not to claim token-identical full-benchmark output.

## Figures

- `plots/05_stage_b_speedup_substrate.png`
- `plots/07_ep2_work.png`
- `plots/10_ep2_communication.png`
