# Active frontier (live queue)

The queue is intentionally kept above eight candidates. `PENDING` means a
cheap causal probe is still possible; it is not evidence of a positive result.

| ID | Hypothesis | Evidence so far | Expected signal | Required control | Runtime | Novelty risk | Priority | Status |
|---|---|---|---|---|---:|---|---:|---|
| C1a | fixed text concurrency changes EP phase geometry nonlinearly | text c8→c16 T_MoE p50 +19.8% | >5% at same prompt shape | c2/c8/c16 fixed text, randomized order | 20m | high (batching literature) | 1 | PENDING |
| K1a | request queue/CPU delay, not MoE, explains E2E spread | c2/c8/c16 E2E p50 differs 2–3× | >15% request mass outside MoE | arrival/queue timestamps | 25m | medium | 2 | PENDING |
| D2a | phase composition changes exposed dispatch vs expert overlap | dispatch share 15–33%, wait 3–15% | >=5% phase critical span | fixed M with prefill/decode mix | 25m | medium | 3 | PENDING |
| G1a | visual workload changes expert kernel regime at matched M | vision expert share 42.2% vs text 31.0% | >=5% expert time at matched M | same M/layer and token volume | 25m | medium | 4 | PENDING |
| H1a | layer groups have stable distinct runtime regimes | layer IDs available, not yet mapped | repeated layer cluster >=5% | early/mid/late matched M | 20m | low/medium | 5 | PENDING |
| J1a | multimodal cache state changes MoE critical path | cache-hit rate reported by driver | repeated cache-state effect | cold vs warm image cache | 30m | medium | 6 | PENDING |
| L1a | residual tail cluster tracks stage/stream shape | Model M/load/fanout R²≈0 | residual cluster >=10% mass | stage-level residual clustering | 20m | high (closed-tail overlap) | 7 | PENDING |
| B1a | prefill/decode overlap window is workload-dependent | mixed c8 has both phases | >=5% E2E oracle | same arrival schedule, phase matched | 30m | high | 8 | PENDING |
| D1a | non-MoE attention timing co-varies with expert stage | not instrumented in atlas | separate MoE vs generic runtime | attention range hook | 30m | medium | 9 | PENDING |
| F1a | sender timing alignment, not fanout, predicts dispatch | fanout null | incremental error >=5% | preserve M/load, permute sender timing | 30m | high | 10 | PENDING |
| C1c | vision↔text boundary leaves a persistent DeepEP shape/workspace state | text→high-res vision 2.395ms vs matched warmup 1.204ms; text-only alternation null | >=10% request mass after clock/cache controls | same image shape, matched warmup, telemetry | 25m | medium/high (shape-aware batching) | 1 | CLOSED/TRIVIAL |

Queue update policy: each completed experiment must close, promote, or spawn at
least one child, and every promoted node receives a direct E2E headroom check.
