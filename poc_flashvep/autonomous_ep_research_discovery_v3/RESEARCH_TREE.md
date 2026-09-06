# Research tree — autonomous EP discovery v3

The tree is headroom-first. A node is promoted only when a fresh live trace
shows a plausible direct request-level opportunity; a micro-level anomaly is
not a paper direction by itself.

## Root

`MLLM/LLM online MoE-EP execution waste` — breadth probes on Qwen3-VL,
TP2/DP2/EP4, DeepEP high-throughput, physical GPUs 1–4.

## Families and nodes

| Node | Family | Hypothesis / control | Fresh evidence | Headroom gate | Status | Children |
|---|---|---|---|---|---|---|
| A1 | dependency | outstanding event/queue state creates removable critical-path debt; compare fixed shapes and stage waits | dispatch outliers remain in all fresh runs; native event handle unavailable in hook | direct E2E not joined; old direct cap 1.09% | CLOSED (prior E2E kill) | A1a readiness instrumentation, not pursued now |
| B1 | overlap | phase composition changes hidden communication slack | stage shares vary with concurrency; no causal intervention | <15% direct oracle established | CLOSED / LOW_HEADROOM | B1a not promoted |
| C1 | continuous batching | same request shape at different concurrency changes MoE phase geometry | text c8→c16 T_MoE p50 +19.8%, E2E p50 +17.8%; c2/c4/c8/c16 and repeats collected | potentially high, but load-confounded; obvious throughput trade-off | CLOSED / TRADEOFF | no method |
| C1b | continuous batching | max batched-token budget changes EP phase even at same M bins | mixed c8 MBT8192→4096: M=114 T_MoE 1.155→2.133ms, E2E p50 901→1278ms | direct effect large, but one obvious knob recovers it | TRIVIAL_ENGINEERING | C1c adaptive token budget only if a nontrivial trade-off appears |
| C1c | continuous batching / MLLM state | crossing the vision encoder/text boundary leaves a persistent DeepEP shape state | one text→vision run: 2.395ms vs matched 1.204ms; text↔vision 2.12ms; vision-only 1.16ms; 32-wave text 1.175ms; telemetry-tagged replication 1.20ms | high single-run signal, but not independent-run robust; matched warmup/telemetry removes it | CLOSED / STATE-CONFOUNDED | no method; only a future DVFS/cache-controlled study |
| C2 | composition | vision-heavy vs text-heavy requests alter MoE stage mix at matched serving load | vision c8 expert share 42.2% vs text c8 31.0%; T_MoE p50 +4.3%; matched warmups normalize | E2E difference includes encoder cost; no direct MoE oracle | CLOSED | no method |
| D1 | interference | attention/vision compute interferes with EP expert/dispatch | no attention timing in hook; stage event wait rises under high c16 | unknown | UNTESTED | D1a non-MoE range timing |
| D2 | interference | expert-heavy vs dispatch-heavy phases trade critical-path resources | expert is 22–42% normal mass; dispatch 15–49%; wait 3–16% across fresh runs | direct request oracle unavailable | CLOSED / DESCRIPTIVE | no method |
| E1 | temporal | history-dependent state is frequent/high-mass beyond fixed-shape tails | dispatch maxima 1.2–2.6s, event waits small in normal rows; prior request-level kill | <10% direct E2E | CLOSED | E1a recheck only if new state field appears |
| F1 | communication structure | fanout/incidence adds load-independent signal | all fresh Model2→fanout changes <=0.001% or negative | no | CLOSED | none |
| G1 | expert/runtime | expert work dominates normal MoE time and changes with image shape | vision expert share 42.2%, text 31.0%; shape-matched warmup controls normalize | no direct request oracle | CLOSED / DESCRIPTIVE | no method |
| H1 | layer heterogeneity | early/middle/late layers have distinct cost regimes | fresh layer p50 ratios only 1.18× prefill / 1.25× decode; low mass and no stable orthogonal feature | <15% direct oracle | CLOSED / LOW_HEADROOM | no method |
| I1 | modality | visual embeddings cause a distinct MoE execution regime | vision vs text stage mix difference, but no routing incremental signal | <15% E2E attribution | HOLD | I1a image-count control |
| J1 | data-centric | prompt/image cache state creates repeated high-mass execution class | transition anomaly is not reproducible; telemetry clocks co-vary | no causal cache state | CLOSED / CONFOUNDED | no method |
| K1 | scheduler | scheduler bubbles dominate request E2E while MoE stays ~1ms | c2/c8/c16 E2E varies 286–970ms while normal T_MoE ~1.2–1.4ms; lower concurrency trades throughput | direct scheduler instrumentation out of scope; no oracle | CLOSED / TRADEOFF | no method |
| L1 | residual | high residual cluster is caused by shape not captured by M/load | 90,240-row residual cluster is high-M/high-active-expert; fanout adds ~0% held-out information | no orthogonal variable | CLOSED | no method |

## Promotion rule

C1/K1 are the only live frontier nodes with potentially material E2E impact;
they must be tested with request-level controls before any method work. All
other nodes remain observational until a 15% direct oracle is demonstrated.
