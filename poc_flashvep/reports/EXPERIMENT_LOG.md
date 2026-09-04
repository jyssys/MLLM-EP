# Autonomous discovery experiment log

| Time (KST) | Experiment | New live GPU data | Result | Next questions |
|---|---|---|---|---|
| 2026-09-05 01:26–01:30 | H1 controlled fragmentation, M=512/1024, EP4 | 2 persistent vLLM runs, 5 A values × 5 reps | A16 faster than A2; H1 NO-GO as stated | H1b/H1c/H1d: why reversal/crossover? |
| 2026-09-05 01:36–01:43 | H4/H5 target-B history conditioning, 4 prime conditions, M=256 | 4 persistent vLLM runs, 3 reps per case | H4 <5%; H5 similar-prime noisy +27.8% | H5b/H5c: warmth vs position/allocator state |
| 2026-09-05 01:44+ | Offline H2/H6/H10/H13/H15 mining | preserved real route tables, leave-request-out/residual files | no robust new control-plane feature | H15b/H10b; continue live queue |
| 2026-09-05 02:22–02:25 | H6 fanout geometry, Qwen3-VL layer24, M=128/512/1024, F=1/2/4 | 9 cases × 5 iterations on persistent 4-GPU DeepEP HT | F1→F4 expert −17.1% at M512 but +114.4% at M128; same assignments/rank load | H6b: fanout×M kernel regime; replicate at other layers |
| 2026-09-05 02:25–02:27 | Generic Qwen3-30B-A3B cross-model H1 control, M=128/512 | 10 cases × 5 iterations, same EP4 backend | M512 expert +5.9% (critical wall −0.6%); M128 has small-shape tail | inverse fragmentation is not MLLM-specific; test phase/kernel boundary |
| 2026-09-05 02:27–02:31 | H6 fanout layer robustness, layers4/44, M=128/512 | 12 cases per layer × 5 iterations | M512 F1→F4 expert −16.3% (L4), −12.4% (L44); M128 +113.0%/+104.2% | communication geometry interacts with M across layers |
| 2026-09-05 02:32–02:34 | H1 A32 extension, Qwen3-VL layer24, M=128…8192 | 30 cases × 5 iterations, active 2/4/8/16/32 | A2→A32 expert −54.7% (M128), −35.7% (M512), −22.7% (M1024), −16.6% (M2048), −1.1% (M4096), −6.2% (M8192); critical wall −57.6…−8.6% | H1b/H1c: active-expert ceiling and kernel regime; check natural-route proximity |
| 2026-09-05 02:40–02:42 | H1 A32 extension repeat, 100 iterations/point | 30 cases × 100 iterations, same Qwen3-VL layer24 grid | A2→A32 expert +3.7% M128, +9.3% M512, +3.6% M1024, +1.5% M2048, −0.8% M4096, −1.4% M8192; wall all within +1.7/−0.8% | first positive grid not reproducible; H16/H23 state control |
| 2026-09-05 02:43–02:47 | H16 same-M order permutations, Qwen3-VL M512 A2/A32 | 2 processes × 50 iterations/case | A32-first A2→A32 expert +60.8%, wall +134.6%; A2-first expert −37.3%, wall −54.0% | sign flip identifies first-use/worker state confound |
| 2026-09-05 02:47–02:50 | H1 A32 layer4/44 M512 | 2 independent runs × 20 iterations/case | A2→A32 expert −44.2% (L4), −47.6% (L44) after A2-first; supports layer/shape/state interaction | require global per-shape prewarm before any policy claim |

The loop is not complete until the active queue has been revisited with new
live data or explicitly marked blocked by runtime constraints.
