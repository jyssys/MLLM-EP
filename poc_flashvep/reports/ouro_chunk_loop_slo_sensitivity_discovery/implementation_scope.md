# Implementation scope

The experiment reuses the previous isolated RLT fork and its G2 adapter.
The RLT revision is `4a350344027960a48384626aa646f1765fda6d4f`.
The existing patch changes three files with 23 additions and nine deletions.
This study does not change the model, attention kernel, cache arithmetic, or executor math.
It does not implement a native-vLLM Ouro engine, TP, PP, or a trained scheduler.

New code resides in `poc_flashvep/ouro_chunk_loop_slo/`.

| File | Responsibility |
| --- | --- |
| `dispatch.py` | Four independent single-GPU processes. Dynamic queue, identity, per-GPU compiler caches, telemetry, and resume markers |
| `worker.py` | Calibration, closed-cohort runs, arrival/token clocks, numerical/observer/profile entry points |
| `traffic.py` | Frozen Poisson/burst inputs, actual natural stops, declared drains, and no-chunk control |
| `numerical.py` | Isolated numerical probes and diagnostic LM-head replay |
| `characterize.py` | Actual full-R4 work counts and per-token host submission exposure |
| `analyse.py` | Paired confidence intervals, all-offered SLO definitions, and exploratory rate brackets |
| `controls.py` | Same-GPU crossover, interference, observer, and bounded-output parity |
| `figures.py` | Measured response surfaces, distributions, Pareto points, and SLO curves |
| `pack.py` | Lossless evidence compression and checksum verification |

Every child receives one allowed physical GPU through `CUDA_VISIBLE_DEVICES`.
The runtime additionally checks the expected physical UUID.
It asserts exactly one visible CUDA device.
No distributed collective or model-parallel process group is required.

Ordinary compared policies use the same 1,024-token budget and 54GiB KV reserve.
The no-chunk control raises the token budget to 16,384.
It selects one whole prompt at a time without reducing resident decode capacity.
That exception prevents a partial prompt from masquerading as unchunked prefill.
It is not a matched pure C/G comparison.

Natural serving uses native EOS0 and observed chat-turn end token2.
The frontend cancels only after the model actually emits token2.
No future output token is supplied to the engine.
The continuity lane retains the original 128-token cap without that frontend stop.
The two completion contracts remain separate.

Adaptive Joint and multi-GPU serving portability remain conditional stages.
Independent experiment workers do not satisfy either stage by themselves.
