## 10. Physical timing and observer validity

Separate independent clean bypass, event-only timing, and heavy routing passes.
Warmup3 denoising requests precede measurement; bypass/timing order is
counterbalanced by frozen request index. No per-layer global host synchronization
or `.item()` was added. Combine's existing native completion-event dependency is
placed before timing-end; its own subsequent native wait is unchanged.
The chosen critical rank is the one with largest cumulative denoising-forward
CUDA time; Dispatch/Expert/Combine all use **that same rank**, not separately
maximized component ranks. Router here means native gate linear time only;
top-k/layout/shared expert/source all-gather are not silently counted as expert.

| Domain | Gate ms | Dispatch ms | Expert ms | Combine ms | Forward sum ms | Denoise span ms | Clean request wall ms | Observer overhead |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 76.92 | 199.67 | 495.77 | 127.53 | 2754.04 | 2791.71 | 2623.89 | 8.65% |
| humaneval | 19.16 | 201.29 | 189.21 | 106.97 | 1370.94 | 1390.36 | 1372.52 | 7.91% |

All stage times are request totals for denoising calls, not per-layer averages.
Forward-only communication share 19.523% (request-bootstrap95% CI 15.884–22.919%) is
diagnostic: it omits acceptance/controller/cache gaps. First-to-last denoising
model span includes inter-iteration gaps, but excludes final acceptance epilogue.
The promotion gate conservatively divides denoising D+C by **whole request wall**,
which also includes initial prompt prefill. This lower bound cannot inflate the
communication share by deleting controller time. No speedup is measured.

Instrumentation threshold was frozen at10% absolute mean paired overhead plus
exact output/NFE parity. Heavy trace overhead: 14.307% (request-bootstrap95% CI 13.361–15.591%);
heavy trace wall is never a headline latency. All requests and failures are
retained, no fastest-run selection. Missing tuned Triton mini expert config,
20 DeepEP SM default, eager manual bridge and disabled TorchDynamo are important
runtime-specific limitations; no claim about every optimized dLLM server.

