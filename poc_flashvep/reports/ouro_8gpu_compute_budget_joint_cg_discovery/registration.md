# Prospective registration

The model is pretrained Base Ouro-1.4B, BF16, Static R4 for every real prompt and decode token.
The new study uses eight independent single-GPU workers.
It does not implement TP/PP or model early exit.

Primary experimental contrast: Joint versus tuned adaptive C-only with identical fairness, budget controller, backend, and arrivals.
Secondary contrast: Joint versus native and fairness-controlled RLT Mixed.
The practical screening target is at least 5% goodput or qualified rate gain.
Its paired 95% interval must exclude zero.
TTFT p95 regression must not exceed 5%, and quality must not regress.
Exploratory cells do not establish production capacity.

The initial C menu is 64/128/256/512/768/1024/1536/2048.
The G menu is 1/2/3/4, clipped at the legal remaining depth.
Fixed budgets are 10/20/40/60/80/120/160/240 ms plus unlimited.
Budget is not a next-token deadline or TBT bound.
The budget-overrun screening target is at most 5% per decision-relevant prefill path.
Failure permits a labeled exploratory lookup/fallback lane, not a bounded-service claim.

The primary service boundary starts after native request-dependency waits and before GPU preparation.
It ends after the native model execution on that stream.
CPU selection, dispatch, queued waits, and client latency remain separate fields.
Required staged prelude belongs to a new chunk's total opening cost.
Continuations retain the original staged token span.
Only Full-R4 publishable prompt spans receive completed-work credit.

All methods receive the same bounded completed-token fairness mode in matched comparisons.
Stock fairness remains an independent control.
Fairness cannot credit an issued recurrent batch as a completed output.

Before held-out evaluation, freeze fresh source-family partitions and path-specific cost margins.
Previous HELDOUT inputs remain previous evidence, not a new blind set.
Single-step discrete enumeration precedes any optional multi-step solver.
Unreliable estimates do not automatically stop safe comparative pilots.

Capacity initially requires at least 1,000 offered requests per arm/rate across three independent starts.
Queue stability and confidence-aware attainment remain mandatory.
The SLO grid will freeze from decode-only calibration before candidate held-out outcomes.
