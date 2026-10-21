# Eight-arm discovery analysis

Updated UTC: 2026-10-21T08:36:29.235050+00:00.

## Current decision

The campaign is IN_PROGRESS. No final architecture GO/NO-GO verdict is issued in this snapshot.
The current controller stage is `Static pilot`.
`eight_arm_matrix.md` lists completed samples and remaining execution states.

## What the numerical evidence supports

Real TP4 exists. The model has sharded attention/FFN weights and actual four-rank collectives.
The old bounded operator audit passed 192/192 operations. The historical full-model P512 gate remains FAIL.
The new five-engine synthetic study contains 112 teacher-aligned positions across eight synthetic requests.
Every BF16 pair agrees on argmax at those positions. Native TP1 already differs from HF internally.
These observations do not prove benchmark accuracy or identify one unique numerical root cause.
See `numerical_tp_diagnosis.md`, `first_divergence.csv`, and `logit_metrics.csv`.

## What the serving evidence must distinguish

Loop PP reduces physical stage handoffs relative to re-entrant Layer PP.
That count reduction is structural accounting, not a measured E2E speedup.
The primary controlled replay uses common no-refill waves and identical token work.
The additional matched-natural cohort uses the same successor rule on all eight arms.
The larger Static task cohort preserves legacy asynchronous PP and therefore has a scheduling confound against TP.
The native vLLM controls remain separate from custom backend comparisons.
A four-GPU ratio against Single uses four times the GPU count. It is not a claim of four-GPU resource efficiency.

## Dynamic interpretation

The locked method uses the original CDB cumulative-hazard gate, q=0.5, R2-R4, full-depth prefill, and last_exited KV.
The prototype must physically skip later loops after exit.
Current exit KV rows must reach every deeper owner before the next token starts.
Loop PP can therefore add exit-KV communication that does not exist in layer-local filling.
This is a prospective physical hypothesis. Its latency contribution requires actual timing.
Do not mix this method with another paper's gate or KV policy.

## Completion requirements

The final review must check all eight quality rows, numerical warnings, dynamic reference checks, five-repeat replay, and actual overlap timelines.
It must report blocked arms rather than invent results.
Only measured matched serving evidence can support an architecture conclusion.
Full public-suite evaluation is USER_REDUCED_NOT_RUN under the user's 400-question amendment.
