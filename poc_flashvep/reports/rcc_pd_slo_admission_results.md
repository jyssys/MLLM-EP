# RCC-PD: conditional admission status and design review

## Status

`NOT_RUN_NOVELTY_GATE` applies to online conditional admission.
This is not an experimental `NO_GO_ADMISSION` result.
No online controller, SLO threshold calibration, or serving-rate sweep ran.
The study does not claim SLO safety or capacity improvement.

The fixed-cut execution path already exists as A plus Native Mixed.
The source spec permits a stop after Stage 1 when no additional fusion mechanism exists.
The implementation audit and physical operand probes establish that overlap with the existing engine.
The fixed-cut comparison remains useful as a baseline ablation.
It does not establish a new RCC execution architecture.

## What remains distinct

Existing A selects a fixed cut for every request.
Native Mixed selects compatible local rows after those requests arrive.
Neither adapter implements the proposed pre-cut P/D credit protocol.
Therefore, conditional admission remains a separate, unimplemented control question.

That controller could improve a P-heavy operating region.
The present evidence does not establish such a region against strong equal-budget DP_Mixed and load-aware deflection controls.
The eight-request mechanism cohorts cannot answer that question.
They also cannot calibrate long-output decode slack.

## Minimal safe protocol for a later, distinct hypothesis

1. D reserves an import slot, KV capacity, and bounded continuation work before P reaches the cut.
2. P offers only the fixed cut or full-depth prefill.
3. D checks current decode rows, observed ITL, queue work, and memory headroom.
4. D estimates incremental continuation delay from disjoint calibration measurements.
5. D accepts only when its conservative delay estimate fits the shortest relevant decode slack.
6. P commits the selected ownership path before it suspends the prompt.
7. A refusal retains prompt computation on P and uses ordinary full-prefill PD.
8. D limits admitted continuation rows per scheduling cycle.
9. Every refusal, prediction, observed delay, and fallback records an explicit reason.
10. The controller never drops recurrent work or changes model depth.

Receive completion alone is not permission to read an in-flight buffer.
The existing producer event and transport-completion conditions remain mandatory.
A failed D reservation cannot trigger an unsound rollback after P releases its state.

## Required controls before a new serving claim

| Control | Required equality | Current status |
|---|---|---|
| Always A + Native Mixed | Same cut, cache, graph settings, and workloads | Small mechanism cohorts only |
| Never RCC / PD_G1 | Same corrected runtime and GPU budget | Small mechanism cohorts only |
| SLO-aware RCC | Same information and fixed-depth work | NOT_RUN |
| DP_Mixed + chunked prefill | Same total GPU budget | NOT_RUN |
| Decode-local prefill / generic deflection | Same admission and SLO information | NOT_RUN |
| Long-output W1 controlled generation | Legal per-request caps and frozen arrivals | NOT_RUN |
| W1 Natural EOS | EOS enabled, observed lengths reported | NOT_RUN |

The comparison must separate state-transfer savings from shared-core batching.
It must also separate admission benefits from ordinary P/D load balancing.
A generic prefill-deflection baseline must receive the same queue and SLO information.

## Long-context prerequisite

The current A adapter requires a complete prompt inside one configured chunk.
A multi-chunk implementation needs legal frontier barriers and retained state for every prompt token.
Removing the length check is not sufficient.
Changing only the candidate chunk size would weaken the fairness of a baseline comparison.

Huginn has a pinned 4096-token total context.
A 512-token output cap limits its prompt to 3584 tokens.
The requested 3–4K prompt stratum therefore needs an explicit legal subset for this checkpoint.
The study does not extend context or silently truncate a workload to satisfy that stratum.

## Decision

Do not interpret the missing controller as evidence that admission cannot help.
The current stop concerns the proposed fixed-cut mechanism's overlap with existing A and Native Mixed.
A later admission study needs a distinct, predeclared hypothesis and stronger controls.
It should not present existing mixed execution as its new contribution.
