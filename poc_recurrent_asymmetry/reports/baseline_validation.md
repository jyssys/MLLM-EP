# Baseline validation

## Exact reference: PASS

The reference uses the pinned official HF implementation without a model patch.
It sets `exit_at_step=3` and retains all four recurrent passes.
The checkpoint sets the exit threshold to 1.
The code executes 24 layers per pass and applies the terminal normalization after each pass.
Selecting an exit output does not demonstrate skipped computation in this implementation.

Cache index `24 * recurrent_depth + physical_layer` selects an independent slot.
The probes check 96 distinct key storages and 96 distinct value storages.
They do not alias depths or overwrite one common KV slot.

| Check | Measured result | Scope |
|---|---|---|
| Padded B1/B2, four cached steps each | 8/8 comparisons pass | Full logits and all KV tensors are bitwise equal |
| Actual layer execution | 96 per checked forward | Ordered `(depth, layer)` list |
| Natural regression request 1 | 251/251 tokens match | Includes EOS |
| Natural regression request 2 | 772/772 tokens match | Includes EOS |
| GSM8K smoke | 2/2 correct | Greedy natural completions |
| MATH-500 smoke | 2/2 correct | Greedy natural completions |

The smoke outputs contain 656, 509, 1,654, and 1,297 generated tokens.
All four completions reach EOS within the 8,192-token limit.
These small samples do not estimate full benchmark accuracy.
They do not reproduce the paper's evaluation protocol.

Raw evidence is in `outputs/baseline_v1/integrity.json` and the four scored output files.
The scoring code and prompt template come from the previous exact-R4 study.
This study imports its loader and cache inspection helpers only.
It does not invoke that study's predictor experiments.

## Observer parity

The broad component observer preserves full logits and all KV tensors bitwise.
The single-family observer preserves its final logits bitwise.
The Nsight lane preserves all 30/30 checked logits across six actual requests.
These checks establish output parity for observers, not negligible observer latency.

## Optimized native controls: REFERENCE_PARTIAL

The numerical limits precede the native comparisons.
They remain unchanged after failures.

| Quantity | Limit |
|---|---:|
| Full teacher-logit maximum absolute difference | 0.5 |
| Full teacher-logit mean absolute difference | 0.05 |
| Sampled KV relative L2 difference | 0.02 |
| Natural token prefixes | Two exact 64-token prefixes |

| Native control | Mean logit difference at 1K / 16K | Sampled KV relative L2 range | Natural prefixes |
|---|---|---|---|
| TP1 | 0.05591 / 0.06222 | 0.01326–0.01614 | 2/2 match |
| TP2 | 0.06332 / 0.06198 | 0.01400–0.01638 | 2/2 match |
| TP4 | 0.06262 / 0.05775 | 0.01408–0.01746 | 2/2 match |

Every native control exceeds the mean-logit limit.
The sampled KV checks pass, but they do not inspect every KV element.
Each comparison uses four teacher-aligned positions at 1K and 16K.
All checked top-1 predictions match.
The two natural prefixes are length-limited outputs, not complete benchmark answers.

These native measurements remain hardware diagnostics.
They do not qualify an exact cross-backend architecture GO decision.
The study does not relax the limit or claim full quality equivalence.

## Serial chunking: REFERENCE_PARTIAL

A later bounded probe checks full 96-slot KV tensors, rather than selected KV coordinates.
It uses two calibration requests, four chunk sizes, and three cached continuations.
The 32 checkpoints include the prefill result and each continuation.

| Check | Result |
|---|---:|
| Top-1 equality | 32/32 |
| Numerical limits pass | 8/32 |
| Worst logit maximum absolute difference | 0.625 |
| Worst logit mean absolute difference | 0.0920164 |
| Worst full-KV relative L2 difference | 0.0776894 |

All eight passing checkpoints use the unsplit 1,024-token chunk.
The smaller chunk shapes change floating-point execution and fail the frozen numerical gate.
The study did not establish their numerical equivalence.
It does not assign the complete difference to BF16 rounding without further evidence.

The native chunk code preserves the intended causal semantics.
However, intended semantics do not replace the failed numerical check.
The wavefront cost model is conditional on a future qualified execution path.
There is no physical wavefront engine, candidate natural generation, or candidate accuracy result.
Those fields are `NOT_RUN`.
