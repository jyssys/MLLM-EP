# Exact and predicted-input execution contracts

## Physical model and indexing

The checkpoint has 24 physical transformer layers. Fixed R4 executes 96 logical layers for each input position.
The candidate divides the physical layers into four stages: `[0,6)`, `[6,12)`, `[12,18)`, and `[18,24)`.
Each logical attention layer owns a separate cache slot. The official index is `24 * recurrent_depth + physical_layer`.

Each transformer layer contains two residual additions and sandwich normalization.
The model applies its terminal RMSNorm after each complete physical-layer pass.
The normalized result becomes the next recurrent pass input. The candidate adds no boundary normalization.

## Exact dependencies

For each generated position, the exact chain is:

```text
Embedding -> F1(0) -> F2(0) -> F3(0) -> F4(0) -> Norm
          -> F1(1) -> F2(1) -> F3(1) -> F4(1) -> Norm
          -> F1(2) -> F2(2) -> F3(2) -> F4(2) -> Norm
          -> F1(3) -> F2(3) -> F3(3) -> F4(3) -> Norm -> Head
```

Every stage reads historical KV from its own depth and physical layers.
Every stage writes fresh current-position KV. The next generated token reads these committed values.
The final head explicitly selects recurrent output index 3. Earlier-logit selection does not remove executed layers.

## Relaxed dependencies

Warm-up 1 preserves the first recurrent pass exactly.
At each later round boundary, the emulator creates all three predicted inputs before stage 1 executes.
Stage 1 receives the available round input. Stages 2–4 receive frozen predictions.
The round output comes from stage 4. The official terminal norm still executes before the next round.

```text
Previous round states + current round input
                      |
               Freeze predictions
                 /    |    |    \
                F1   F2   F3    F4
                 \    |    |    /
                  Round boundary
                       |
                  Norm of F4
```

This diagram describes a proposed dependency graph. It is not a measured GPU timeline.
The single-GPU emulator executes these stages serially with the frozen inputs.
Current-round stage outputs cannot replace another stage's frozen input.

Warm-up 0 also predicts the first round inputs. Its initializer uses the current embedding and calibration-only coefficients.
It uses no exact hidden state from the current evaluation token.
Later rounds use the same causal delta rule as warm-up 1.

## Causal predictors

Stale prediction copies the computed preceding-stage output from the previous round.
In exact execution, this tensor equals the next stage's input.
In approximate execution, the computed output can differ from the input prediction.
The emulator retains computed outputs, not recycled input predictions.
Delta prediction adds a calibrated multiple of the change in round input.
The coefficient is a scalar for each recurrent round and stage boundary.
Warm-up 0 uses a calibrated scalar and a constant residual vector for each first-round boundary.

The fit uses eight GSM8K training questions. Evaluation questions and answers do not enter predictor fitting.
No rollback, exact correction, expert replacement, or model-weight update occurs.
All arms retain exact full-depth prefill. Only subsequent decode positions use predicted inputs.

## Availability and cache checks

The availability trace records logical token position, round, stage, source identities, and host enqueue order.
Same-stream ordering establishes causal execution in the emulator.
Host enqueue timestamps are not GPU completion measurements or cross-GPU clock measurements.
The diagnostic cache observer checks all 96 writes in order for each teacher-forced position.

Batch evaluation uses left padding with explicit masks and position IDs.
EOS row compaction removes completed requests before the next model invocation.
Candidate KV persists across every subsequent decode token. Exact reference KV does not replace it.

## Architecture bounds

The 16/7 and 16/4 ratios assume equal stage costs and zero communication, prediction, and synchronization costs.
They compare against 16 serial stages, not optimized TP4 or DP4.
These ratios are analytical bounds from the specification. They are not measured or predicted serving speedups.
