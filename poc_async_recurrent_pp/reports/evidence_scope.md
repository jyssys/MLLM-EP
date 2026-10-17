# Evidence validity and exclusions

## Valid measurement types

The exact reference checks use the pinned official Ouro forward and generation code.
The model remains BF16 with fixed R4 and 96 independent depth-indexed KV slots.
The numerical emulator changes only the selected stage-input tensors during decode.
It preserves exact prefill, official masks, position handling, and greedy EOS rules.

The four quality arms use independent full-model replicas on physical GPUs 4–7.
Each replica performs serial numerical emulation on one GPU.
This arrangement supports a quality comparison of the proposed concurrent input semantics.
It does not demonstrate concurrent pipeline execution or four-GPU speedup.

The primary source is `outputs/pilot_cap8192_v2/` after every worker completes successfully.
The common 8,192-token cap follows the frozen expansion rule for the 2,048-token screening wave.
Both waves use the same questions, batch groups, coefficients, and greedy settings.
Repeated caps and arms do not create additional independent questions.

The final analysis rescans all primary raw outputs with parser version `parpp-final-v2`.
It stores new grades in a separate directory.
Original generations and their encoded grades remain unchanged.
The decimal parser correction uses a generic hand-written formatting regression test.
It changes no generated token or predictor coefficient.

## Preserved exclusions

| Evidence | Validity limit |
|---|---|
| `outputs/diagnostics_v1/` | Invalid approximation bookkeeping. The first version recycled input predictions instead of retaining computed preceding-stage outputs. |
| `outputs/batch_reference_v1/` | Failed long generation parity with custom EOS row compaction. |
| `outputs/pilot_cap2048_v1/` | Excluded from primary quality because it used the rejected compacting driver. |
| `outputs/compaction_probe_v1/` | Diagnostic only. Its first reduced-batch forward matched, so the later divergence's exact cause remains unresolved. |
| `outputs/pilot_cap2048_v2/` | Valid shorter-cap screening. The complete higher-cap wave is primary after truncation triggered expansion. |
| `outputs/validation/cpu_tests_final.json` | Failed invocation under system Python without PyTorch. It is not a model test failure. |

The valid pinned-environment test result is `outputs/validation/cpu_tests_final_v2.json`.
All 11 CPU regression tests pass in that environment.
The earlier failed invocation remains available.
CPU test success does not establish model quality or distributed correctness.

## Statistical limits

The initial pilot contains 16 GSM8K test questions and 16 MATH-500 questions.
Eight exact task-sanity questions overlap this pilot.
They are not an independent confirmation cohort.
Predictor calibration instead uses eight disjoint GSM8K training questions.

The bootstrap uses paired questions, not tokens, stages, layers, or loops.
It uses 10,000 resamples with seed 20261017.
Pilot rejection concerns a large loss under this frozen setup.
It does not establish complete benchmark scores or universal failure of every possible causal predictor.

The controller requires a large regression in both benchmarks before its automatic `QUALITY_FAIL` decision.
The independent final adjudicator also records rejection on each individual mandatory benchmark.
A candidate must pass both required benchmarks to support quality GO.
Both decision paths retain the same numerical regression threshold and the same two accuracy endpoints.

## Timing limits

Single-GPU quality wall times include Python hooks, cache growth, and output observation.
Different natural output lengths also change those times.
These observations are not optimized serving latency comparisons.

Availability events use one local GPU clock.
The diagnostic records 2,712 source-before-prediction checks and 908 prediction-before-consumer checks.
These checks support causal emulator execution, not cross-GPU timing alignment or concurrent execution.
No kernel-service total becomes an E2E speedup claim.
