# Native TP quality calibration

## Scope

This comparison contains native fixed-R4 baselines only.
Both checkpoints are Base Ouro, not Thinking checkpoints.
Each arm uses the frozen eight GSM8K and eight MATH-500 calibration requests.
The held-out partition remains untouched.
No candidate changes the computation in these runs.

The raw generations use greedy sampling, native EOS, frozen text stops, and an8192-token cap.
The scorer requires a correct final answer and a completed output.
A length-capped answer does not count as a correct completion.

## Observed results

| Model | Dataset | TP1 correct/N | TP4 correct/N | Exact output parity/N | TP1/TP4 truncations |
| --- | --- | ---: | ---: | ---: | ---: |
| Base1.4B | GSM8K | 3/8 | 4/8 | 5/8 | 0/0 |
| Base1.4B | MATH-500 | 4/8 | 4/8 | 1/8 | 0/2 |
| Base2.6B | GSM8K | 7/8 | 7/8 | 7/8 | 0/0 |
| Base2.6B | MATH-500 | 4/8 | 4/8 | 0/8 | 3/4 |

The [comparison artifact](../artifacts/native_quality_tp1_tp4_calibration_v1.json) records request-level paired bootstrap95% CIs.
It uses4000 resamples of eight paired requests per dataset and model.
It also records every first divergent token and both raw result paths.
The [scored checkpoint](../artifacts/quality_checkpoint8.json) retains parser identity and request outcomes.

Equal aggregate correctness does not establish token parity.
Equal observed correctness also does not establish population quality equivalence.
Eight requests per dataset cannot resolve a small noninferiority margin.
These scores do not reproduce an official full-benchmark result.

## Consequence for performance comparisons

Native TP sizes can produce different natural continuations under the same frozen prompts and sampling settings.
This comparison does not identify the numerical cause.
The result prevents an unconditional interpretation of natural elapsed-time ratios as systems speedup.
Controlled-work timing remains separate from this quality evidence.

The current data do not establish a new mechanism's quality.
No candidate receives a correctness or quality PASS from these baselines.
Any later candidate must compare against its matched native execution mode.

## Reproduction

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.compare_native_quality_tp --scored artifacts/quality_checkpoint8.json --output artifacts/native_quality_tp1_tp4_calibration_v1.json
```

Use a new output path if the recorded artifact already exists.
The script refuses to overwrite it.
