# Seed and numerical reproducibility

This report separates repeatability within one engine from repeatability across engine sessions.
Neither check establishes equal complete-output quality across runtimes.

## Observed checks

The prompt diagnostic repeats four sampled P1 requests within one TP4 engine.
All four 8,192-cap output sequences repeat exactly.
The full quality engine also repeats its initial 256-token seed probe exactly.
Selected same-configuration teacher states repeat bitwise.
These checks do not cover every request or every stochastic trajectory.

The first full-quality wave differs from the corresponding prompt-diagnostic wave.
Both waves use the same frozen four requests, engine arguments, requested cap, and per-request seed.
Their earlier request histories differ.
This comparison therefore does not isolate a restart with an identical complete history.

All four output sequences differ across these two engine sessions.
The first differing indices are 180, 333, 1,926, and 39.
The full collection preserves these differences.
The collection does not restart failed or low-scoring samples.

Actual KV pools match across the compared sessions.
Each TP4 rank has 11,081 blocks and 69,715,623,936 allocated KV bytes.
The cache geometry and compilation configuration also match.
Different actual pool geometry therefore does not explain this particular comparison.
Batch history and arithmetic remain possible causes.
Equal engine arguments do not establish equal internal execution histories.

## Installed sampler path

The native startup log names FlashInfer sampling support.
The installed CUDA sampler uses another branch when per-request generators exist.
That branch calls the PyTorch-native sampler.
The model runner creates a device generator and applies the request seed.
The native sampler draws exponential noise with that generator.

This source audit does not measure generator offsets during the observed failure.
It also does not capture the first differing pre-sampling logits.
The restart difference therefore remains causally unresolved.

The seeded source path does not prove a defective sampling distribution.
The observations also do not establish universal restart determinism.
The final report must preserve both limits.
No checkpoint, sampling setting, graph policy, or full-quality prompt changes because of this diagnostic.

## Official runtime guidance

The [v0.26.0 guide](https://docs.vllm.ai/en/v0.26.0/usage/reproducibility/) does not guarantee output reproducibility under default execution.
It describes deterministic offline scheduling and batch invariance as separate controls.
The primary frozen campaign retains the ordinary native configuration.
The [v0.26.0 batch-invariance list](https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/) does not list Ouro among its validated models.
The campaign does not assume that beta mode supports Ouro or preserves headline native performance.
This guidance does not identify the cause of the observed difference.

`restart_history_lock.json` defines a separate fixed-history control.
It repeats the full-quality engine's two 256-token waves before the first 8,192-token wave.
Only a failed fixed-history control authorizes two additional deterministic-scheduler probes.
Those probes do not replace primary quality outputs.

## Scientific tests versus code tests

`tests/test_restart_property.py` checks an observed scientific property.
Its expected equality fails on the stored cross-session outputs.
That failure remains an experiment result, not an unreported passing test.
The contract and logit-metric regression tests check separate implementation properties.
Their status cannot replace the failed restart property.

Evidence: `results/quality_engine_restart_reproducibility.json` and `results/seeded_sampler_source_audit.json`.

## Completed fixed-history controls

The fresh default-core control repeats the same client calls as the first quality wave.
The call caps are 256, 256, 8,192, and 8,192 on the same four frozen prompts.
The first warm outputs match the quality run in all four requests.
The first long outputs differ from the quality run in all four requests.
The first divergence indices remain 180, 333, 1,926, and 39.
The two long waves within the fresh control match exactly.

Matching the client history does not reproduce the original long wave.
It does not establish identical internal scheduling or arithmetic.
The first differing model logits and actual request-generator offsets remain NOT_CAPTURED.
The experiment does not assign the difference to one operator.

The separate control sets `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
Two fresh sessions reproduce all four compared long outputs exactly.
Both sessions also reproduce their own second long wave.
Those outputs differ from the original primary run.
This four-prompt diagnostic does not establish universal determinism or a new benchmark score.
No primary output changes to the alternate scheduler configuration.

The installed sampler also passes a separate fixed-logit GPU test.
Fresh processes, in-process repeats, and the first row of B1/B4 agree in the tested draws.
The probe uses 128 repeated sampling steps per condition.
It does not execute a model forward or capture the real model request's RNG offsets.
Its agreement cannot prove complete-output determinism.

The reported benchmark confidence intervals condition on the executed native sessions.
They do not estimate unmeasured between-session runtime variation.
These reproducibility limits remain part of the weak-quality decision.

The first history attempt fails before generation because of a burner restoration race.
The fixed retry preserves the same model and engine arguments.
See [restore_race.md](restore_race.md) for that independent experiment-control failure.

Evidence: `results/restart_history_gate.json` and `results/fixed_logits_sampler_comparison.json`.
