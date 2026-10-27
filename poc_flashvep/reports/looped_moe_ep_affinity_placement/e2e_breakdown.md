# Whole-E2E component attribution

## Status

This file contains only completed attribution runs.
The attribution campaign waits for the active accuracy and control campaigns.
A queued configuration is not a measured result.

## Measurement contract

The denominator is the complete observed CUDA-event span for 32 greedy output tokens.
The rollout includes prefill, subsequent forwards, sampling, and output completion.
These official model paths recompute the full prefix. They do not use cached decode.
EOS is disabled in this hardware comparison. Official benchmark accuracy uses separate task protocols.

Each component occupies an exclusive part of the endpoint timeline.
Nested intervals do not count twice. Unrounded component percentages sum to 100%.
The table does not sum different ranks or multiply one forward by the output length.

These API spans include GPU execution, submission gaps, and stream waits.
A dispatch or combine span is not pure network transfer time.
An expert span is not pure GEMM service time.
Rank waiting remains inside the corresponding communication span unless a separate measurement can identify it.

Clean host latency and observer latency remain separate.
The timing gate requires at most 5% mean observer overhead and at most 1% event-to-host disagreement.
A passing gate does not prove unbiased clean component shares.
The report never rescales observer percentages into claimed clean measurements.

## B1, prefix 128, original static EP

Status: **PENDING_MEASUREMENT**.
No whole-E2E component percentage is available yet.
Existing per-forward traces do not replace this measurement.

## Complete data

- `e2e_component_percentages.csv` retains each observer detail and timing status.
- `e2e_component_percentages_raw.csv` retains each traced rollout and its evidence path.
- `e2e_component_phase_percentages.csv` separates first output completion from the remaining outputs.
- `raw/*_e2e_share_*/attribution_status.json` retains clean repetitions and observer timing gates.
- `e2e_attribution_contract.md` defines the accounting rules and interpretation limits.

The first-output phase includes prefill and first-token sampling.
Phase component percentages identify both their phase denominator and the whole-E2E denominator.
A lower component percentage alone does not prove a lower latency. Compare the raw milliseconds too.
