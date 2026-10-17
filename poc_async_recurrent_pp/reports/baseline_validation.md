# Exact R4 baseline validation

## Model and reference

The model is `ByteDance/Ouro-1.4B-Thinking`.
The pinned checkpoint revision is `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`.
The official model source SHA-256 is `cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30`.
The experiment uses the unmodified official forward implementation.
No model weight or custom model-source patch applies.

The loaded model has 24 physical layers and four recurrent passes.
Its hidden size is 2,048. It has 16 query heads and 16 KV heads.
The FFN intermediate size is 5,632. The vocabulary has 49,152 entries.
The loaded parameter storage accounts for 2,869,305,346 bytes.
The config declares 65,536 positions, while the model card describes 32K SFT context.
This experiment does not test those context limits or extend RoPE.

The reference environment uses PyTorch 2.8.0+cu128, Transformers 4.57.1, and CUDA 12.8.
The checkpoint includes a cache compatibility fix. The experiment validates this actual revision rather than assuming older-library behavior.
The attention backend is SDPA. Weights and model compute remain BF16.

## Executed integrity checks

| Check | Actual scope | Result |
|---|---|---|
| Official vs instrumented prefill logits | 16 smoke prompts | Bitwise equal |
| All depth-indexed key and value tensors | 16 smoke prompts, 96 slots each | Bitwise equal |
| Stage-boundary capture count | 16 smoke prompts, 4 rounds × 4 stages | PASS |
| Official `generate()` vs custom greedy loop | 4 complete natural generations | Exact token parity |
| EOS behavior | Same 4 natural generations | EOS in both implementations |
| Independent cache storage | 96 key tensors per checked request | PASS |

The four natural outputs contain 251, 772, 831, and 639 tokens.
These are arithmetic integrity probes, not benchmark accuracy measurements.
The output cap is 2,048 tokens. None of these four outputs truncates.

The official cache index is `24 * recurrent_depth + physical_layer`.
Every checked cache slot has the same logical sequence length.
Different depth/layer slots have distinct storage.
The official terminal norm executes after every recurrent pass.
`exit_at_step=3` explicitly selects the fourth output after all four passes execute.

## Predictor calibration

Eight disjoint GSM8K training questions supply 136 observed positions.
The calibration includes the last prefill position and 16 generated positions for each question.
The fit uses no evaluation answer or evaluation activation.
The predictor manifest preserves all coefficients, question IDs, and arithmetic conventions.
Diagnostic copies use FP32. The actual model remains BF16.

The original exact-only integrity run predates the corrected output-retention hook.
The corrected emulator repeats teacher-forced logits and all-slot cache parity in `diagnostics_v2`.
The analysis excludes the first approximation diagnostic because the implementation audit identified incorrect retained-state bookkeeping.
The audit and failed diagnostic evidence remain available.

## Quality scope

Baseline task sanity and paired pilot accuracy are separate from these integrity checks.
The accuracy report records their actual sample counts, natural completion rates, and parsing failures.
These checks do not reproduce a published full benchmark score.
Full GSM8K and MATH-500 evaluation is mandatory before any quality GO claim.
