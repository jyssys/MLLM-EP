# Environment and reuse

## Model and local runtime

| Field | Value |
|---|---|
| Model | Base `ByteDance/Ouro-1.4B` |
| Model revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` |
| CDB source revision | `abdfae0adbdd12be8c6123ccb34eee98a38afb39` |
| Preceding evidence commit | `0eb9c04d1eef6af33c30a0ca2c0244fb26e41950` |
| Precision | BF16 |
| Online exit | q=0.5, minimum R2, maximum R4 |
| KV policy | `last_exited`, 96 depth/layer K tensors and 96 V tensors |
| Prefill | Full R4 |
| Decode sampling | Greedy, native EOS token 0 |
| Decode CUDA graphs | Enabled in timing and quality runs |
| Prefill CUDA graphs | Disabled |
| Async batching / delayed gate | Disabled / disabled, as in the selected v3 runtime |
| Local resident limit | 16 requests |
| KV pool | 2048 blocks × 16 rows |
| Exact KV tensor allocation | 25,794,969,600 bytes, or 24.0234375 GiB, per replica |
| Model-length limit | 10,240 prompt-plus-generation tokens |
| KV pressure policy | Reserve before admission, queue excess requests |

Native Python is `/home/esjung/.venvs/cp-composition-vllm026/bin/python`.
The existing environment supplies PyTorch 2.11.0 with CUDA 12.9.
The isolated runtime imports Transformers 5.9.0 and FlashAttention3.
The experiment reuses the preceding loader and its checkpoint-gate tensor checks.
It does not install or upgrade global packages.
The earlier independent reference comparison remains `REFERENCE_MATCH_PARTIAL`.
This study does not upgrade that result to exact official-reference parity.

The shared warmup uses both local scheduling modes.
Some later shapes can still trigger native lazy CUDA graph capture.
The raw native counters record those captures.
The measured times retain their cost.

The inference sampler uses deterministic argmax, not stochastic sampling.
Seed 20261019 controls question selection and the shuffled-order diagnostic.
The frozen manifest's seed list does not mean seeded stochastic generation trials.
The harness does not reset the PyTorch random generator before each case.
The selected model weights and original gate tensors remain fixed.
The report measures repeat output parity instead of assuming that a seed guarantees it.

## Physical GPUs

All four selected devices are NVIDIA H100 80GB HBM3 GPUs.
The driver reports 81,559 MiB per GPU.
The topology reports NV18 connections between every selected pair.
DP1 uses physical GPU4.
Each DP4 worker sees one selected physical GPU as logical CUDA device 0.

| DP rank | Physical GPU | UUID |
|---:|---:|---|
| 0 | 4 | `6076e2f2-5b63-3761-5586-56ceb7df8139` |
| 1 | 5 | `a1a1cfcf-93a1-3544-9a5e-e58144b68730` |
| 2 | 6 | `e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` |
| 3 | 7 | `4cc26b88-19fc-1988-f9e0-17858aa7a99b` |

This is TP1/EP1/PP1 inside each independent replica.
It is not TRUE EP4 or TP4.
No inter-replica tensor or expert collective belongs to these measurements.

The environment record includes read-only observations of other GPU users.
The experiment does not signal their processes.
Each ownership guard stops only identified owned burners on the requested devices.
The guard restores those burners after its run.
The shared burner script defaults to a different GPU set.
The guard therefore sets both GPU masks explicitly.

## Reused assets

- The v2 runtime loads the original checkpoint and validates the selected gate tensors.
- The v3 wrapper supplies the validated native scheduler admission interface.
- The v3 discovery trace supplies historical future decode work for the speed placement rule.
- The earlier quality manifest supplies version-locked questions, prompts, stops, and answer labels.
- The earlier answer parser supplies unchanged answer-extraction and grading rules.
- The existing ownership guard supplies scoped cleanup and burner restoration.

The new worktree does not modify any of those source files.
`artifacts/continuity.json` stores their hashes.
It also stores the original repository's staged and unstaged diff hashes.

## Frozen workloads

The speed cohort reuses 64 historical discovery requests.
The domains are GSM8K, MATH-500, Alpaca, and ShareGPT.
Nested cohorts use C4, C8, C16, C32, and C64.
Every speed request permits natural EOS with a 256-token cap.
This is bounded-progress hardware evaluation, not complete-answer quality evaluation.

The quality cohort has 128 questions per domain.
It covers 128/1319 GSM8K test questions and 128/500 MATH-500 questions.
Seed 20261019 selects the questions before new inference.
The earlier source manifest retains dataset revision and prompt provenance.
The quality cap is 8192 tokens, with the frozen native stop sequences.
Old split labels remain provenance fields, not a new train/test partition.
This subset is not an untouched held-out confirmation.

The complete five-arm benchmark would require 9095 generations.
The minimum permitted subset requires 1280 generations.
The bounded study uses that subset to preserve all five actual quality rows.
It does not claim full-benchmark accuracy.

See `reproduce.md`, `artifacts/arm_manifest.json`, and the per-run source snapshots for exact commands and configurations.
