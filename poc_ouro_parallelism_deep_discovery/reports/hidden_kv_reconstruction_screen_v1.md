# Hidden-state publication instead of terminal KV

## Question and outcome

This is a narrower variant of existing state-ownership and KV-rematerialization screens.
It adds no candidate card or runtime method.
It asks whether adaptive depth owners can publish terminal state by sending normalized layer inputs instead of full K/V.
The reference remains original-gate `last_exited` inference.

The mechanism overlaps directly with generic hidden-state restoration.
[HCache](https://arxiv.org/html/2410.05004v1#S3) stores intermediate activations, transfers them, and rebuilds K/V through projection.
Its pipeline balances restoration compute and I/O.
Its source result does not establish low-request H100 peer-transfer profitability for Ouro.

The outcome is `NO_DISTINCT_MECHANISM_ESTABLISHED`.
Physical rebuild cost, BF16 cache parity, and end-to-end benefit remain unmeasured.
The CPU controls prevent an invalid zero-cost reconstruction argument.

## Required state

The sender must retain each physical layer's exact normalized QKV input from the terminal pass.
The final pass output alone does not contain those layer inputs.
Every record needs request, position, physical layer, terminal depth, and publication epoch.
The receiver also needs original weights and the correct RoPE position.

The original gate determines whether a pass is terminal.
Publishing a terminal version cannot consult a future gate decision.
Tentative early transfers need private state and cancellation accounting.
A late copy must complete before dependent attention reads that historical position.
Sending layer inputs does not eliminate this dependency.

## Target-shaped structural accounting

Both loaded Base models use H2048, sixteen MHA heads, and BF16.
One normalized layer input contains 4096 B per token.
One full K/V pair contains 8192 B per token.

| Base model | Physical layers | Layer inputs/token | Terminal K/V/token | KV-only weight payload | Native fused-QKV weight payload |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.4B | 24 | 98304 B | 196608 B | 384 MiB | 576 MiB |
| 2.6B | 48 | 196608 B | 393216 B | 768 MiB | 1152 MiB |

The input payload is half the K/V payload under this construction.
The table does not count source capture, buffering, metadata, or publication work.
Weight payload means tensor size, not measured HBM traffic.
Weights can remain resident without remaining in on-chip cache.

An optimistic KV-only rebuild needs 402653184 or 805306368 FLOPs per token across the physical layers.
Reusing the original fused-QKV operator instead raises those counts to 603979776 or 1207959552 FLOPs.
The latter also produces unused Q values.
Batch shape, kernel choice, rounding, RoPE, and cache packing still need numerical validation.
Dropping Q from a fused projection does not automatically preserve its BF16 output words.

The [CPU artifact](../artifacts/hidden_kv_reconstruction_screen_v2.json) sweeps rows 1,4,16,32,128.
It reports structural bytes and FLOPs only.
It does not convert those quantities into measured latency or speedup.

## Why absorption does not supply free reconstruction

A position-independent query pullback is invalid for general existing RoPE weights.
The exact-rational CPU fixture uses two key positions with different rotations.
Native scores are [1, -1].
A constant pullback produces [1, 2].
The position-dependent pullback restores [1, -1].
This positive control distinguishes algebra failure from a claim that reassociation is always impossible.

Removing the original BF16 key-projection boundary also changes results in a separate toy.
Its head dimension is 128 and its query's nonzero value is 1.5.
The native BF16 output is 0.53125.
The reordered BF16 output is 0.53515625.
These values are CPU counterexamples, not observed Ouro attention outputs or task-quality losses.
An [actual Torch CPU control](../artifacts/hidden_kv_torch_cpu_validation_v1.json) reproduces both rounded outputs.
It also checks the software BF16 rounder against 8193 Torch CPU values.
That check does not measure GPU accumulation order or pretrained-model behavior.

The [DeepSeek-V2 architecture](https://arxiv.org/html/2405.04434v5#S2) uses a separate positional path for its latent-attention design.
That source does not authorize changing the pretrained Ouro attention architecture.
The [hardware analysis](https://arxiv.org/html/2506.02523v1) also treats projection reuse and reconstruction as distinct compute-memory tradeoffs.
No source-reported speedup transfers to this experiment.

## Charged decision condition

For a declared serial critical path, the alternative needs this inequality:

```text
capture + hidden transfer + KV rebuild + RoPE/packing + publication
    < terminal-KV transfer + baseline publication
```

If operations overlap, compare legal event-DAG makespans instead of adding overlapping service times.
The current transport microbenchmarks do not measure this complete path.
In particular, a maximum rank duration is not pure link bandwidth.
The campaign therefore records the condition without inventing its numerical solution.

This screen supplies neither a new parallel axis nor a positive physical headroom result.
A future different regime needs measured reconstruction, valid state publication, and a matched generic HCache-style control.

## Reproduction

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.hidden_kv_reconstruction_screen --output artifacts/hidden_kv_reconstruction_screen_v2.json
```

The script loads no checkpoint and calls no GPU API.
Its toy values do not substitute for a pretrained-model experiment.
