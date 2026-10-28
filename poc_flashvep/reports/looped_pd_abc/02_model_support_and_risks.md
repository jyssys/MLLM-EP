# Model support and risks

## Model semantics

| Model | Physical structure | Fixed logical execution | Required KV slots | Allocated slot geometry |
|---|---|---|---:|---:|
| Ouro | 24 shared attention/MLP layers | 24 × 4 = 96 layers | 96 | 96 |
| Huginn | 2 prelude + 4 core + 2 coda layers | 2 + 4 × 32 + 2 = 132 layers | 132 | 8 × 32 = 256 |
| Dory | 16 input + 40 core + 16 output sublayers | 16 + 40 × 2 + 16 = 112 sublayers | 56 attention slots | 36 × 2 = 72 |

The slot counts are structural accounting, not measured latency or speedup.
Huginn and Dory use boundary-layer KV only at depth zero.
The RLT page geometry reserves additional unused boundary-layer positions at deeper planes.
The boundary adapter excludes those unused positions from NIXL payloads.
This is a validity correction. It is not a loop-specific performance contribution.

All models use their pinned official dense BF16 checkpoints.
The experiment does not use MoE, EP, quantization, training, or early exit.
Each destination holds a complete weight replica.
All decode tokens execute their complete fixed depth on the decode worker.
A migrates prefill once. It is not loop-partitioned decode PP.

## Huginn

Upstream rejects PD and staged prefill for boundary-layer models.
The adapter adds boundary KV transfer, initialization RNG transfer, and staged coda cache completion.
It also supplies one initialization generator per prompt row.
The initial failures remain in the raw logs.

The official EOS and initialization sequence differ from upstream RLT defaults.
The explicit reference-compatibility flag preserves the official B1 initialization across token chunks.
The experiment must qualify that protocol independently.
It must not replace the original raw results silently.

Huginn's checkpoint context limit is 4096 positions.
A 4096-token prompt plus further decode exceeds that limit.
Such a condition is unsupported. The experiment does not increase the model limit.

## Dory

Dory is absent from the pinned upstream model and configuration registries.
The minimal adapter preserves all alternating sublayers and the two original RoPE theta values.
It uses exact sliding attention with a 1024-token window where the checkpoint requires it.
It retains input reinjection and nGPT normalization.

Adapter v1 used separate projection GEMMs and separate BF16 rotary products.
The official vLLM code packs QKV and gate/up projections.
It also uses a native rotary operator without an intermediate BF16 product rounding.
Adapter v2 reuses that rotary operator and packs the unchanged BF16 projection weights.
The experiment records additional memory for the derived weight packs.
These generic execution changes apply to every v2 baseline and candidate.
They are not A/B/C gains.

The official model plugin needs compatibility wrappers under the installed Transformers version.
Those wrappers preserve the checkpoint's nested attention RoPE map.
They exclude MLP-only sublayers from RoPE standardization.
The official Dory model source remains unchanged in `sources/`.

## Cache layouts

The primary RLT configuration uses independent per-loop storage, named `last_exited` by RLT.
With fixed full depth, it performs no skipped-depth repair.
For Dory, this corresponds to the official `per_loop` mode.

Dory `last_loop` overwrites current-token KV between loops.
An intermediate loop's current-token KV is therefore not an immutable final cache fragment.
B cannot export that fragment as final state.
The experiment does not treat a cache-mode change as an exact optimization.

## Remaining limits

Native reference parity, RLT parity, and downstream accuracy are separate checks.
A matching RLT output does not establish official paper accuracy.
A completed small probe does not qualify high-concurrency cancellation or preemption.
The initial A adapter accepts one whole prompt chunk and rejects larger unfinished chunks.
It uses a conservative destination initialization barrier.
