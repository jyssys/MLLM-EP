# Tied-Unrolled contract

The decisive control changes execution organization. It does not change weights, effective depth, or model mathematics.
It is not a separate pretrained AR checkpoint.

## Native execution

The loaded Ouro model contains 48 physical decoder layers.
Each pass applies the same module objects in the same order. R1, R2, and R4 execute 48, 96, and 192 logical layers.
The native model resets the residual state at each pass boundary. It applies the native final normalization after every pass.
The cache index is `pass * 48 + physical_layer`.

## Explicit execution tape

`tied_tape.py` generates a straight-line forward function before model construction.
Each logical invocation explicitly names its physical layer and pass. No outer recurrent Python loop remains in that generated function.
The function preserves embedding, residual resets, per-pass normalization, positions, RoPE, and return state.
It uses the same loaded modules. It creates no parameters and copies no weight tensors.

`ownership.json` records module identities and actual QKV storage pointers for the execution tape.
Repeated calls to one physical layer must retain the same storage pointer within that process.
Native and tied processes have different virtual addresses. Equality across processes means matching loaded weights, not matching addresses.
Every logical layer retains a distinct native KV slot.

The public worker extension installs this override only inside its experiment process.
The installed vLLM source and checkpoint files remain unchanged.

## Frozen correctness gate

The tolerance policy freezes before candidate timing: bitwise equality at the same TP and R.
At 1K and 16K, each arm captures four complete next-token logit vectors and four final-layer hidden vectors.
It also samples actual written KV positions in every logical cache slot and every rank.
The sampled positions are 0, 1, P−1, P, and P+2. Native block tables identify the written physical pages.
The probe samples up to two local heads and coordinates from both K and V.

This is broad KV sampling, not a hash of every byte in the KV pool.
The full logits and hidden vectors provide additional functional checks.
Teacher-forced output equality alone is not proof because the hook sets those output IDs.
The separate natural lane checks unforced output behavior.

A mismatch stops the affected tied arm before its timing collection. The protocol does not relax tolerance after observation.
Cross-TP differences use a separate descriptive audit. BF16 reduction order can change logits across TP degrees.

All primary selected probes passed before timing. A later retained-graph natural test found a TP4 output mismatch.
That failure remains preserved. Short-probe success cannot establish universal output parity.
The ordinary-graph follow-up records selected same-prefix logits. It cannot erase a failed broader correctness check.
No positive causal or method-speedup claim can rely on a control that fails the frozen numerical policy.

## Schedule interpretation

The native forward already passes through compilation and CUDA Graph capture.
A fixed Python loop can therefore produce an explicit GPU operation tape before replay.
Changing Python syntax does not necessarily change runtime scheduling.

The comparison records actual nonempty forwards, graph replays, cache slots, physical kernel order, and collective order.
The main policy destroys the raw CUDA graph after instantiation. It cannot report raw graph-node counts directly.
A separate retained-graph diagnostic can inspect nodes without replacing the main timing evidence.
Equal kernel tapes and replay boundaries support a matched schedule. They do not prove every hidden dependency or rank wait is identical.

If the explicit source and native source produce the same captured execution, tied unrolling is a falsification control.
It is not a new accelerated graph scheduler.
