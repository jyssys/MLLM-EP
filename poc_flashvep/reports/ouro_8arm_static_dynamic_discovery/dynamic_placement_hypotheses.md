# Prospective Dynamic placement hypotheses

These hypotheses precede the Dynamic task evaluation. They are not measured performance results.

## Current-row cache repair

CDB `last_exited` copies the current token's exit-loop KV row into every deeper logical slot.
Each Ouro loop has 24 layers, 16 KV heads, and head dimension 128.
One BF16 K/V row across those layers occupies `24 * 2 * 16 * 128 * 2 = 196,608` bytes.

Loop PP stores different loop slots on different GPUs.
An R2 exit therefore requires two remote current-row copies, totaling 393,216 logical bytes.
An R3 exit requires one copy, totaling 196,608 logical bytes.
This accounting excludes transport protocol overhead. It does not measure communication latency.

Layer PP stores every depth slot for one physical layer on the same GPU.
Its last-exited repair can remain local. The runtime must still synchronize safe publication before the next token.
TP4 also repairs each rank's local head shard. It must coordinate exit decisions across its collective sequence.

## Tests

1. Check current-row equality after each official CDB smoke copy.
2. Compare all 96 slots between the matched Single, Layer PP, and Loop PP probes.
3. Count actual remote KV-copy calls and logical tensor bytes in live Dynamic execution.
4. Include copies and acknowledgements in Dynamic serving time.
5. Compare live Dynamic timing with a separate frozen-depth replay.

Do not treat skipped loop computation as free overall speedup.
Do not remove the required cache-repair dependency from a latency explanation.
