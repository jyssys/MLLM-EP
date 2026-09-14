# RefineEP design disposition

## Frozen contract considered

The candidate contract was H100, single-node NVLink/NVSwitch, EP4, hidden
4096, top-k8, 256 experts/64 per rank, BF16, forward-only, and observed
M=1--1024. The proposed first implementation would have kept the existing
fused expert GEMM and replaced dispatch/combine only.

## Mechanism screen

| mechanism | measured support | disposition |
|---|---|---|
| fixed-capacity buffers | cached normal handle saves ~40--46 us/wave | insufficient; still loses to LL in 23/25 real cases |
| specialized top-k8/EP4 layout | normal layout ~32 us; LL already ~9 us event-side prep | little request mass remains |
| NVLink-only transport | LL already uses the intranode fallback | not a new uncovered path |
| medium/small-M mapping | LL wins every real and controlled case | target regime already occupied |
| persistent refinement service | no O3 headroom; likely SM contention and strong prior-art collision | not implemented |

There is no defensible v0 mechanism whose expected gain reaches the 12% gate.
Accordingly, no `.cu` file, PTX, persistent service, or production selector was
created. This is a gated research decision, not an environment blockage.

## What a future reopening would require

Reopen only if a materially different environment invalidates the envelope:
for example, a compacted EP8/RDMA regime, a second model with much larger
high-frequency payloads, or a demonstrated defect in the LL correctness/path.
Merely implementing the same fixed-buffer or small-M specialization more
aggressively is not enough because O1 already prices all control at zero.
