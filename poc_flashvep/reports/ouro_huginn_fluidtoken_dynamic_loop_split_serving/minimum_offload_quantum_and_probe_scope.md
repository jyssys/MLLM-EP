# Minimum offload quantum and probe scope

## Structural comparison

The earlier D-local probe used 16 prompt tokens and the complete recurrent depth.
It was an independent request, not migrated Serving work.
A successful probe certifies only its tested work and latency contract.
It does not certify every larger continuation.

| Model | Earlier full-depth local probe | Smallest nonzero whole-loop continuation | Fine-Token control |
|---|---:|---:|---:|
| Ouro R4, T4096 | 16 x 4 = 64 recurrent rows | Loop3: 4096 x 1 = 4096 rows | Suffix256: 256 x 4 = 1024 rows |
| Huginn R32, T2048 | 16 x 32 = 512 recurrent rows | Loop31: 2048 x 1 = 2048 rows | Suffix32: 32 x 32 = 1024 rows |

Ouro's minimum whole-loop continuation has 64 times the probe's recurrent rows.
Huginn's minimum whole-loop continuation has four times the probe's recurrent rows.
These ratios exclude Attention context, Prelude, Coda, padding, and state movement.
They are not FLOP ratios or latency predictions.

Huginn also requires valid prompt Coda KV on D after recurrent continuation.
Loop31 adds 4094 Coda block-token rows for T2048.
A 16-token local probe adds 30 such rows before its first output.
The common first-token Coda execution remains separate from this count.

## Decode scheduling contract

The native `SplitPrefillPolicy` extends `RefillPolicy`.
The fixed default permits one Prefill batch before recurrent Decode becomes due.
Wavefront Prefill advances one recurrent depth per scheduling step.
Recurrent Decode also advances one depth per scheduling step.
The scheduler can therefore interleave separate Prefill and Decode submissions.
It does not combine their rows into one shared-core GEMM in these experiments.

One Decode output still requires the complete model depth.
Interleaving one small Prefill stage per Decode depth can accumulate before the next output.
Per-submission duration and request-level continuation duration are different quantities.
Neither quantity alone proves a safe token deadline under concurrent execution.

The online predictor currently uses an isolated acceptance-to-first-output enclosure for D continuation.
That enclosure includes control, scheduling, Coda, and delivery.
It is not exclusive GPU blocking time.
The conservative predictor can reject work that a better interference model might admit.
Its false-rejection rate remains unmeasured.
No future arrivals or actual future output lengths enter the policy.

## Claim boundary

A D-headroom motivation does not guarantee a useful minimum whole-loop migration quantum.
The experiment must test actual migrated requests under the frozen TPOT and ITL limits.
The fine-Token controls prevent an artificial restriction to coarse matched-row Token cuts.
Any observed Loop benefit still requires the logical-layer equivalence limitation.
No Loop-specific advantage follows from the structural counts in this note.
