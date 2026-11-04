# Output-paced probe recovery control

## Reason for this control

The independent replacement start failed one common-window Decode gate.
The run was `recovery_huginn_allphase_slots8_audit_replacement_joint_s4_profile512`.
Its common-window ITL p99 was 194.877074808 ms.
The frozen limit was 192.821253818 ms.
The test therefore failed. The limit remains unchanged.

The same run measured P kernel union of 95.5062476%.
Finite service proximity was 108.9289739%.
The common-window D TPOT p95 was 122.686734 ms.
Minimum coordinator-visible D credit was 7,648 blocks.
Twelve admitted local probes completed with exact outputs and full recurrent work.
These successes do not override the ITL failure.

The first earlier confirmation run overlapped a repository audit.
That run is excluded from clean confirmation.
The raw data and exclusion record remain available.
Two other earlier starts passed, but they do not replace a required third independent pass.

## Intervention and hypothesis

The new control uses the existing staged Decode scheduler.
It limits a local probe to one Prefill stage per observed background output step.
A recurrent stage executes at most one complete recurrent loop.
The scheduler retains the exact hidden state and valid KV between stages.
The request still executes all 32 loops.

The hypothesis is that smaller scheduling units reduce interference with background Decode.
This control changes staged execution and probe pacing together.
It cannot identify their separate contributions.
It applies only to independent local probes.
It does not qualify migrated Token Split or Loop Split requests.

Only one local probe can be in flight.
The native KV admission check remains active.
Credits do not accumulate across earlier output steps.
The policy does not use future tokens, future arrivals, or future SLO outcomes.

## Memory failure and controlled retry

The first state test used 28,288 D KV blocks.
It failed with CUDA OOM during the local Prefill warm-up.
The failed allocation requested 2 MiB.
The device reported approximately 2.5 MiB free and 79.18 GiB of process memory.
This was an allocation failure, not evidence of an output mismatch.

The next state test uses 26,752 D KV blocks.
The reduction releases 4,152,360,960 bytes, or 3.8671875 GiB, for staged workspace.
Both pool sizes can reserve six complete 2048-input/16-output requests.
Each complete reservation requires 4,128 blocks.
The comparison does not change BF16 precision, model weights, or recurrent depth.
The smaller pool requires new correctness and performance qualification.

## Required validation

1. Compare the N8 output sequence, valid prompt KV, hidden state, and recurrent work with the native reference.
2. Check the local 16-token Prefill state against its native reference.
3. Repeat the previously failed arrival seed first.
4. Pair each clean N512 run with its own profiled N512 run.
5. Require the common-window and whole-run Decode gates.
6. Require all three independent starts before method expansion.

The new independent-start order is 4, 2, and 3.
The controller records every failure and stops expansion when a required gate fails.
The terminal result is in `state/paced_recovery_v5.json`.
This document describes the intervention. It does not assert a terminal recovery pass.

## Measurement limits

P kernel union is not SM occupancy or peak-FLOP utilization.
The service reference is a measured finite service rate, not a theoretical capacity limit.
A completed local probe is not offloaded serving work.
The frozen ITL p99 condition is not a maximum-ITL guarantee.
Whole-run maximum ITL remains a separate reported metric.
The accepted P-only capture cannot supply a D kernel timeline.
No missing D device activity is inferred from host events.
