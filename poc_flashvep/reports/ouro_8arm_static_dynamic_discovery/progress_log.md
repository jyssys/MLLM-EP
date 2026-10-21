# Progress log

## 2026-10-21

The study froze 1,819 question records and selected 64 GSM8K plus 50 MATH-500 pilot questions.
It preserved the authoritative spec and previous source hashes before new GPU execution.
It collected five-engine numerical traces at P128, P256, P512, and P2048.
The P512 traces include 32 aligned output positions and two distinct synthetic inputs.
The study collected 90 local FP32 attention/projection references.
Those references include the historical failed P512 input and steps 18, 19, 26, and 27.

Native TP1 and TP4 completed the 114-question actual generation pilot.
A corrected native TP1 run also completed 114 questions with valid token-publication timing.
Five CPU contract tests passed. They include exact policy comparison against the pinned CDB source.
Four-request Static PP probes passed output, depth, and all-slot bitwise KV comparison against the matched Single runtime.
The TP4 Graph probe completed on all four physical GPUs and exited normally after the teardown correction.

The custom Static quality campaign now runs before the Dynamic stage.
The full 1,319/500 evaluations and Dynamic quality evaluations have not started at this checkpoint.
This checkpoint is not a final experiment verdict.

## Resumed checkpoint and scope reduction

The user approved approximately 400-500 questions per setting.
The study froze 400 questions per arm: 300 GSM8K and 100 MATH.
It preserves all original pilot questions and cohorts. It samples the remaining IDs without answer, depth, or timing information.
Nine CPU contract tests passed after this amendment. They include sample counts and cohort-domain integrity.
The full 1,819-question evaluation is USER_REDUCED_NOT_RUN, not an unfinished mandatory target.

The Static custom campaign continues without interruption.
The continuation controller includes the 286-question extension for each valid arm.
It also includes same-input CDB/prototype Dynamic probes and matched scheduling performance controls.
No Dynamic accuracy or full performance verdict is available at this checkpoint.

Four PP worker PIDs appear on each of the four GPUs because the workers import peer CUDA IPC storage.
The device monitor therefore shows 16 process-device entries, not 16 separate model workers.
Each GPU has one main worker allocation and three smaller peer contexts.
Those peer contexts remain part of the prototype's reported memory footprint.
