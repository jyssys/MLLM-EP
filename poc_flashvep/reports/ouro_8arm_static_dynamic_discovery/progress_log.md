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

## 2026-10-21 09:05 UTC continuation review

All four custom Static pilots completed 114 canonical-input questions each.
Layer PP and Loop PP matched the Single output IDs on all 114 questions.
The custom TP task pilot completed with its numerical warning unchanged.
Its GSM8K score was 47/64. The Single score was 46/64.
Both scored 31/50 on the pilot MATH questions.
These pilot results do not prove population-level accuracy equivalence.

The official Dynamic reference completed 114 questions.
The custom Single-Dynamic task pilot is running.
Physical forced-exit and live-gate LD/OD probes matched SD output, depths, and all 96 KV slots bitwise.
The bounded TD output and depth traces matched, but its assembled KV comparison remained numerically unresolved.
The official adapter checked all 269 original model tensors bitwise against the pinned checkpoint.
Optional adapter gates remain unused.

The CPU cache-valid-length regression failed before the correction and passed afterward.
Ten CPU tests passed after that correction.
A later CPU test checks first-difference indexing for free-running output analysis.
No existing report or protected source file changed.

Checkpoint commit `c95fac0ca8efa56b3fa59e8406a456366c0ee80d` reached the dedicated remote branch.
The three commit-pinned primary report URLs returned HTTP 200.
That commit is an in-progress checkpoint, not the final experiment verdict.
The continuation remains active. It includes the 400-question extension and separate timing/profiling lanes.

## 2026-10-21 12:48 UTC checkpoint

All eight custom arms completed the same 114-question natural-generation pilot.
SD, LD, and OD matched all 114 output sequences and exit-depth traces.
The official CDB reference also completed 114 questions, but cross-backend natural output and exit parity did not hold.
The reference comparison and recorded gate margins remain separate from within-custom PP parity.

All eight arms completed the small matched-natural control.
The campaign completed 40 Static and 32 trace-locked replay comparisons.
Each comparison contains five measured repeats after one warmup.
The 400-question extension is now running. No final architecture verdict is issued.

A source audit identified cap-dependent KV mask shapes in the custom runtime.
The additional profiling plan preserves the actual O4096 serving allocation while limiting trace generation to 32 tokens.
This is a separate bounded diagnostic, not an accuracy run or a new inference method.
The original primary and replay allocation defaults remain unchanged.
Fifteen CPU tests passed, including mocked construction of all planned trace commands without GPU use.

## Final recovery and review: 2026-10-22

All eight 400-question custom evaluations and both native 400-question controls completed.
The original profiler failure began at 2026-10-21T19:23:23.914599+00:00.
The native TP1 retry began at 2026-10-22T01:33:54.456426+00:00.
The stopped interval is not experimental inference time.
The repair selected the child Python environment's executable PATH.
The CPU regression failed before that repair and passed afterward.
Recovery reused all successful quality and replay results.
It preserved the failed profile and used a new retry label.

All 18 heavy captures completed. All 18 bounded observer comparisons preserve output IDs.
Heavy wall overhead is 24.9% to 405.0% and remains unqualified for headlines.
The 72 clean replay comparisons each contain five measured repetitions.
The final CPU suite passes 18 tests.
Artifact checks confirm 122 unchanged earlier files and 4,411 canonical-input records.
Report checks match 104 printed benchmark cells to source tables and check 58 local links.
All 18 full SQLite exports have verified lossless archives.

The main-agent review sets NO-GO for a demonstrated optimized-native Loop-PP advantage.
The decision retains the historical strict TP failure and the native/custom CDB behavioral warning.
No new scheduler, RL policy, or architecture was implemented.
Physical GPUs4,5,6,7 returned to the requested burners.
Only the two new study directories enter the final commit.
