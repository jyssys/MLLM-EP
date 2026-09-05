# Experiment log

1. Created branch `flashvep/fixed-shape-tail-root-cause`; restricted all new runs to `CUDA_VISIBLE_DEVICES=1,2,3,4`. Killed only stale processes owned by this user on GPUs 1--4.
2. Read/implemented `fixed_shape_tail_root_cause_specs.md`; ranked measurement artifact, pre-MoE backlog, DeepEP dispatch dependency, expert/workspace, combine, and global interference hypotheses.
3. Baseline online serving: concurrency 8, waves 10, 3 warmups, max batched tokens 4096. 80,256 invocation rows; M=1 decode n=71,424 with dispatch-only 1,944 ms maximum; prefill M=284 n=384.
4. Added same-device stage wrappers for layout/dispatch/expert/combine and stable route identifiers. Python compilation passed.
5. Diagnostic synchronization: same workload family with `FLASHVEP_SYNC_BEFORE_MOE=1`. Decode >10 ms frequency fell from 17/71,424 to 6/44,352 and >20 ms events from 10 to 0; large-prefill behavior is shape-dependent.
6. Fixed-route isolated replay: exact hidden input/route, layer 45/rank 0, 20 warmups + 100 iterations; no giant tail.
7. Controlled exact-request DeepEP replay: 100 iterations × 4 ranks; dispatch/expert/combine distributions stable and no giant tail.
8. Fixed-slot multimodal serving control: exact M=284 was not produced by the scheduler, but M=137/411/1/2/2048 traces retained as fixed-shape controls.
9. Nsight Systems child trace: `trace-fork-before-exec=true`, `wait=all`, CUDA/OSRT trace, no CPU sampling, one measured serving wave. `.nsys-rep` and SQLite succeeded; actual child kernels and DeepEP/NCCL families are present.
10. Root cause decision: dispatch/communication-stream backlog with implicit DeepEP event/peer synchronization; intervention meets the pre-registered >=30% extreme-tail-frequency reduction criterion. No production method implemented.
