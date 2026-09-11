# Experiment log

Date: 2026-09-11 (Asia/Seoul).

1. Audited physical GPUs 6/7 and NV18 connectivity; stopped only the verified
   user-authorized burn on those GPUs.
2. Verified Naive TP1/DP2/EP2 ownership, communication manager, fused expert,
   rank agreement, and profiler collectives.
3. Ran three clean fixed-work restart sweeps for Naive over active labels
   2--256 and batch factorization 1/2/4.
4. Captured cache-off gen=64 batch=1 trajectory: unresolved count changes but
   model M remains 128.
5. Built DeepEP in an isolated environment and verified the DeepEP HT modular
   path and real intra-node kernels.
6. Tried DeepEP LL; stopped at the pinned modular BF16 hidden-state dtype
   assertion. Did not bypass it.
7. Audited AGRS and PPLX availability. AGRS is not exposed by pinned vLLM;
   PPLX is absent. No environment failure is reported as a method failure.
8. Ran three fixed-work restarts for DeepEP HT with identical inputs/routes.
9. Ran randomized clean gen=64 requests, then repeated with five warmup
   requests: three independent restarts/backend.
10. Captured prefix-cache gen=128 and gen=256 controls. Natural EOS truncated
    the initial controls, so a fixed-length gen=256 run disabled EOS and early
    stop to obtain four complete block trajectories.
11. Captured matched Naive and DeepEP HT gen=256 fixed trajectories and
    computed per-forward zero-cost oracle.
12. Added batch=4 cache-off trajectory. Masked positions range 256--1, but
    every forward remains M=512.
13. Ended all task measurement jobs and started the authorized burn on physical
    GPUs 6/7 only.
14. Generated tables/plots, ran the full test suite, completed prior-art audit,
    and issued EP2 NO-GO.

Instrumentation note: clean request runs install no stage/route hooks. Traced
runs add gate calculations, CUDA events, and synchronization; their rough tax
is 77.3%, so only clean request runs support request-latency claims.
