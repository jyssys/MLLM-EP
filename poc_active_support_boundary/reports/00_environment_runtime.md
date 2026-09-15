# Environment, model and runtime truth

- Project HEAD before this PoC: `9ef1eab2a9587f7a47cf62fc9ddb6da9d3714ec0`;
  new worktree branch `flashvep/active-support-mlp-boundary-poc`.
- Checkpoint: `/home/esjung/models/LLaDA2.0-flash-744c3f8`,
  downloaded `config.json` SHA-256
  `ac35e9dc8313f49b2600da2a8b3ec31c53c01d22a5943ec26c8e21f8e208ab07`
  checked again on 2026-09-15. BF16, hidden 4096, 256 routed experts,
  top-8, intermediate 1024, 32 layers and one shared expert.
- dInfer commit `9132ce9b2580ac2e64bcaaf975b863a5005c7739`;
  previous validated DeepEP source revision
  `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`.
- Owner-local replay executable imports PyTorch 2.8.0+cu128 and
  vLLM 0.10.2 from `/home/esjung/anaconda3/envs/dinfer-ep-poc/`.
  Its chosen `cuda:0` is physical GPU 4. Nsight Systems 2024.6.2 was
  available; Nsight Compute/device HBM traffic counters were not.

The existing **validated** full-model baseline is dense TP4, routed EP4,
DP1, 64 complete expert weights/rank, replicated shared expert, DeepEP
Normal dispatch to owning rank and reverse combine. Remote assignments
and ownership were verified in the frozen route corpus
[`poc_refinegemm/reports/00_environment_and_baseline.md`](../../poc_refinegemm/reports/00_environment_and_baseline.md).
The new benchmark only runs owner-local expert execution; it does **not**
claim to fresh-profile 4-rank dispatch/combine or to integrate grouped_mm.

| Physical GPU | UUID | mapping | peer (previous audit) |
|---:|---|---|---|
| 4 | `GPU-6076e2f2-5b63-3761-5586-56ceb7df8139` | cuda:0 | NV18 |
| 5 | `GPU-a1a1cfcf-93a1-3544-9a5e-e58144b68730` | cuda:1 | NV18 |
| 6 | `GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28` | cuda:2 | NV18 |
| 7 | `GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b` | cuda:3 | NV18 |

Before measurement, the previously started burn was verified as the
repository-local `/home/esjung/vllm-ep/utilize.py` owned by `esjung`,
supervisor PID 1806749 and GPU4--7-only descendants; it was stopped by
SIGTERM. Other-user processes were never terminated. Repeated prelaunch
`nvidia-smi -i 4,5,6,7` confirmed 81,090 MiB free per GPU and no compute
process. No GPU 0--3 launches were made. After all measurements, the
user-requested burn started again using the explicitly hardcoded 4--7
`run_utilize.sh` with supervisor PID 1901572 (verify again in final handoff).

Strongest frozen clean request baseline: submitted pool 32, mini32,
block32, generation32, threshold0.9/config42. The three **previous** clean
restarts yield GSM8K BCT 6.075 s (NFE66, score 5/32) and HumanEval BCT
7.343 s (NFE86, score 6/32). These are reused matched same-day controls,
not newly timed full-model runs. Prior observer-heavy critical-rank expert
time divided by clean BCT was 29.47% / 29.55%, an attribution **upper bound**,
not a clean additive per-component critical-path removal proof. Refer to
[`poc_refinegemm/E2E_RESULTS.csv`](../../poc_refinegemm/E2E_RESULTS.csv) and
[`poc_refinegemm/OPERATOR_BREAKDOWN.csv`](../../poc_refinegemm/OPERATOR_BREAKDOWN.csv).

The installed vLLM BF16 fused expert warns that there is no tuned
`E=64,N=1024,H100` config. Consequently the previous operator study found
`torch._grouped_mm` faster for all 102 comparable real route/restart cases.
That is the **strongest tested expert operator** for the new-oracle
comparison; it is not claimed to be already integrated as the EP4 engine.
