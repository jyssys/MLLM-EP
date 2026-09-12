# Experiment log

## 2026-09-12 12:56 KST — environment audit

- Verified the physical UUID mapping and NV18 connectivity of GPUs 0–3.
- Confirmed that the only 0–3 workload was the repository-owned burn started by
  the prior task. Terminated that process tree before measurement. Processes on
  GPUs 4–7 were not touched.
- Read the complete 788-line working specification.

## 2026-09-12 12:57–13:10 — EP substrate validation

- Added EP1 support to the validated EP2/EP4 reference runner.
- Direct GPU conversion of the 30B EP1 model OOMed because fused-layout
  construction transiently duplicated expert weights. This is an environment
  conversion failure, not a method result.
- Corrected EP1 by converting experts to fused layout on CPU before a single GPU
  transfer. EP1 smoke succeeded at about 61.1 GB allocated HBM.
- EP2 and EP4 layer equivalence passed; EP4 ownership was `[0,31]`, `[32,63]`,
  `[64,95]`, `[96,127]`.

## 2026-09-12 13:10–13:21 — measurement-trust correction

- Rejected an adaptive-threshold partial sweep because topology-dependent FP16
  differences altered NFE and output.
- Restored reverse-A2A rows to original branch order and selected fixed-NFE
  `threshold=1.0` for the fair topology experiment.
- EP2/EP4 one-request corrected smoke: same 43 forwards and identical output.
- Started randomized, three-restart matched Stage-0 clean sweep.

## 2026-09-12 13:25–14:00 — second measurement-trust correction

- The first threshold-1.0 cycle still allowed EOS early stop. Although its
  first three topology runs happened to execute 968 forwards each, a second
  Single run executed fewer forwards. This exposed the latent confound.
- Stopped the task-owned sweep, retained it under
  `results/stage0_early_stop_unmatched_threshold1/`, and added an explicit
  no-early-stop control to the released generation harness call.
- Corrected EP1 to bypass world-size-one A2A/broadcast calls. It still uses the
  identical vLLM fused expert primitive and full 128-expert checkpoint.
- Restarted the final randomized three-restart sweep with a fixed output
  budget. No result from either rejected sweep is used as performance evidence.

## 2026-09-12 14:00–14:48 — final Stage-0 clean sweep

- Completed nine independent model starts in balanced topology order, with
  eight matched GSM8K/HumanEval requests per start.
- Every run executed 1,166 model forwards and 14,487,552 expert assignments.
- Median aggregate wall was 161.511 s (Single), 184.931 s (EP2), and
  192.096 s (EP4). Thus the exact reference EP2/EP4 paths were 14.5%/18.9%
  slower than Single under this small-row diffusion workload.
- Median bounded task quality was 62.5% for all three topologies. Full strings
  were not bitwise deterministic, so performance evidence is conditioned on
  equal work, layer-level numerical equivalence, and equal bounded quality.
- Measured one 9.437 MB FP16 expert copy across GPU0→GPU1 at 0.0619 ms median;
  eight experts (75.5 MB) took 0.2311 ms. These costs feed the exact-replica
  oracle and are not themselves a request-speedup claim.

## 2026-09-12 14:47–15:17 — temporal and candidate oracles

- Captured exact route/state/contribution histories on EP4, plus EP1/EP2
  controls. Observer tax was measured separately and raw stage times were not
  mixed with clean wall without scaling.
- Ran low-capture CUDA-event timing, rank-local top-k 1/2/4 views followed by
  exact EP4, and identical-input EP1/2/4 M=1–256 replays.
- All candidates failed the 5% feasible E2E gate; no large prototype was
  implemented. GPU0–3 were returned to the user-requested idle burn while CPU
  analysis continued.

## Verification

- `py_compile` passes for the extended runner and all new analysis scripts.
- Every shell launcher passes `bash -n`; all derived JSON parses; `git diff
  --check` passes.
- The pre-existing `poc_team_positive_ep2/tests/test_analysis.py` cannot run in
  either worktree because its ignored 2026-09-11 raw fixture directory is no
  longer present. All seven failures are `FileNotFoundError`, not assertion or
  code failures. This PoC instead has fresh layer numerical validation and
  end-to-end true-EP executions as its runtime verification.
