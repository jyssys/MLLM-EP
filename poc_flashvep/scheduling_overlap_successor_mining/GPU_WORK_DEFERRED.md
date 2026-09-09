# GPU work deferred — scheduling/overlap successor mining

## Latest disposition — 2026-09-09 bounded screening closed

This is the **Layered Prefill / FastPP / NanoFlow** checklist, not SERE/Libra/MoDES.
Resumed native/control collection completed at about 14:58 KST. No queued
research engine remains. Status: **PARTIAL_ENVIRONMENT_BLOCKED**; not completed
faithful native-MLLM successor validation, and not universal NO_GO.

Completed: Layered native/21 graph engines/three mechanism diagnostics; FastPP
dense18/native-MoE9/static-partition6/full-warmup15; NanoFlow native/HF/Nsight
plus prefill18/decode36/long-decode18; common real-image vLLM observer controls
including 120 actual-token-exact visual/text pairs.

| Remaining work | Why not counted complete |
|---|---|
| Layered native VL + arbitrary group request oracle | Native port/direct request semantics still required |
| FastPP native VL/PP×EP + exact joint chunk/partition oracle | Native Qwen3 is PP4 without EP; stage proxy failed to transfer to E2E |
| NanoFlow optimized MoE search | Real expert profiling/database/lifecycle hooks missing; manual plans are not searched optimum |
| NanoFlow dynamic VL/warmed continuous-serving plan portfolio | Fixed-cohort/HF diagnostics insufficient; decode envelope null |
| Kimi / 5 paired successor runs / minimum successor | NOT_RUN: no material non-trivial Qwen gate passed |

These are future baseline/validation tasks, **not background research jobs**.
The user-authorized burn is separate, restricted to 4–7 and excluded from all
research accounting. See `COMMON/FINAL_GPU_STATUS.md` for verified final state.
The dated estimates below are preserved history, not currently queued jobs.

## 2026-09-09 resumption

### Active continuation — 14:00 KST

Completed: Layered 12-engine full-warmup cap/chunk screen; NanoFlow resource
numerical/HF diagnostic and 18-engine pure-prefill portfolio; large matched
Qwen3-VL clean/observer six-engine screen. The old stopped Layered build wrapper
has been identity-checked and terminated without resuming its CUDA-capable tail.
Compiled artifacts and logs are preserved.

Running: FastPP full-workload-warmup five-existing-knob screen on physical 4–7.
Queued after it: NanoFlow B4/B64/B256 decode four-plan portfolio, three restarts
per cell. Analysis starts only after each complete manifest, and remains CPU-only.
There is no overlapping burn or second GPU experiment. Below are historical
snapshots, not instructions to rerun completed work. Kimi/prototype still gated.

### Active continuation — 13:05 KST

The user reconfirmed continuing work. Layered full-workload-warmup official
cap4/12/24 versus chunk2048 is running on physical 4/5 (12 restarts planned).
NanoFlow's same-prefix prefill/resource correctness check is queued **after**
that supervisor exits and rechecks all four permitted GPUs before launch.
No other GPU experiment or burn runs concurrently.

FastPP's six static-partition engines are complete: the measured-cost partition
loses 11.98–14.91% paired request E2E. Pending GPU work now consists of NanoFlow
numerical/HF and plan repetitions, token-matched larger real-image vLLM control,
and FastPP full-warmup/static-chunk control. These remain the current three-system
study, not SERE/Libra/MoDES.

### Live progress — 12:17 KST (supersedes the old deferred estimates below)

- Native Layered graph screen (9 engines) and eager mechanism (3) complete.
- Native FastPP Qwen3 policy screen (9 engines) and exact four-rank sparse
  mechanism join complete. Existing static-partition 3-pair control is running,
  with shared full-workload warmup and conservative equal KV capacity. An opt-in
  KV layer-count compatibility correction is documented and CPU-regression-tested.
- NanoFlow independent HF same-prefix and Nsight overlap checks complete.
  Pure-prefill FFN plan first-use had an adapter initialization failure; stopped,
  source-local cause identified, CPU-tested fix prepared. Fresh smoke is next,
  followed by a bounded clean plan portfolio if correctness passes.
- Six fresh Qwen3-VL clean/lite-observer engines completed. 1,248 measured
  requests total; median observer overhead still 2.5–6.5% by family. Do not use
  instrumented times as clean native-system E2E. No native VL port is claimed.
- Additional Layered official-cap/full-warmup control remains pending. Final
  ranking still waits for comparable screening; Kimi/method gates not passed.

GPU experiments currently use physical 4–7 only. No burn is co-running with
research workers. The pause marker below is historical and was archived.

The user explicitly reauthorized physical GPUs **4,5,6,7 only**. The release
checkpoint below is historical. Resume the same three-system study, not the
older SERE/Libra/MoDES study. Idle/final burn is reauthorized only on 4–7 and
must be stopped during measurements. FlashAttention CPU build completed during
the pause; remaining native Layered installation and deferred correctness tests
are first. No winner has been selected.

### Archived release checkpoint — superseded by the resumption above

Historical status: **USER_REQUESTED_GPU_PAUSE**, not research completion or environment failure.
Physical GPUs 4–7 were released at 2026-09-08 ~19:47 KST. No burn is running.
Resume only on a new explicit user request. No GPU 0–3 use is authorized.

## Preserved state

- Branch: `flashvep/scheduling-overlap-successor-mining`.
- Results: `poc_flashvep/deepep_revalidation/results/scheduling_overlap_successor_mining_20260908_161510/`.
- `GPU_RELEASE_STATE.json` verifies no compute processes on 4–7 at release.
- `COMMON/GPU_PAUSED_BY_USER.md` blocks `run_scoped.sh`.
- All analysis now uses `COMMON/run_cpu_only.sh` (`CUDA_VISIBLE_DEVICES=""`).
- No automatic queued GPU run or idle/final burn may restart.

## Resume priority — all three before winner selection

| Priority | Work | Why needed | Estimated GPU wall after setup |
|---|---|---|---|
| 1 | Layered Prefill native Qwen3 TP2 chunked/layered smoke, correctness, graph sanity | No native GPU evidence yet; equal screening incomplete | 0.5–1.5 h |
| 2 | NanoFlow same-prefix logit checks and same-plan restart control | 82-token smoke passes, but diverse free continuations differ | 10–30 min |
| 3 | FastPP Qwen3 PP4 uninstrumented PP/greedy/ALP restart comparison | Native MoE works; only one instrumented workload run exists | 0.5–1.5 h |
| 4 | Lower-overhead Qwen3-VL EP trace/control repetitions | Full observer adds observed 16–20% request latency in one restart pair | 15–45 min |
| 5 | Layered group-count and measured-cost partition controls on identical arrivals | Stage-cost proxy alone cannot establish request benefit | 1–3 h |
| 6 | Native NanoFlow actual stream overlap / plan search compatibility / portfolio | Manual fixed plan is not the official searched optimum | Bounded 1–3 h |
| 7 | Material candidate's exact chunk/stage/partition oracles | Existing-policy dense envelope is small; need nontrivial residual | Signal-dependent |
| 8 | Kimi and 5 paired restarts / one minimum successor | Only if all screening gates justify a finalist | Not authorized to start prematurely |

These are rough experiment times, not a guaranteed completion estimate. A
configuration or compile failure remains PORT/ENVIRONMENT failure, not method
failure. Do not revive SERE/Libra/MoDES or the earlier closed EP directions.

## Progress at 2026-09-09 11:00 KST

- Native Layered dependencies installed; eager pair and graph-enabled 9-engine
  chunk512/group4/group16 screen completed (864 measured requests).
- NanoFlow B4 same-prefix four-engine numerical comparison completed: 128/128
  argmax agreement cross-plan and within-plan. B16 extension currently running.
- FastPP Qwen3 three-policy/three-restart run is next. Dense evidence is unchanged.
- CUDA/NVTX-only Nsight wrapper is prepared for NanoFlow native stream overlap;
  it does not profile unrelated processes and its latency is excluded from clean results.
- `COMMON/GPU_PAUSED_BY_USER.md` was moved to `GPU_PAUSE_HISTORY_20260908.md`;
  the pause guard is no longer active. Old build wrapper must not be resumed.

## Previously prepared items (historical notes; see progress above)

- `NANOFLOW/run_numerical_screen.py`: two fresh restarts each of graph/split2,
  teacher-forced identical prefixes with vocabulary-logit capture. CPU parent
  control tests pass. Runtime capture seam has **not** been GPU-tested.
- `NANOFLOW/run_cohort_screen.py`: full 3-restart multi-plan grid is unrun. Do
  not launch until correctness and cold-versus-steady scope are resolved.
- FastPP native PP4 is **not EP4**. vLLM EP traces are separate labelled transfer
  diagnostics, not a faithful native FastPP/Layered/NanoFlow MLLM port.
- Current FastPP every-16-local-call sampling gives few exact four-stage joins.
  Shared request/position-based sampling must be validated before a cross-rank
  critical-path oracle. Per-rank stage distributions remain useful.

## CPU build safety

The Layered build wrapper (PID 2579722 at pause) was SIGSTOP-paused while its
already-running pip/Ninja child continues **CPU compilation only**. This prevents
the wrapper from automatically reaching a CUDA-capable import after the build.
Do not blindly SIGCONT it. Check process identity and build log, then finish any
remaining installation explicitly with GPUs hidden for CPU build work. GPU
correctness tests require renewed user permission. Keep compiled intermediates;
do not delete editable-build temporary directories or reset the worktree.

## Required interpretation on resume

- FastPP dense ALP gains/losses are original-method directions, not successor gain.
- Across four tested existing policies, best-static versus per-regime envelope is
  only 0.25–0.64% in the present dense screen; new work must beat the best existing
  control, not a selectively weak ALP/PP-only baseline.
- Layer cost partition improvements are NOT direct request-E2E improvements.
- Detailed vLLM timing is diagnostic; clean request data is the latency reference.
- All three are still unranked. There is no approved winner or STRONG_GO.
