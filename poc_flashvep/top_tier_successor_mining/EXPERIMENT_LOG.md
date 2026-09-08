# Top-tier successor mining — live log

## 2026-09-08 11:03 KST — deferred GPU work reauthorized

- Read the complete working contract and GPU deferred list. Resume uses physical
  4,5,6,7 only; other GPUs have unrelated jobs and remain untouched.
- Found run_utilize.sh still hard-coded to1–4. Added an explicit environment
  override preserving its default, then launched it with4–7 while auditing.
  User-requested burn supervisor1914781 was stopped before calibration restart.
- Centralized current GPU-policy validation and physical-rank metadata. Old saved
  results retain original mappings; new runs record4–7 explicitly.

## 2026-09-07 13:59 KST — start, audit only

- Contract: `../reports/top_tier_moe_successor_mining_spec.md`, read in full.
- New branch: `flashvep/top-tier-successor-mining`; unrelated dirty files preserved.
- Results: `../deepep_revalidation/results/top_tier_successor_mining_20260907_135950/`.
- Physical GPU restriction: 1,2,3,4 only. No experiment has started yet.
- User's contract prohibits burn but final instruction explicitly requests idle burn.
  Following that final instruction, inherited, separately identified burn workers
  remain running during CPU audits. Burn is never research time and will be stopped
  before measurement, with a thermal/common-state warmup control.
- No paper winner selected. All three must receive serious screening first.

## Source provenance and environment

- SERE official code: `JL-Cheng/SERE`, 8512f39b108cd1fa04e9261a5b6846fb173989e4.
- MoDES official code: `ModelTC/MoDES`, 933b3ccac3c23e01c763771f1a7d60c4e7ed13a4.
- Libra's ICLR proceedings identify `SNU-ARC/Libra`, but GitHub web, git and API
  return not-found. Final proceedings paper downloaded; supplemental/public
  organization inventory audit underway. Missing source is not method failure.
- Validated runtime imports: torch 2.11.0+cu129, transformers 5.14.1, vLLM 0.20.0,
  DeepEP from `/home/esjung/.venvs/flashvep-deepep-v020/`.
- Separate CPU audit venv created; baseline torch/vLLM environment not upgraded.

## Initial source findings (hypotheses, not measured failures)

- SERE's official router uses the batch union of top-S primaries, retains weights,
  and replaces secondary IDs subject to a similarity cutoff. The published CUDA
  path does not explicitly merge duplicate token/expert IDs. Must inspect actual
  expert execution before assuming assignment work decreases.
- MoDES Qwen code explicitly calls skipping simulated; original top-k IDs still
  reach expert execution while selected weights become zero. Thus timing the
  reference Python implementation is not a reproduction of the paper's fast path.
- MoDES layer-ablation path appears to return the normalized input, not zero, for
  skipped branches. Need test official behavior versus paper equation; do not
  attribute a possible implementation discrepancy to the method.
- Libra uses the next block's gate on current hidden states, not previous top-k
  persistence. An actual lookahead-gate probe is required.
# 2026-09-07 14:38 KST — calibration and first mathematical transfer probes

- Primitive parity: 133 official SERE/MoDES tests PASS, including unchanged
  official SERE CUDA extension, zero-threshold no-op and ablation semantics.
- SERE Qwen-VL FineWeb calibration completed: 400 sequences × 128 tokens,
  51,200 token inputs at every layer. Streamed FP32 Frobenius accumulation has
  ~0.2% sampled difference from the official BF16 norm (recorded, not hidden).
- MoDES 64-request smoke calibration completed across physical GPU 1–3. This
  pilot used a short-answer instruction. Matching `paper_zero` control completed
  on GPU 4. These pilots are NOT the definitive official calibration protocol.
- Protocol audit caught the official calibration using the raw question rather
  than the short-answer instruction. The partially started 1,024 run was stopped
  (PIDs 923582/923754/923923); its partial raw rows are retained and excluded from
  primary calibration. A distinct `modes_calib1024_official` run is now active,
  with raw-question prompt, official answer logit mask, official-input ablation,
  1,605,632 max pixels, and image-disjoint calibration/held-out sets.
- Libra's genuine next-gate predictor on the 64 pilot requests: token-weighted
  recall Vision 0.79083, Text 0.85518. This alone is NOT a failure: both overlap
  the published range, and material lost performance is not yet measured.
- Appendix B/C count/placement diagnostic passes 120 conservation/capacity
  tests. This is a labeled paper port, not the unavailable official Cython
  system. Underspecified ownership/ties remain explicit and cannot support a
  claim about original runtime overhead.
- Started paired SERE/vanilla 64-request held-out GQA evaluation on GPU 4;
  policy order randomized within each request. HF runtime is quality evidence,
  not faithful EP serving-speed evidence.
- Started 5-second telemetry restricted to physical GPUs 1/2/3/4. Idle burn is
  off during all measurements. Historical burn time is excluded from research.

## 2026-09-07 ~16:00–16:15 KST — supplied Libra source replaces fallback

- User supplied README/REPRODUCE and Libra/internal/Lina SGLang diffs. Both
  documents read; hashes preserved. Three isolated source checkouts use the
  documented SGLang commit 023288645b80fb41b3eed55fd413dd69a7904593.
- Paper-only Libra GPU probe was prepared but NOT executed. It is paused in
  favor of supplied baseline source; no evidence relies on it.
- Actual Cython planner built unchanged; 48 implementation invariants PASS.
  Fresh-route analysis rerun with the real planner rather than paper pseudocode.
- Separate original-version Python3.10/SGLang0.4.10/torch2.7.1 environment built
  in minutes, not by changing the validated vLLM installation. A native reduced
  model functional sanity is being attempted; timing under concurrent quality
  calibration is explicitly excluded from performance claims.
- MoDES official full-grid calibration continues. No all-three ranking yet.

## 2026-09-07 16:20 KST — native Libra functional path executes

- Supplement native four-GPU, four-layer dummy functional check completed on
  physical1–4. Eight measured Libra rank comparisons are logit-exact to vanilla.
  Dummy model and concurrent calibration exclude performance/quality claims.
- Required Python3.10 datasets/PyArrow compatibility fix is isolated; no author
  algorithm, assertion or existing vLLM environment was modified.
- Actual supplied Cython planning on fresh 96-request Qwen-VL traces shows
  median lookahead recall .826 overall, .821 Vision, .885 Text. The corresponding
  perfect-prediction local-fraction improvement is only .78125 percentage points
  median. This is a planner statistic, not a measured E2E bound or failure verdict.
- Prior-art attack expanded to full AnyExperts and MACS methods/evaluation.
  Semantic importance, OCR sensitivity, modality-adaptive capacity and local
  semantic rerouting are already explicitly discussed; none alone is novelty.

## 2026-09-07 17:28 KST — active research resumed after status-only turns

- Re-read all 1,728 lines of the working contract and the negative-space map.
  Status answers did not terminate the MoDES process, but they did end active
  agent turns. Do not conflate continuing GPU computation with autonomous
  follow-up analysis. User now explicitly requested continuation of the full task.
- MoDES full frontier passed 277 evaluations and completed the first target:
  target skip .70, actual .713284, answer-position KL .0110761. The .85 target
  remains running. These are calibration statistics, not held-out quality.
- All-three ranking remains deferred. Remaining gates are clean direct-request
  SERE timing/quality, full native Libra text/VL parity and transfer timing, and
  held-out MoDES results using the completed official calibration frontier.
- Use only physical GPUs 1–4. No new GPU process will contend with the full
  calibration for performance measurements. Burn remains off.

## 2026-09-07 17:37 KST — user requests GPU release; CPU-only continuation

- Stopped own waiting handoff PID1143543 FIRST, preventing it from launching
  captures when calibration exited. Then SIGTERM to own torchrun PID1021871;
  its workers1022115–1022118 exited. Own telemetry936515/936664 also stopped.
- Read-only verification: all four permitted GPUs show 0% utilization and
  0 MiB used; no listed own worker remains. No foreign process was signalled.
- MoDES importance calibration is complete. Full threshold search is PARTIAL:
  target70 finished, target85 unfinished; last persisted unique evaluation301.
  Existing evaluations and first-target frontier are retained for CPU analysis
  and a later explicitly authorized resume. No completed marker manufactured.
- Native full-layer Libra, VL tensor capture, updated SERE EP comparisons and
  Kimi validation have NOT run. CPU bridge unit tests passed six dtype/shape
  combinations, but are not full-model numerical evidence.
- GPU_EXECUTION_POLICY.json disables new GPU runs, auto-resume and burn.
  Latest resource-release instruction supersedes the earlier idle/final burn
  instruction. Continue audits/analysis on CPU; do not infer a method NO_GO
  from unavailable hardware or promote an unmeasured E2E benefit.

## CPU-only checkpoint after explicit user confirmation

- Consolidated 19 quality evidence rows with image-cluster uncertainty; clean
  E2E fields remain empty rather than replaced by quality-replica wall time.
- Matched actual Libra planner outputs in 1,081 quartet/layer/budget conditions;
  lower Vision recall is not treated as a proportional runtime loss.
- Compared SERE calibration arithmetic on 6,336 sampled route conditions.
  These are CPU calculations on prefill snapshots, not new decode GPU evidence.
- Preserved MoDES301-point partial frontier; checked462 observed monotonicity
  comparisons. Largest skip reversal .903pp occurs far from the low-KL frontier.
- Six CPU Libra bridge boundary checks pass. Full-model parity remains pending.
- Generated and inspected three plots and wrote the CPU-only interim main report.
  All-three final ranking and successor verdict remain deferred.
- No GPU or burn process started. No automatic resume is scheduled.
# 2026-09-08 live resumption measurement controls

- Official1024/grid100 MoDES frontier resumed on physical4–7. A cached point
  reproduces exactly in KL and skip fraction before the unfinished sweep resumes.
  No held-out threshold tuning and no partial-best-as-completed label.
- CPU request-timestamp unit test passes. Primary resumed EP request latency is
  submission to engine token readiness; frontend receipt and polling delay are
  retained separately. Source confirms all used absolute timestamps are host
  monotonic clocks, not cross-device CUDA timestamps.
- Unique image UUIDs prevent A/B multimodal encoder-cache reuse; pixels and
  prompts remain identical. Instrumented parity is separated from clean timing.
- Confirmatory quality launcher preserves original128-request selection and
  complete B16 cohort membership when distributing jobs over4replicas.
- CPU oracle-injection test passes without skipping predictor GEMM. Actual
  native48-layer Libra parity remains required before running this diagnostic.
- New ACE primary paper audit is recorded in PRIOR_ART_REFRESH_20260908.md;
  generic calibration-free contribution estimation is not a free novelty gap.

## 2026-09-08 12:26 KST — native VL port first-divergence controls

- Completed official MoDES1024/grid100 frontier:436 distinct evaluations. The
  resumed cached-point KL/skip match exactly. Confirmatory128+128 quality and
  B16 SERE controls completed; see RESUME_CHECKPOINT_1H.md and raw summaries.
- Native VL checkpoint matrix layout differs from the native Linear loader.
  A real-checkpoint CPU regression reproduced the original load exception;
  explicit transpose/split translation fixes it, max expert arithmetic error
  6.56e-7 in the FP32 diagnostic. Author Libra algorithm remains unchanged.
- First native VL forward matches HF greedy tokens, while repeated forwards
  degrade. Same-device selected hidden captures show layer0 input exact on the
  first forward, not a missing vision/DeepStack input. A separate red-capable
  CPU regression demonstrates input capture mutation through the native residual
  alias. Fresh-per-forward input clone restores the stock embedding-buffer
  lifetime contract. Live randomized repeat validation is running.
- All pre-lifetime-fix VL timings are PORT_DIAGNOSTIC_INVALID, not method
  performance or a claim that Libra fails on MLLM. Diagnostic tensor capture
  forwards are also excluded from speed results.
- Fresh DeepEP identity-expert compatibility on physical4–7 completed288
  rank-local tests covering duplicate expert IDs, -1 sentinels and empty sources
  across two CUDA streams. This is a transport correctness check, not speedup.

## 2026-09-08 12:34 KST — cold-cache and request-timing sanity PASS

- Source/live control catches deprecated vLLM raw-string preprocessing dropping
  explicit image UUIDs. Interrupted `ep_clean_chart_b1_run0_20260908` is preserved
  as INVALIDATED, never included in clean statistics. Only its own validated
  worker tree was signalled; stuck collective workers required SIGKILL after
  TERM grace. No dataset or unrelated job was modified.
- Current renderer input path preserves UUIDs:60/60 hashes unique, MM cache
  hit0.0%; same eight measured requests pass full greedy vanilla/no-op and
  zero-weight/fast-sentinel parity. Actual four-rank DeepEPHT/Triton path proven.
- Core token timestamps and frontend receipt are separately checked. First-token
  time uses recorded first_token_ts even if polling coalesces outputs; exact ITL
  distribution excludes coalesced observations instead of inventing zero ITLs.
- Six clean natural-EOS conditions (GQA/ChartQA × DP-local B1/B4/B16),128 requests
  each,three cohort repetitions,all six official policies,started sequentially.
  Each engine has per-policy warmup and randomized policy order; independent
  process repetitions will be added only after inspecting material effects.
# 2026-09-08 14:19 KST — length and port controls

Fixed32-output controls completed at offered B1 and B16. Cached-reference
SERE remains about15–16% slower and MoDES about7–8% slower in these bounded
length-controlled streams. These numbers are not benchmark-quality scores:
forcing continuation after EOS invalidates short-answer quality interpretation.
The next no-op decisions retain unchanged routing/output while paying the
ported decision path, separating port/execution cost from treatment effects.

CPU regression found the old bridge test still demanded input-storage aliasing,
which contradicts the already validated immutable-capture fix. It failed at that
assertion; updated it to require exact values and distinct storage. All six CPU
bridge boundary cases plus input-lifetime/oracle/timing regressions now pass.
An attempted unittest discovery found zero tests (these are standalone scripts)
and is not counted as a validation pass. No GPU was used by these CPU tests.
# Final GPU resumption closeout — 2026-09-08

All twelve clean primary engines completed;two further independent MoDES GQA-B16
engines challenge the only initially positive point. Across-engine median gain
is−.909% for70 and+.889% for85. First-run7.587% is not a robust headline.
Native Libra eight-group prediction gain is.146%;actual Nsight overlap exists.
Large M8192/source oracle-interleaved output corruption is INVALIDATED. The same
size without oracle hooks passes80/80 first tokens and has−5.401% paired prefill
reduction versus native vanilla. It is not full request E2E.

Final ranking after all-three screening:SERE23,MoDES20,Libra18(outof60).
SERE's existing S4 repairs100%/88.9% of the observed B1 quality loss,so the
deep-dive successor fails the trivial-fix gate. Kimi and a new prototype NOT_RUN.
Status:FOUND_INCREMENTAL_ONLY,not a refutation of original-paper performance.
Recorded resumed GPU-resident hours14.360;historical+resumed25.309,including loads
and CPU control. Not CUDA-active hours. Burn counted zero. Measurement workers
are stopped;final user-requested4–7 utilize launch is separate from experiments.

Historical experiment log follows.
