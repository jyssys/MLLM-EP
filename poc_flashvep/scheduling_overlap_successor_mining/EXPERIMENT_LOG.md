# Scheduling/overlap successor mining — live experiment log

Study start: 2026-09-08 16:15:10 KST (07:15:10 UTC).
Branch: `flashvep/scheduling-overlap-successor-mining`.
Base commit: `03dc3ebfaa1bd10da8fa52e5ed863b92306a8fce`.

## 2026-09-08 — contract and safety

- Read all 880 lines of the user-provided scheduling/overlap spec.
- User corrected physical GPU allocation to **4,5,6,7** and subsequently explicitly authorized idle/final burn on those GPUs only.
- No experimental use of GPUs 0–3. No termination of other users' processes.
- Existing prior-study burn verified: parent 2407937, workers 2408259–2408262, explicit `--gpus 4,5,6,7`. This is not research time; stop our process tree before any experiment.
- `/home/esjung/vllm-ep/run_utilize.sh` defaults to 1–4: any future authorized invocation MUST set both `VLLM_UTILIZE_GPUS=4,5,6,7` and `CUDA_VISIBLE_DEVICES=4,5,6,7` explicitly.
- Preserve unrelated dirty/untracked historical work. Separate reference checkouts and environments; no in-place upgrades of validated runtimes.
- No new GPU measurements yet. Initial work is paper/source/history auditing of all three candidates.

## Execution order

1. Audit paper/code and original evaluation assumptions of all three systems.
2. Official baseline smoke/correctness/mechanism, with bounded isolated setup.
3. Faithful MoE/MLLM transfer and direct request-level controls; distinguish unsupported port from method failure.
4. Equal-milestone ranking only after all three screenings, then at most one successor deep dive.

## 16:30–16:50 KST — isolated builds and first smoke

- All three paper technical bodies and source mechanisms audited; audit files
  saved. FastPP dependency import passes. Qwen2.5-32B and native NanoFlow
  Qwen1.5-MoE-A2.7B official snapshots downloaded and pinned.
- NanoFlow default branch is not the full research surface: official dev-h100
  adds Qwen MoE/EP paths. Prefer that supported code before any full VL port.
- Layered editable build initially failed because bundled setuptools 59.6 lacks
  PEP 660. Isolated setuptools 75.8.2/CMake 3.31.10 fixes configuration; official
  FA2/FA3 CUDA build is progressing. No baseline runtime upgraded.
- NanoFlow local liburing 2.6 built without global installation/sysctl changes.
  CMake incorrectly chose global Python 3.14 despite venv PATH; stopped that
  compile and explicitly pinned Python 3.10 executable/root. Dependency API and
  small Gurobi feasibility sanity pass; restricted license remains a full-search
  feasibility risk.
- 16:41: stopped only our burn parent 2407937 and its registered worker groups;
  verified 4–7 returned to 4 MiB each before FastPP launch. Burn excluded from
  research GPU accounting.
- First FastPP PP4 run stalled during initialization, before model loading.
  Second instrumented restart reproduced it. Py-spy attach denied by host ptrace
  policy; opt-in child faulthandler succeeds and localizes the stall to installed
  vLLM PyNccl communicator warmup stream synchronization. NCCL debug reports
  NET/IB completion warnings. Investigate transport/config, not method failure.

## 16:50–17:10 KST — correctness before benchmarking

- FastPP PP4 starts after changing only `NCCL_IB_DISABLE=1`. Four HTTP smoke
  requests finish, but this is **not correctness PASS**: greedy arithmetic and
  capital answers continue into `Human:` text instead of HF EOS. The sharp
  `FASTPP/check_short_answers.py` regression fails on both saved cases. No
  latency from these requests is accepted as baseline performance evidence.
- Independent same-checkpoint HF BF16 reference produces `42<|im_end|>` and
  `Paris<|im_end|>`. SGLang tokenizer IDs exactly match HF, including EOS 151645.
  Next diagnostic distinguishes prefill teacher forcing from cached decode.
- FastPP optional PP `return_logprob` fails because the PP worker returns a
  placeholder logits output; excluded from subsequent tests. Opt-in sampling
  diagnostics instead capture first logits/IDs and are disabled for measurement.
- NanoFlow native extension import and official CPU weight packing pass. Packing
  completes in 116.23 s with no CUDA context, so this is not GPU experiment time.
  Official H100 Qwen smoke exposes stale KV-range and flag APIs. Isolated
  compatibility changes are documented; the original tracebacks are retained.
  Runtime now reaches the executor and FlashInfer JIT. Setup/JIT residence is
  not meaningful measured GPU execution time.
- Current GPUs 4–7 contain only our NanoFlow smoke workers. GPUs 0–3 and their
  jobs remain untouched. No burn runs concurrently with validation.

## 17:10–17:30 KST — EOS attribution and live regression

- FastPP teacher-forced and free-decode raw special tokens both produce the
  correct EOS. The actual fault is `Req.check_finished` indexing the first
  output token (`output_ids[0]`) after new tokens were appended at the end.
  An actual-official-class regression fails before and passes after changing
  just that index to `[-1]`. `ignore_eos=True` remains unchanged.
- Removed opt-in sampling instrumentation; a fresh engine reproduces HF short
  answers `42` and `Paris` with correct termination (2/2), and all four smoke
  requests complete. This is baseline compatibility, not research novelty.
- NanoFlow GPU workers were temporarily SIGSTOP-paused while their CPU-only
  FlashInfer JIT continued. FastPP used a reduced 0.60 memory fraction only for
  correctness. These shared-residency diagnostic latencies are excluded from
  performance comparisons. Stopped our FastPP server 2710682 and all descendants
  after the regression; resumed only our NanoFlow workers 2686974–2686977.
- Official FlashInfer 0.3.1 compiles 153 fused-MoE objects on first use; this is
  environment setup, not measured GPU experiment time. Layered FA3 compilation
  is also progressing; both remain within the per-environment time bound.
- Current literature attack identifies DynaFlow and TokenWeave as serious
  collisions for generic adaptive overlap claims. Their technical bodies were
  read; generic workload-dependent plan selection is not sufficient novelty.

## 17:30–17:45 KST — controls and native-path preparation

- Created deterministic real FineWeb-content traces: 48-request heterogeneous
  steady/bursty with exactly identical request contents and output lengths,
  plus a 512-request short high-concurrency case to reach the official batch
  rebalancer's 128-request scale. Arrivals are explicitly designed controls,
  not falsely labeled the paper's Azure trace. Source and trace hashes saved.
- CPU stream-client tests pass for SGLang cumulative SSE and Layered token
  deltas, including fragmented network frames and duplicate final messages.
  SGLang's official `rid` field now carries the client request identity into
  internal batches. Client timestamps remain client-observed, not internal
  token-ready timestamps.
- Added opt-in FastPP same-device CUDA stage/layer hooks with request IDs and
  local invocation IDs. They do not synchronize; primary E2E comparisons leave
  the hook disabled. Mechanism data will be collected separately and validated.
- Independent FP16 Qwen1.5-MoE and BF16 Qwen3-30B references completed. The former
  uses the official NanoFlow plain prompt; the latter uses the checkpoint's
  explicit `enable_thinking=False` chat template. No serving speedup is inferred
  from these HF references. NanoFlow workers were paused during these short
  GPU correctness jobs and resumed; CPU JIT continued.
- Native H100 NanoFlow Qwen MoE factory has no default search-plan file, and
  BasePipeline's optional split path retains an old flag name. Default execution
  success must not be counted as nano-batching/overlap activation. A faithful
  plan-path diagnostic follows output correctness; unsupported plan adaptation
  will not be labeled a failure of NanoFlow's research method.

## 17:57 KST — GPU-scope safety incident and correction

- A one-off command intended to inspect NanoFlow's plan API on CPU imported
  `BasePipeline`/`FusedMoE` without an explicit CUDA mask. Contrary to expectation,
  import-time extension discovery initialized CUDA (`CUDA_INITIALIZED True`).
  The parent environment had no CUDA mask, so context creation outside 4–7,
  particularly default GPU 0, cannot be ruled out. No model or benchmark was
  run by this command. Exact context duration/device was not captured.
- The command (PID 2766574) already exited before the post-check. No associated
  process remained on any GPU; other users' processes were not terminated.
  The diagnostic is excluded from performance and research GPU-time evidence.
  User was informed immediately; do not claim perfect GPU-scope compliance.
- Every subsequent Python invocation, including intended CPU diagnostics, must
  explicitly use CUDA_VISIBLE_DEVICES=4,5,6,7 and TORCH_CUDA_ARCH_LIST=9.0.
  `COMMON/run_scoped.sh` provides the direct guarded execution wrapper.
  Source inspection through rg/sed is preferred when CUDA initialization is not
  needed. Never assume that an import is CPU-only.

## 18:00–18:16 KST — first measured online pilot and restart screen

- NanoFlow official H100 Qwen1.5-MoE EP4 smoke finished: all four prompts and
  82 generated tokens exactly match the FP16 HF reference. Initial run included
  compilation and diagnostic pauses; its ~50-minute residence is NOT counted
  as 50 minutes of meaningful GPU measurement. Warm-cache reruns have explicit
  per-step timing and initialization exclusions.
- FastPP PP-only pilot collected 48 heterogeneous steady, 512 short concurrent,
  and 192 officially sampled Azure requests. All completed without client stream
  coalescing. Pilot is calibration, not a randomized treatment comparison.
- Started randomized 3-restart blocks for PP-only/greedy/ALP/ALP+BR, with equal
  warmup and identical per-block workload order. Each block includes steady,
  bursty, short high-concurrency and the official Azure-arrival control.
  Cross-config output checks remain mandatory before performance acceptance.
- Created opt-in bounded NanoFlow fixed-plan adapter using the official native
  splitter, executor and numerical operations. Not yet GPU-tested. No claim
  that a hand-written diagnostic plan is the paper's searched optimum. Its
  first probe rejects changing shapes after fixed decode-cohort activation.

## 18:30–18:46 KST — repeated comparisons, no early successor claim

- FastPP first block shows workload-dependent original-method direction, but
  PP-only varies substantially across independent restarts. The second PP-only
  run is slower across several traces despite normal sampled GPU clocks. Do not
  interpret a slow baseline restart as ALP successor headroom. Complete the
  randomized screen, then use stage/batch observations and the official transport
  option as bounded controls.
- Four short-answer probes agree across completed configurations. Natural long
  continuation hashes differ, while requested output lengths agree. Exact text
  equality is not by itself a numerical correctness test; targeted divergence
  checks are required before accepting a quality-preserving performance claim.
- Optional ALP observation now has a CPU regression test proving that logged
  pre-update prediction, original update, and subsequent prediction retain the
  original values. Two instrumentation control-flow tests pass; no CUDA was
  initialized in these guarded CPU tests.
- Layered Prefill pinned native dependency continues compiling (210/399 at
  approximately 18:43); no numerical or scheduling change has been made to
  accelerate the build. Its native TP2 baseline is still unmeasured.
- Preparing NanoFlow warm-cache eager/graph/split correctness diagnostics.
  Graph capture and splitter activation must be separately verified. A default
  eager smoke is not evidence for the paper's searched overlap configuration.

## 18:53–19:16 KST — graph/split correctness and dense transport control

- FastPP dense randomized screen completed all 12 engine restarts and 9600
  measured requests. ALP median across three paired restart blocks improves
  steady mean request E2E by 24.25%, but worsens bursty by 10.52% and short-load
  by 5.82%; Azure improves 7.51%. One PP-only restart is much slower, so the
  mean across restarts is misleading. These are original-method directions,
  not novel successor gains. Cross-config long-output validation remains open.
- Same-policy long-continuation hashes also vary substantially across restarts
  (for example PP-only steady exact fraction 0.333 between blocks 0/1), so
  strategy-specific numerical failure has not been established by text hashes.
- Started three paired PP-only/ALP restarts with NCCL P2P enabled as a trivial
  transport control. IB remains disabled; no other runtime or request change.
- NanoFlow warm-cache graph/split activation now has five token-exact smoke
  conditions. The two-part sigmoid-copy bug was reproduced with an AST seam,
  fixed with one constructor argument, and verified in fresh native workers.
  Its failed output/runtime is excluded from all performance conclusions.
- Prepared real-content native NanoFlow cohorts, including equal total prefill
  volume with different request counts/context lengths. CPU parent-loop tests
  pass. Request interval includes target plan setup/capture. No claim yet of
  arbitrary-shape or continuous-batching support.
- Prepared 128 real-image/text transfer inputs and a clean process-local vLLM
  operation observer. No prior SERE/MoDES/Libra policy code is imported. Native
  baseline and vLLM trace-driven diagnostic evidence remain explicitly separate.

## 19:23–19:35 KST — transport control complete; native MoE transfer

- FastPP P2P-enabled control completed six independent engines / 4800 measured
  requests. Median paired ALP mean-request-E2E reduction: steady +20.63%, bursty
  -14.11%, short -15.23%, Azure +11.77%. Azure has an outlier restart with -49.10%,
  so neither a stable Azure gain nor an MLLM-specific failure is established.
  The workload direction is not removed by simply enabling P2P.
- Native Qwen3-MoE PP4 startup confirms NCCL P2P/IPC in channel-level logs. It is
  TP1/PP4 with all 128 experts per owning layer, NOT EP4. Do not conflate this
  transfer with the separate vLLM TP2/DP2/EP4 diagnostic.
- The first instrumented MoE attempt failed in our observer: official ALP
  initializes chunk entries to zero and its learner is a no-op before startup
  profiling, but the observer prematurely invoked predict(). The real official
  ALPScheduler regression is RED before, GREEN after a positive-baseline guard.
  All three observer tests pass. This is measurement repair, not method failure.
  A fresh corrected run is underway; failed startup contributes no performance.
- NanoFlow real-content graph-versus-split2 pilot completed four fresh engines.
  Cross-plan exact continuations: 14/16 (M-prefill 8192, decode 16), 3/4 (M-prefill
  512, decode 4). Hold these rows out of performance claims until same-prefix
  numerical/restart controls. Official four-prompt HF smoke alone was too narrow.
- Nano pilot request path includes ~0.51-second unsplit versus ~1.93–2.07-second
  split graph capture. Steady decode host medians are ~6.31 vs 8.37 ms (B4),
  ~8.13 vs 10.87 ms (B16). Neither cold-capture overhead nor poor manual plans
  establish failure of the paper's searched optimum. Avoid a 190–205% headline.

## 19:36–19:47 KST — completed transfer capture; GPU return requested

- Corrected FastPP Qwen3-MoE PP4 run completed both 48-request steady/bursty
  traces with short-answer checks passing. Instrumentation source fix remains
  isolated; the failed startup v1 is not included in performance data.
- Clean vLLM Qwen3-VL TP2/DP2/EP4 smoke verified DeepEPHTPrepareAndFinalize,
  TritonExperts, BF16 and DBO off. Full observer and no-observer grids each
  completed 208 measured requests, plus the separate eight-request smoke.
- At approximately 19:45 the user requested GPU return and CPU-only work.
  No subsequent experiment or burn was started. The already-running clean grid
  finished normally, both DP frontends exited 0, and all owned workers shut down.
- GPU_RELEASE_STATE.json confirms no compute apps on 4–7 at ~19:47; 0% utilization
  and ~4 MiB each. Other users' processes were not terminated. User informed.
- Added GPU pause marker and CPU-only launcher. Paused Layered build wrapper
  PID 2579722 before its eventual CUDA-capable import; ongoing child compilation
  is CPU only. Do not automatically resume the wrapper or any GPU job.

## CPU-only interim analysis after GPU return

- All 42,672 captured logical MoE identities join to four ranks, with zero unknown
  request IDs. There are 38,160 measured logical invocations after warmup/dummy
  filtering. Do not add four rank rows to a request's latency.
- Full observer versus clean request E2E differs by 16.37–19.58% across families
  in one paired input grid. This is a measurement confound, not optimization
  headroom. Native three-system transfer and lower-overhead replication remain.
- Measured fixed-cohort prefill-only E2E upper bound obtained by zeroing ALL
  TTFT is 5.95–11.10% by family. This does not prove a continuous-arrival SLO
  bound or settle the native Layered candidate.
- FastPP actual pre-update ALP APE median: EXTEND 3.06% (27 matched samples),
  MIXED 2.69% (86). No median predictor collapse on this native MoE run.
- Dense existing-policy lower envelope adds just 0.25% (request-mix weights) or
  0.64% (equal workload weights) beyond best-static greedy. Untested chunk/
  partition and native MoE regimes are not bounded by that restricted result.
- Interim report and GPU_WORK_DEFERRED.md written. No winner, final research
  status, Kimi experiment or successor prototype is claimed.

## 20:08 KST — CPU-only handoff validation

- Recomputed transfer summaries with explicit fixed-timeline TTFT-zero bound
  fields. Rechecked assignment conservation on all 38,160 measured logical
  invocations and complete four-rank joins; all checks pass without CUDA.
- GPUs 4–7 are now allocated to other users' processes after our earlier release.
  None belongs to this study; none was touched. No study GPU/burn worker remains.
- Layered dependency compilation still runs on CPU. Its parent build wrapper
  remains stopped, so the subsequent CUDA-capable import cannot run automatically.
  Raw data and pending numerical/oracle experiments remain preserved for restart.

## 2026-09-09 10:13–10:18 KST — reauthorization and external occupancy

- Full spec read; user reauthorized physical 4–7. Pause marker archived, history
  retained. Burn defaults to 1–4 in the local script and needs explicit override.
- FlashAttention build completed; remaining native Layered CPU build started
  with GPUs hidden. No CUDA model/import command follows this pip process.
- NanoFlow same-prefix screen stopped at occupancy guard before native workers.
  New kaist3 jobs arrived after the initial empty-device check. Five exact PIDs
  were confirmed scoped to 4–7, but SIGTERM failed with OS PermissionError for
  all of them. Noninteractive sudo requires a password. No job was terminated.
- No fresh GPU measurement or burn was run. This is an external resource blocker,
  not method failure. See COMMON/RESUME_STATUS_20260909.md for precise evidence.

## 2026-09-09 10:27 KST — GPU work actually resumes

- User cleared GPUs 4–7; fresh query confirms all four empty. No other process
  was terminated. Native Layered build/install completed successfully.
- NanoFlow same-prefix graph/split2 two-restart numerical screen started with
  the original cohort, identical teacher-forced continuation and native logits.
  This is correctness diagnosis, not latency evidence. Burn remains off while
  the research workers are running.

## 10:27–11:10 KST — fresh native measurements, no winner yet

- Layered BF16 Qwen3 TP2 eager pair and graph-enabled 3-config × 3-restart
  screen completed. Graph screen has 864 measured requests. Cap4 versus chunk512
  median paired mean-E2E gain: bursty 19.665%, steady 3.471%. Cap16 includes a
  materially slow third restart; no outlier was discarded. Existing cap4 is best
  on both workload medians, so the restricted per-regime E2E envelope adds 0%.
- Native source/log inspection confirms the configured stage count chooses a
  token-dependent table, not a fixed per-request group count. Actual selected
  counts and admission token volumes saved in native_scheduler_choices.json.
  Native max-ITL SLOs expose the known TTFT/TBT tradeoff; mean TPOT is insufficient.
- NanoFlow same-prefix B4 (128 positions) cross-plan and within-plan argmax
  agreement 100%. B16 (512 positions): cross-plan 99.61%/99.41%; same-plan split
  99.80%. Differences are at near-ties; cross-plan mean KL 0.000052/0.000077.
  These four-engine diagnostics per batch size are not performance measurements.
- FastPP native Qwen3 PP4, P2P enabled/IB disabled, now runs three original
  policies × three independent restarts. Same four request streams and warmup.
- Prepared a same-request/position hash sampler for cross-PP stage diagnostics;
  four CPU regression tests pass, fresh GPU join validation still pending.
- Prepared CUDA/NVTX-only NanoFlow Nsight diagnostics; no unrelated-process
  sampling. Profiling runs will be excluded from clean latency evidence.
- GPU accounting now unions intervals per physical device: native Layered TP2
  is not counted as four-GPU compute. Model loading/JIT/burn remain excluded.
- Additional prior-art attack found Bullet's dynamic layer-level phase/resource
  scheduling; generic adaptive NanoFlow plans alone have a direct collision.

## 11:10–12:05 KST — native mechanism and transfer controls

- FastPP native Qwen3 nine-restart screen completed: 7,200 measured requests.
  Existing-policy E2E envelope is 0.587% request-weighted / 2.351% equal-weighted.
  Restart variance remains material; do not headline favorable 40% individual
  comparisons. Full workload warmup control prepared, not yet collected.
- Improved FastPP sparse-layer/all-stage instrumentation joins all 3,950 measured
  identities across four PP ranks without duplicates. 127 full 48-layer profiles.
  Rank 0 is critical in 3,885. Measured-cost partition proxy is not portable-cost
  or request-level evidence. Uneven native PP allocation needs a KV-layer-count
  compatibility check before any static-partition toggle is safe.
- Layered eager diagnostic collected 288 requests, all 48 layer events, sampled
  expert-use histograms and request identity. Sampled histograms conserve routed
  assignments. Weight-use estimates are logical proxies, not measured DRAM bytes.
- Independent HF validation confirms NanoFlow mismatches concentrate at near-ties.
  CUDA/NVTX profiles verify overlap increases while the manual split2 plan is
  slower; no claimed speedup from an overlap-intersection statistic.
- Six fresh Qwen3-VL engines now compare clean/layer-sampled measurement with
  shared request order in each restart block. All GPU work remains serial on
  physical 4–7; native Layered uses 4/5 only. No burn during measurements.

## 12:05–12:40 KST — counterfactuals tested rather than inferred

- Clean/lite Qwen3-VL control completed all six engines and 1,248 measured
  requests. 14,399 sampled measured MoE invocations join all four EP ranks,
  unknown request IDs zero. Median observer overhead remains 2.54–6.54% by
  family. Clean request latency remains the reference.
- FastPP explicit uneven partition had a native KV layer-count incompatibility;
  actual-function CPU regression reproduced it and verified the opt-in repair.
  A first launch used an incorrect local snapshot path and failed at config
  loading, not inference; corrected to the previously verified `ad44e777...` and
  added a prelaunch config-existence guard. Failed launch excluded.
- Native uninstrumented three-pair existing partition control completed:
  8/12/14/14 versus equal12, same KV cap/full-trace warmup/greedy/arrivals.
  Cost-static worsens E2E in all pairs: median -14.91% steady / -11.98% bursty.
  The 23–26% instrumented stage-max proxy is NOT request headroom.
- NanoFlow initial pure-prefill partial plan lacked initialization of unlisted
  attention operators. Own supervisor stopped; native default-tag lifecycle
  initialization added, preserving already initialized operators. CPU regression
  and fresh six-engine smoke completed. Plain versus full-SM split2 token IDs
  match at M4096/M8192. Single-pair E2E is -17.40% / -0.64%; no positive signal.
  Existing 112/16 green-context control is slower but one M8192 next-token
  mismatch remains, so its timing is excluded pending numerical diagnosis.
- New CPU-prepared real four-chart inputs have 1,733–1,909 / 3,785–3,961 prompt
  tokens and 1,700–1,876 vision tokens. Sixteen natural-text matched pairs have
  exactly equal processor token counts. No upsampling or GPU used in preparation.
- Native Layered extra-cap/full-workload-warmup graph screen now runs on 4/5.
  No burn on 6/7 during measurement, to avoid shared-host/fabric interference.
  No Kimi/model optimization or winner promotion yet.

### 2026-09-09 12:57–13:12 KST continuation

- User asked to continue. Verified the ongoing Layered TP2 job uses 4/5 only;
  6/7 are idle to prevent co-running burn interference. Twelve full-warmup
  cap/chunk engines are progressing; no new independent GPU job overlaps them.
- Queued NanoFlow four-worker-group same-prefix M8192 numerical diagnostic
  after the Layered supervisor exits, with an empty-GPU guard before every run.
  Independent HF reference follows; diagnostic times remain excluded.
- Native Layered CPU minimax diagnostic now covers 22 complete 48-layer
  rank/phase/prefill-active profiles. Decode proxy gain is zero; prefill can show
  25–40% group-proxy reduction. No request E2E projection is made.
- Source inspection of native NanoFlow finds last-only means last layer, not
  last token; final vocabulary projection appears prompt-wide. Capture actual
  logits shapes before attributing any overhead. Last-token selection would be
  a trivial compatibility optimization, not a new scheduling successor.
- Large-image vLLM harness preflight found an empty warmup selector for the new
  family names. Extracted actual selector, CPU RED reproduced zero instead of
  two requests per DP, then GREEN after deterministic image-family fallback.
  Legacy single-image selection is unchanged. Matched-request/token identity
  analysis has its own CPU fixture test. No GPU result was fabricated or lost.
- Rechecked native FastPP instrumentation and explicit-partition CPU tests:
  four and three tests pass. An initial test invocation omitted its source-path
  environment variable; corrected invocation passes and is not a runtime failure.
- Latest adversarial literature search continues to find adjacent dynamic
  operation scheduling (DynaFlow/Bullet), and native Layered already adapts
  group count to token length. No novelty claim is made from a static knob sweep.

### 2026-09-09 13:13–13:51 KST — completed controls and remaining serial jobs

- Layered four-cap/chunk full-workload-warmup screen finished all twelve engines.
  Finite additional E2E envelope 0.657%; held-out selection zero in all blocks;
  nine-SLO additional attained-goodput maximum 0.0301%. Slow restarts retained.
- NanoFlow M8192 resource same-prefix native repetitions plus independent HF
  reference completed. One mismatch sits at HF top-two gap zero; it remains
  excluded from performance. Native vocabulary tensor [8192,151936] verifies
  prompt-wide final projection; no unmeasured E2E saving is assigned.
- NanoFlow three-plan pure-prefill portfolio finished 18 engines / 72 requests.
  All twelve paired first-token checks are exact. Plain wins M4096 and M8192;
  finite static-to-regime E2E envelope zero. Scope is not paper-searched optimum.
- Large actual-image Qwen3-VL observer/clean screen finished six engines / 480
  measured requests. All 120 clean visual/text pairs exactly match actual prompt
  tokens. Common-state variation is substantial; stage observer data is not
  used as clean E2E or modality-causal evidence.
- FastPP five-existing-knob full-warmup comparison is now running serially on
  physical 4–7. The next NanoFlow B4/B64/B256 decode portfolio waits for its
  supervisor to exit. No simultaneous GPU experiment or burn is running.
- CPU regression checks pass: FastPP 12, COMMON 5, NanoFlow lifecycle 1 plus
  parent-identity assertions. Missing filename in a source-discovery command
  was corrected; it was not a GPU/test failure.
- Canonical transfer/failure/trivial-fix/oracle/prior-art documents now distinguish
  native reproduction from common VL trace diagnostics for every candidate.
  No candidate yet qualifies for Kimi or successor implementation.

### 2026-09-09 13:52–14:45 KST — final finite-policy controls

- FastPP completed all 15 full-workload-warmed engines / 1,440 measured requests.
  Existing chunk2048/PP-only wins both workloads: additional finite E2E envelope
  zero, all three held-out selections zero. Largest nine-SLO attained-goodput
  addition 6.967%; not capacity. ALP loses all three paired request comparisons.
- NanoFlow B4/B64/B256 × four plans × three restarts is collecting. Initial
  results show substantial first-decode setup cost and nonidentical continuations;
  neither is accepted as a method failure or successor gain. Completed analyses
  will retain strict paired output-equality exclusions.
- A final bounded amortization control is queued behind this screen: same
  B4/B64 real inputs, common output length 96 instead of 16, three native plans
  and three restarts. No wait/setup time is subtracted from E2E. This is a fixed
  cohort diagnostic, not a warmed continuous-serving claim or a new method.
- Small evidence bundle indexing and CPU provenance/tests continue concurrently;
  raw weights, profiler binaries and old experiments are not modified or staged.

### 2026-09-09 14:50–15:03 KST — bounded GPU collection closed

- NanoFlow 36-engine 16-output screen completed: 3,888 requests, only 2/27
  whole-cohort pairs token-exact. Correctness-gated envelope is null, not zero.
- The same-input 96-output control completed 18 engines / 612 requests at
  about 14:58. Only 1/12 pairs exact; envelope again null. First-decode cost
  share shrinks with longer output but is never subtracted from E2E. Neither
  screen is the full asynchronous searched-paper baseline.
- All native/control collection is stopped; no queued research GPU job remains.
  Common request index now 33,788 rows. Conservative recorded live union
  148.99 minutes, 9.155 GPU-hours (2.289 four-GPU-equivalent hours), excluding
  loading/build/JIT/burn and user GPU-return interval.
- Exactly-one screening leader is FastPP, score14/50; Layered12; NanoFlow8.
  No eligible research finalist. Dedicated winner documents explain failed
  causal/headroom/non-triviality gates rather than inventing Kimi/prototype work.
- Overall PARTIAL_ENVIRONMENT_BLOCKED distinguishes successful native probes
  from missing native VL/PP×EP/search semantics. It is not universal NO_GO.
- User-authorized final burn is being separately started on 4–7 after verifying
  no compute processes on those devices. First detached attempt did not remain
  alive and produced no log; foreground persistent-session retry is being
  verified. Burn is never counted as research or run alongside measurements.

### 2026-09-09 15:04 KST — handoff verification

- Persistent burn retry verified: supervisor596994, four explicit 4/5/6/7
  workers; all four printed generation-started and reached 100% GPU utilization.
  Existing script duration is 30h after model load; user-visible final state
  explicitly distinguishes burn from research. No script code was changed.
- CPU handoff validation PASS: three score rows/totals, exactly one screening
  leader, 149 selected evidence files with size/SHA256 validation, 33,788 request
  rows, 966 unique time intervals with only authorized physical GPU IDs.
- Python compilation and existing regression tests pass. Git whitespace check
  treats CSV CRLF as valid and excludes unified-patch context whitespace; no
  data was rewritten just to change line endings. Only this study/spec/report
  and the selected result bundle are staged; old unrelated edits remain intact.
