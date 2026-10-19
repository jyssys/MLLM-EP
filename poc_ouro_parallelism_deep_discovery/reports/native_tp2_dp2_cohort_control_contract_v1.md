# Native TP2×DP2 cohort control v1

Status: `CPU_TESTED_GPU_NOT_RUN`. This is a conventional baseline, not a candidate or a protected CDB factorial run.

## Smallest extension and frozen work

Two persistent native TP2 engines serve static subsets of each cohort. Group 0 uses physical GPUs 4,5. Group 1 uses 6,7. Each engine retains native DP1 and PP1. The parent supplies external two-replica coordination. No model operator changes.

New files are the [control](../scripts/native_tp2_dp2_cohort_control.py), [worker](../scripts/native_tp2_dp2_cohort_worker.py), [two-replica helpers](../tp2_dp2_cohort.py), [protocol](../artifacts/native_tp2_dp2_cohort_protocol_v1.json), and [CPU tests](../tests/test_native_tp2_dp2_cohort_cpu.py). Existing TP and DP4 files remain unchanged.

The protocol pins `native_cohort_protocol_v2.json` with SHA256 `240eb229b37734a532c38c7e08d008a3dc2cabb98ec3c7be7d0f64d46b7d42f9`. It retains both Base checkpoints, BF16, fixed R4, C=1/2/4/8, N=1024/4096, and 128 outputs. Every rank reserves 52 GiB for KV. Native asynchronous scheduling and CUDA Graphs remain active. Chunked prefill, prefix caching, expert parallelism, and speculation remain disabled.

Before worker startup, `freeze_plan` checks exact equality of the TP1 and TP2 source input lists. It also checks exact engine-template equality except for TP size. It loads the TP2 template, then applies the same v2 limits and a group-local cache root. The CPU freeze check passes for both models. Frozen source token 378 is non-EOS. Native `allowed_token_ids`, greedy sampling, and `ignore_eos=True` retain controlled work.

Each model job contains three 16-output warm-ups per case, five clean repeats, and a separate trace lane. Engines remain loaded throughout that job. They do not persist across model-size changes. Engine startup, compilation, ownership checks, and warm-ups remain outside clean timing.

| Submitted C | Group 0 slots | Group 1 slots |
|---:|---|---|
| 1 | 0 | idle |
| 2 | 0 | 1 |
| 4 | 0,2 | 1,3 |
| 8 | 0,2,4,6 | 1,3,5,7 |

Both engines remain reserved at C1. The idle group executes no `generate` call. This control therefore reserves four GPUs without claiming that all four accelerate one request. Two prompt patterns repeat. Static parity placement groups their even/odd slots together. It does not establish optimal placement or independent natural populations.

## Arrival and timing contract

The parent reuses `collect_messages` and `serve_worker`. Both frontends must send matching `READY` messages before release. The timer starts before the first sequential `RELEASE` send. It ends after both matching `DONE` receipts. There is no per-token barrier between engines.

The interval includes release skew, local native submission/execution, output assertions, serialization, completion transport, and controller work. Prompt/parameter preparation and readiness waits occur before it. Result-file writes occur after it. A release is not an atomic simultaneous native arrival. Each engine still admits and schedules its own requests.

| Recorded quantity | Boundary | Interpretation |
|---|---|---|
| Parent makespan | First release send through both completion receipts | Primary observed closed-cohort wall time |
| Local generate wall | One active frontend's `LLM.generate` call | Local cohort API interval, not the parent makespan |
| Local release/start/end | Same-host `perf_counter_ns` timestamps | Consistency checks only. No extra synchronization |
| Native request metrics | Native frontend/core events | Request-specific diagnostics with their original clock definitions |
| Schedule trace | Separate post-clean lane | Excluded from headline timing |

The CPU clock reports `clock_gettime(CLOCK_MONOTONIC)`. Aggregate checks require release receipts inside the parent interval. Active API start/end must follow release and precede parent completion. Their difference must equal local wall. Idle endpoints must both be `None`. These checks do not reconstruct global wall from worker times.

Native `arrival_time` uses a frontend wall clock. Native queued, scheduled, first-token, and last-token timestamps use core monotonic clocks. Do not subtract the wall-clock arrival from those monotonic timestamps. The installed definitions are [RequestStateStats:215–235](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/metrics/stats.py:215). `LLM.generate` returns the cohort together. It does not provide separate host return times for individual requests.

Output TPS is `C*128 / observed_parent_seconds`. Every record must retain the expected slot, source-request ID, 128 output tokens, and `finish_reason='length'`. The worker also checks exact prompt and constant output IDs. Those assertions are not numerical parity or task-quality validation.

## TP workers, ownership, and teardown

One outer guard owns physical GPUs 4/5/6/7. The parent starts two frontend subprocesses without a new session. Each receives its exact two-GPU mask and private cache roots before importing vLLM. Native multiprocessing uses `spawn`. Native DP remains one inside each engine, with separate group environment settings. The parent removes inherited external rank/master selectors.

`inspect_ownership` must return two distinct ranks and PIDs per group. Rank-to-UUID mapping must match the ordered mask, not only the same UUID set. Each rank must report TP world two, eight local Q/KV heads, head dimension 128, and `4*L` cache slots. Physical layers remain 24 or 48.

The installed executor starts `WorkerProc` children through multiprocessing. The engine also uses an EngineCore process. The lease check therefore walks `/proc` PPid chains to each actual frontend PID. It does not assume TP ranks are direct children. Every checked process/ancestor must retain the current UID, owned PGID, and owned SID. Rank/frontend environments must retain the outer guard's output identity.

The check reads only process metadata. It sends no signal and requires no permission change. A real CPU frontend→intermediate→rank chain passes. A wrong ancestor fails. Actual TP process trees remain a GPU-startup obligation.

Normal workers shut down their native engine before `STOPPED`. The parent reuses the existing cleanup function for its explicitly created frontend PIDs. Forced or nonzero cleanup makes the job fail. The existing outer guard retains responsibility for owned descendant GPU contexts and session cleanup. This extension introduces no broad process target or cleanup permission.

## Which existing controls compare

DP4 has the closest timing contract: parent release through all completions, with persistent engines and the same frozen work. It uses four frontends rather than two. Their different controller/release costs stay charged. A residual against the longest local wall includes glue and skew, not pure IPC.

Native TP4 has the same four-GPU resource budget and frozen work. Its saved metric is one API call, not this parent boundary. Keep that difference explicit. No subtraction converts them into a matched unobserved boundary.

Native TP2 reserves only two GPUs. Its earlier pair is 4,6, not either new pair. It is a useful component/resource comparison, not the same four-GPU placement. A pair-specific control would be needed to isolate pair topology or concurrent-engine interference.

Do not compare earlier changing-teacher timings as a speedup. Natural quality, adaptive CDB, protected-factorial conclusions, steady-state serving, and candidate gains remain out of scope. Existing DP4 analysis must not silently classify this new TP2/DP2 schema as TP1/DP4.

## CPU validation and later run

Twenty new tests pass with both the default and dedicated campaign interpreters. The combined new and retained DP4 suites pass 34 tests. Syntax checks pass. The [validation record](../artifacts/native_tp2_dp2_cohort_cpu_validation_v1.json) preserves implementation hashes, commands, and actual times.

The CPU tests do not start native engines or execute CUDA. The coordinator must review the source before a guarded run. Both runner and guard success remain required before analysis.

Example command for that later review, not an executed command:

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python -m poc_ouro_parallelism_deep_discovery.scripts.gpu_session \
  --gpus 4,5,6,7 --output outputs/native_tp2_dp2_cohort_base14_v1 --timeout 5400 -- \
  /home/esjung/.venvs/ouro-deep-discovery/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.native_tp2_dp2_cohort_control --size 1.4B
```

Run from `/home/esjung/MLLM-EP-ouro-deep-discovery`. Use size `2.6B` and a new `base26` output directory for the second model. No engine, GPU command, environment update, or protected experiment runs during implementation.
