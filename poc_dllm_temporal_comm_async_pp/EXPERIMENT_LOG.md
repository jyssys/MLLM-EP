# Experiment log

## 2026-09-13 — initialization

- Read the 572-line working contract in full.
- Created isolated project and dInfer worktrees; the dirty primary checkout is
  not modified.
- Verified the local LLaDA2.0-Flash config: BF16, 32 layers, hidden 4096, 256
  routed experts, top-8, one shared expert, GQA 32/4 heads.
- Reused only the prior validated true-EP4 bridge and diagnostics as substrate.
- Stopped the previous task-owned GPU0--3 burn because this task forbids those
  devices.  The task-owned GPU4--7 burn remains until measurement begins.

## 2026-09-13 — baseline and Track A

- Reproduced the strongest validated dense-TP4/routed-EP4 DeepEP configuration
  for three clean restarts on each bounded task.  Median clean BCT was 5.932 s
  on GSM8K and 7.361 s on HumanEval; NFE was 66 and 86, respectively.
- Captured actual dispatch-input activations at layers 1/8/16/24/31.  The
  observer increased wall time by roughly 2--3x, so its event times are not
  used as request timing evidence.
- Lag-1 remote destination cacheability was high (84.65% GSM8K, 81.03%
  HumanEval), but fresh DeepEP payload sweeps showed a large fixed/startup
  component.  Payload-sensitive request oracles remained below 1%; unfused
  FP8/INT8 codec cost erased even that gain.  Track A therefore hit its kill
  gate before a live protocol or trajectory-quality run.

## 2026-09-13 — Track B

- Captured all-layer timings and lag-1 boundary states after layers
  8/16/24/32.  The ideal no-dependency flow-shop ceiling is large, but is an
  analytical ceiling measured on the existing TP4+EP4 replay, not a live PP
  result.
- Evaluated 18 boundary-staleness policies after one model load on each of
  GSM8K and HumanEval.  These runs are sequential full-trajectory quality
  emulation; their wall times include heavy observer/cache overhead and are not
  performance evidence.
- Two bring-up attempts failed because the first cache design retained full
  wave views and then because a 2048-slot cache could not hold all logical
  request/block/position keys.  Fixed slot buffers with 8192 entries bounded
  HBM at about 75 GiB and allowed both policy sweeps to finish.  These were
  instrumentation failures, not method failures.
- SGLang rank-group audits validated representable PP4 and PP2xEP2 groups.
  Exact BF16 neighboring-stage transport was measured at 0.040--0.079 ms p50
  for 0.25--8 MiB messages.
- Initial PP2xEP2 stage-local weight loading exposed an unhandled
  `PPMissingLayer` in dInfer's fused-expert conversion.  A bounded fix that
  skips non-local placeholders allowed a second attempt to load successfully:
  PP stages own layers 0--15 and 16--31, EP groups own half the experts, and
  HBM allocation is 51.1/52.9 GiB per rank.  PP4 stage-local loading also
  succeeds at 47.7--52.4 GiB.  Neither result is a live PP request execution.
- Full dLLM PP execution remains blocked by the non-last-stage LM-head return
  contract and the diffusion runner's all-32-layer KV-cache assumption.  The
  current upstream vLLM dLLM plugin likewise declares PP>1 unsupported.
