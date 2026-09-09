# Frozen comparison protocol before performance screening

## Evidence hierarchy

1. Official source identity and actual active execution path.
2. Same-checkpoint output sanity, then cross-configuration correctness.
3. Uninstrumented request-level measurements under independent restarts.
4. Separate instrumented mechanism runs with request joins and same-device CUDA
   events. Host spans and CUDA durations are distinct; no cross-GPU CUDA timestamp
   subtraction. JIT/build/loading, burn and idle residence are not experiment time.
5. Minimal faithful transfer; unsupported port is never method failure.

FastPP's one-line EOS repair is retained in every configuration. Its original
`ignore_eos=True` fixed-output behavior is unchanged. Isolated NanoFlow API repairs
are versioned separately from any scheduling/plan adaptation.

## Paired comparisons

- Same request contents, arrival schedule, precision, warmup and physical mapping
  within each candidate. Randomized run order, at least three independent
  restarts for important results and five for a strong candidate.
- Report all requests, including errors and timeouts separately. Performance
  exclusion reasons must be explicit, not silently removed slow samples.
- Primary effect: per-restart request E2E mean/median reduction and paired request
  effects. Bootstrap restart blocks for uncertainty; rows from one engine are not
  independent serving runs. Report TTFT, TPOT/ITL tails and throughput jointly.
- Compare output lengths and output hashes. Divergence triggers diagnosis; neither
  a nonempty response nor a hash mismatch alone establishes model correctness or
  failure. Inspect trusted-reference prefixes and numerical semantics as needed.
- Client stream coalescing is exposed. Client-observed emission is not claimed
  to be an internal token-ready timestamp.

## SLOs

FastPP original control uses TTFT 2 s and TPOT 0.2 s. Report separate tighter H100
SLO controls if the original threshold is slack; do not choose SLO post hoc to
maximize a headline. Goodput means the highest tested arrival rate attaining
the declared joint-request SLO target, not simply compliant requests divided by
one arbitrary burst's duration. Report the whole rate/attainment curve.

The completed finite-arrival screens currently measure **attained SLO goodput**
(compliant requests / observed trace duration), not that maximum sustainable
capacity. Their tables are explicitly named attained-goodput envelopes. A
capacity claim would still require an arrival-rate/attainment sweep and cannot
be inferred from these fixed traces. The distinction is retained even if an
attained-goodput ratio is favorable.

## Oracles and trivial fixes

- Best static: one measured configuration selected for all workload regimes.
- Per-workload static: one measured configuration per workload, selected on
  calibration restarts and evaluated on independent restarts when possible.
- Small portfolio: at most a few existing plans; include transition/setup cost.
- Per-request minima from different counterfactual runs are only a clairvoyant
  optimistic bound, not an executable joint serving schedule. Changes in batching
  and interference prevent blindly composing independent request savings.
- A stage-duration or partition-score oracle is not direct E2E evidence. Validate
  its request timeline against live baseline execution before promotion; otherwise
  label it a trace-driven diagnostic with unverified E2E headroom.
- A simple knob recovering >=80% of the measured failure makes an incremental
  candidate. Additional direct E2E <5% closes a supported candidate; 5–10% is
  weak, 10–15% serious, >=15% strong. Strong SLO-goodput signal requires >=15%.

## Interpretation guardrails

Unsupported Qwen3-VL encoder/MRoPE/DeepStack is a port constraint. Adding encoder
time to the denominator and observing smaller LLM-only speedup is ordinary
Amdahl dilution, not an unexplained MLLM failure. A modality claim needs matched
token/context controls and a causal interaction beyond known encoder scheduling.

No ranking until all three have been seriously screened at the same milestones;
environment-blocked milestones retain their missing-evidence labels.
