# Exact D-local Prefill frequency control

Declared on 2026-11-03 at approximately 19:45 KST, before these frequency runs.

The existing Ouro size sweep tests 32, 64, 128, and 256 input tokens.
It uses sparse probes and stops at the first failed acceptance gate.
Sparse successful requests do not establish a useful continuous borrowing budget.

After that sweep and the primary shorter-input Huginn control finish, select the largest accepted probe size.
Test requested probe intervals of 1.0, 0.5, and 0.25 seconds, in that order.
Stop at the first failed gate or the existing experiment deadline.
The maximum requested probe count is 128.
The native active-background and measurement-window checks still limit actual injection.

Each frequency has a new matched no-probe baseline.
Background prompts, arrivals, output caps, GPU mapping, and warmup remain unchanged.
The policy does not use future output lengths or future arrivals.
Native scheduling, KV allocation, and exact full R4 computation remain active.

Record actual completed probes, additional prompt tokens, full-depth recurrent rows, and response-wall duration.
Requested injection frequency is not actual admitted throughput.
Report TPOT p95, ITL p99, maximum ITL, and exact background/probe output checks.
Do not hide a failed frequency because a lower one passed.

These runs have no P device capture.
They are D-load sensitivity controls, not new same-run P/D saturation proofs.
They do not implement P-to-D continuation or establish Dynamic A speedup.

The controller is `poc_flashvep.looped_pd_autonomous_headroom.qualified_probe_load`.
The state ledger is `state/qualified_probe_load_results.json` in the campaign result directory.

If a frequency passes and the original closeout budget permits, run one P-side capture of the accepted level with the highest actual added prompt throughput.
Use the same configuration, arrival trace, and matched clean run.
The controller is `poc_flashvep.looped_pd_autonomous_headroom.qualified_load_joint`.
All original same-window gates apply.
This optional capture is one additional start, not a replacement for the independent 16-token confirmations.
