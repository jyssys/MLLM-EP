# Live dynamic-block PoC decision

No production controller was implemented.

The final 56-schedule-per-task exhaustive rollout confirms the gate. At equal
bounded score, the fastest mixed trajectory is 29.22% slower than fixed on
GSM8K and 51.10% slower on HumanEval.

The bounded schedule harness is a correctness and actual-trajectory oracle
mechanism, not a live policy. It changes B only at completed block boundaries,
preserves prefix/KV and position semantics, and records clean wall time and
quality. Fixed-schedule negative controls match the ordinary fixed-B path
exactly on all 32 answers and generated lengths for both B32 and B64 on GSM8K
and HumanEval.

Mixed schedules fail the mandatory oracle gate:

- non-barrier execution separates requests with different current widths into
  extra physical waves;
- the optional batch barrier avoids width divergence but adds waiting and still
  loses to globally tuned fixed-B baselines;
- isolated mini=1 removes cross-request width fragmentation, yet the median
  future-aware gain remains below 1.1%.

Because the perfect/diagnostic ceiling is below 5%, controller overhead,
runtime integration, TP4 control, and online serving claims are not justified.
