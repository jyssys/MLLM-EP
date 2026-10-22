# Adaptive-policy implementation status

A1 SLOWeave-style C-only: NOT_RUN.
A2 Deadline-aware G-only: NOT_RUN.
A3 Deadline-aware Joint: NOT_RUN.
No fixed G2 run is labeled Adaptive Joint.

The offline action enumerator validates new-chunk C selection and recurrent-frontier G selection.
It preserves staged span size and clips G to remaining R4.
The bounded decode definition includes prelude, all remaining loops, and coda.
It never treats one recurrent batch as an output token.

The current predictor does not qualify these definitions as a safe online policy.
Phase-specific prefill coverage fails.
The full-token predictor also includes future stock-policy interruptions.
No causal candidate-specific rollout is validated.
The implementation therefore stops before changing live serving decisions.

The next registered prototype must share predictor, deadlines, telemetry, and fairness across A1/A2/A3.
It must use decode-first when no supported prefill action is safe.
Unknown states must not receive confident extrapolation.
An already infeasible deadline must remain distinct from a new predictor-induced miss.
These are requirements, not implemented performance results.
