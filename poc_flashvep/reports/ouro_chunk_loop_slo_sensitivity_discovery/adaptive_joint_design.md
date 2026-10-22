# Adaptive Joint design status

Status: **DESIGN ONLY. NO LIVE ADAPTIVE IMPLEMENTATION.**

The fixed-grid and open-loop study must first establish a useful operating region beyond tuned C-only, G-only, and Mixed.
The historical G2 p99 observation alone does not meet that requirement.

## Candidate cost and deadline contract

A candidate could estimate the cost of a legal prefill quantum from:

- Actual prompt rows, current recurrent frontier, and prior KV length.
- Outstanding decode rows and their remaining recurrent passes before the next token.
- Current next-token deadlines and observed SLO slack.
- Calibrated quantum duration, graph bucket readiness, and staged-state memory.
- Measured scheduling and state-transfer overhead.

The estimator must use independent calibration data.
Actual future output, future measured GPU duration, and future arrival information are forbidden inputs.
An offline best-policy envelope is not a live scheduler.

## Important native RLT interaction

The stock fairness counter counts prefill submissions before recurrent decode submissions.
It resets on a `RECURRENT` batch, not on a completed output token.
Static R4 decoding can therefore encounter multiple prefill quanta before one next token completes.
The number of scheduling boundaries is not the number of completed decode tokens.
The study measures this exposure directly before proposing a deadline controller.

A future C-only controller must receive the same remaining-depth and deadline information as a joint controller.
Otherwise the comparison would confuse better telemetry with a new C/G axis.
The same rule applies to adaptive G-only.

## Required controls before any implementation

Compare the best fixed C-only, fixed G-only, fixed joint pair, and stock Mixed.
Then compare adaptive C-only, adaptive G-only, and adaptive joint under identical telemetry and decision budgets.
Do not use a weak default chunk as the only baseline.
Retain all model, precision, KV, graph, and output-stopping rules.
Unresolved numerical or quality failures prevent a quality-preserving method claim.
