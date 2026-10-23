# Prospective online budget contract

The initial policies use a fixed prefill service budget.
The budget is not a token deadline or an ITL guarantee.
The candidate model predicts one prompt span at one legal frontier.
Each adaptive batch therefore contains at most one prefill span.
All adaptive C-only, G-only, and Joint controls use this same packing rule.
Native Mixed remains a separate unchanged baseline.

C1A uses native Atomic G4 and adaptive C.
C1S uses staged G4 and adaptive C.
G1 uses a fixed C and adaptive legal G.
J1 can choose C at a fresh chunk boundary and G at each legal continuation.
All four use the same lookup, budget, completed-token fairness, and selector infrastructure.
The CPU cost of this infrastructure remains part of measured serving latency.

Every continuation retains its original token span.
Every prompt token still completes four recurrent passes.
A new staged chunk pays prelude plus all recurrent groups in the completion utility.
Each issued native prelude or recurrent quantum must independently fit the configured budget prediction.
This definition does not combine two separately schedulable quanta into one uninterrupted unit.

If no candidate fits, the admission filter prioritizes legal decode work.
If no decode request exists, the smallest predicted service action can make progress.
A frontier waiting for 250 ms can use the same explicit progress escape.
Both exceptions can exceed B and receive an overbudget label.
The implementation never claims hard bound compliance for these actions.

The initial objective is U2, completed tokens divided by estimated remaining full-chunk service.
An optional shared decode-yield penalty can account for additional continuation boundaries.
That penalty must come from new calibration and freeze before held-out evaluation.
U0 remains a partial-work diagnostic, not a reported serving gain.

The optional feedback controller uses only previously delivered output-token timestamps.
It does not predict a future token completion time.
It requires 32 observed gaps and 16 output updates before changing its budget.
Its shared target and hysteresis must freeze before candidate held-out serving.

No live adaptive result exists at this registration point.
The sixteen CPU tests use software fixtures, not pretrained quality evidence.
