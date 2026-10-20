# Timing and evidence contract

## Evidence classes

LIVE_MEASURED_PP_ONLY_CONTROL means actual clean PP execution with pretrained weights on the four permitted GPUs.
DIAGNOSTIC_NSIGHT_ACTIVITY means kernels and copies from a separate heavy execution.
NUMERICAL_OBSERVER means actual values with potentially intrusive output copies.
STRUCTURAL means a shape-derived quantity or an execution counter, not a measured network rate.
NOT_RUN means the experiment did not execute.

## Clean clocks

All requests enter as one finite cohort. There is no prompt chunking or tensor batching across requests.
Makespan includes prefill, decode, hidden copies, head work, queueing, publication waits, and scalar token control.
The MP origin is rank 0's admission timestamp on the shared host monotonic clock.
Its endpoint is the final token publication at rank 3.
Worker-stop messages and teardown are outside that endpoint.
The legacy result uses the preserved controller clock. Its stop overhead is small but not identically defined.
That clock difference limits exact legacy-versus-MP causal attribution.

TTFT ends at first token publication and includes admission queueing.
E2E ends at each request's last token publication.
TPOT is `(last_token_time-first_token_time)/(O-1)` for each request.
P50 and P90 refer to requests inside one cohort.
Reported cells are medians of three physical repeats.
Output tokens/s is `C*O/makespan`. Requests/s is `C/makespan`.

Weights, communicator setup, graph capture, buffer setup, and warm-up precede admission.
Raw files record preparation time. All qualified PP controls record zero graph captures inside the serving clock.
Legacy warm-up runs a complete request. MP initially warms local graph stages, not the entire queue/peer path.
First-use queue or peer setup can remain in the MP C1 clock. C4 follows the complete C1 cohort.
This warm-up difference prevents a strict engine-matched comparison with legacy serving latency.
The clean lane records one completion event per stage for safe publication.
It has no per-layer timing events and no Nsight profiler.
The cold process startup time is not the serving clock.

## Output control

The frozen workload stores exact input and feedback token IDs.
The MP hardware control computes the real head and argmax, then uses the frozen token as the next input.
Every computed argmax in the qualified MP controls also matches that frozen stream.
The legacy C0 replication uses live greedy feedback. Its observed IDs match the frozen output stream on every measured request.
These controls therefore execute identical token sequences, but their feedback-control code is not byte-identical.
The separate PP numerical lane uses real free-running feedback.
The TP numerical lane separates aligned comparisons from free-running capped generation.

## Profiler clock

Nsight supplies the common timeline for physical cross-rank kernel overlap.
We do not compare independent CUDA event timestamps across GPUs.
Per-stage CUDA brackets measure local elapsed intervals only.
Stage metadata records host readiness, dispatch, and safe publication separately.
Each receiver reads a tensor only after its producer's completion event permits publication.

Kernel-service sums, activity unions, and clean wall times remain separate.
An All-Reduce kernel can include another rank's readiness delay.
Its service time is not pure payload movement or an exclusive E2E cost.
We report no true bandwidth or additive critical-path breakdown.

## Statistical limits

Paired bootstrap resamples complete physical cohort pairs 5,000 times.
The sampling unit is not a token, layer, loop, or rank.
Only three pairs exist per bounded PP case. Confidence intervals remain exploratory.
No task-level quality confidence interval exists because no accuracy benchmark ran.
