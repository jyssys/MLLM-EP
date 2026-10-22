# Ouro eight-arm discovery

Experiments and main-agent review are complete.
Verdict: **NO-GO for demonstrated Loop-PP superiority over optimized native TP**.
This is not a claim that every future Loop-PP implementation must lose.

Read [analysis.md](analysis.md), [benchmark_summary.md](benchmark_summary.md), and [correctness.md](correctness.md) first.
The [eight-arm matrix](eight_arm_matrix.md) contains all eight actual configurations.
[quality_results.md](quality_results.md) gives every accuracy denominator and paired interval.
[serving_summary.md](serving_summary.md) separates actual task clocks from clean replay.
[profiling_analysis.md](profiling_analysis.md) and [memory_communication.md](memory_communication.md) describe physical evidence and limits.
[reproduce.md](reproduce.md) gives exact commands, environments, masks, and source archives.

The same 400 questions define each primary arm: 300 GSM8K and 100 MATH.
Native Static TP1/TP4 controls use the same 400 questions.
The official CDB Dynamic reference uses the 114-question pilot.
Full public suites are USER_REDUCED_NOT_RUN.

Historical strict TP numerical failure remains FAIL.
Custom T4/TD and native TP4 quality/timing remain numerically exploratory.
Same-backend Static and Dynamic PP preserve all 400 output sequences within their depth mode.
Dynamic PP also preserves all 400 exit traces.

All 72 clean replay comparisons have five measured repetitions.
All 18 heavy captures preserve bounded output parity.
Heavy wall timing is unqualified for latency headlines.
Full SQLite exports have verified lossless archives.
Original Nsight containers, tensor snapshots, and failures remain on the server.

Only physical GPUs4,5,6,7 run this study.
Authorized unused GPUs return to the requested burners.
The initial profiler PATH failure is resolved and preserved.
The terminal controller state and final review are in state/.

Branch: flashvep/ouro-8arm-static-dynamic-discovery.
Use the containing Git commit to identify this exact report version.
Do not run report_snapshot after final review. It writes an in-progress template.
