# Timing validity

The clean forward encloses native model execution and CUDA completion.
A distributed barrier precedes the interval.
Rank-local elapsed durations do not require synchronized absolute GPU clocks.
The maximum elapsed rank duration defines a cohort observation.
Three independent starts cover prefix 128 at every B1/B2/B4 cell.
The prompt cohort also changes across starts, at offsets 0/4/8.
Intervals therefore combine process-start and prompt-cohort variation.
Long-prefix cells use one start with twenty serial repetitions.
The latter confidence intervals are exploratory.

Heavy CUDA-event and CUPTI traces run separately.
API spans include launch gaps and Python control work.
They cannot form a clean E2E percentage pie.
The report uses one cumulative-critical rank, not different maxima per component.
Rank waiting and CPU overlap remain PARTIAL.

Foil's first 32-token clean trace took 2976.160 ms in one EP4 start.
The subsequent instrumented trace took 1922.372 ms and preserved token IDs.
The observed ratio was 0.646, not a credible negative instrumentation overhead.
The diagnosis separated first-shape preparation, hardware state, and observer-path changes.
The warmed follow-up uses the same uninstrumented path four times.
`summary/shape_drift.csv` records first/last and last-two ratios.
The repeated-generation control identifies condition drift but does not prove a specific JIT kernel caused it.

The headline generation table uses only warmed repetitions.
The original first-occurrence matrix remains unchanged and explicitly qualified.
Every generation measurement uses full-prefix recomputation without cached KV.
Output caps are hardware conditions, not natural EOS completion.
