# Track B1 — exact sequential PP baseline and ceiling

An integrated exact PP request baseline could not be established within the
bounded port because the dLLM state/KV contracts described in the topology
audit are missing.  It would be incorrect to call stage-local loading a PP
baseline.

Instead, the validated TP4+EP4 runtime was instrumented at every layer.  For
each real refinement wave, rank-critical layer times were grouped into two or
four contiguous stages and evaluated with an exact flow-shop fill/steady/drain
schedule.  This is an analytical dependency-removal ceiling, not live PP.

| Task | Topology proxy | Waves | Sequential stage proxy | Ideal makespan | Backbone ceiling | Amdahl request ceiling |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | PP2 split 16 | 65 | 10.481 s | 5.891 s | 1.779x | 1.521x / 34.25% reduction |
| GSM8K | PP4 equal 8 | 65 | 10.481 s | 3.371 s | 3.109x | 2.130x / 53.06% reduction |
| HumanEval | PP2 split 16 | 85 | 12.744 s | 6.820 s | 1.869x | 1.536x / 34.91% reduction |
| HumanEval | PP4 equal 8 | 85 | 12.744 s | 3.713 s | 3.432x | 2.138x / 53.23% reduction |

The observer-derived backbone fractions are 78.2% and 75.1%.  The large ideal
ceiling passes the 8% exploration gate, but says nothing about whether stale
boundaries preserve quality.  Boundary transport, PP runtime overhead, and
pipeline contention would lower it.
