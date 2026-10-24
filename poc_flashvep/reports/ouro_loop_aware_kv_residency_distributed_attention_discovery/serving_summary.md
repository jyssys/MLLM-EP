# Serving scope

The native model completed 295 retained timing cohorts.
The primary lane is warmed-engine, bounded hardware inference.
It records actual frontend timestamps and generated token IDs.
It does not use frozen output-token replay.
It does ignore EOS to control hardware output length.

Separate EOS-enabled book-continuation probes retain their natural stop outcomes and 128-token cap.
They are not an unrestricted long-answer benchmark.
The report makes no sustainable serving-capacity claim from those finite probes.

The distributed path is an operator, not a model server.
Its timing cannot populate TTFT, TPOT, or natural SLO tables.
Those candidate serving metrics are `NOT_RUN`.
The pause on native-vLLM Loop PP remains in force.
