# Actual DP scale-out

Each replica owns one complete model and its local KV state.
The dispatcher assigns global request i to replica i modulo DP width.
All replicas use one common offered burst timestamp.
Each cluster receives the same 32 short arithmetic chat requests.
The table measures actual execution, not trace replay or a throughput simulator.

| Model | DP | Arm | Warm-up condition | Start block | Makespan s | Requests/s | Requests/s/GPU | Speedup vs matched model DP1 Mixed | Complete/offered |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | 1 | hybrid | LIBRARY_WARM_ONLY | r1 | 3.8425 | 8.328 | 8.328 | 1.024× | 32/32 |
| huginn | 1 | mixed | LIBRARY_WARM_ONLY | r1 | 3.9351 | 8.132 | 8.132 | 1.000× | 32/32 |
| huginn | 2 | hybrid | LIBRARY_WARM_ONLY | r1 | 1.9695 | 16.248 | 8.124 | 1.998× | 32/32 |
| huginn | 2 | mixed | LIBRARY_WARM_ONLY | r1 | 2.0472 | 15.631 | 7.815 | 1.922× | 32/32 |
| huginn | 4 | hybrid | LIBRARY_WARM_ONLY | r1 | 0.9679 | 33.062 | 8.266 | 4.066× | 32/32 |
| huginn | 4 | mixed | LIBRARY_WARM_ONLY | r1 | 0.9685 | 33.041 | 8.260 | 4.063× | 32/32 |
| huginn | 8 | hybrid | LIBRARY_WARM_ONLY | r1 | 0.5840 | 54.794 | 6.849 | 6.738× | 32/32 |
| huginn | 8 | hybrid | LIBRARY_WARM_ONLY | r2 | 0.6758 | 47.354 | 5.919 | 5.823× | 32/32 |
| huginn | 8 | hybrid | LIBRARY_WARM_ONLY | r3 | 0.5870 | 54.510 | 6.814 | 6.703× | 32/32 |
| huginn | 8 | hybrid | MATCHED_CHAT_WARM | r4 | 0.5317 | 60.187 | 7.523 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | hybrid | MATCHED_CHAT_WARM | r5 | 0.5567 | 57.487 | 7.186 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | hybrid | MATCHED_CHAT_WARM | r6 | 0.5449 | 58.729 | 7.341 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | hybrid | MATCHED_CHAT_WARM | r7 | 0.6200 | 51.615 | 6.452 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | hybrid | MATCHED_CHAT_WARM | r8 | 0.5378 | 59.498 | 7.437 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | mixed | LIBRARY_WARM_ONLY | r1 | 0.6914 | 46.284 | 5.786 | 5.692× | 32/32 |
| huginn | 8 | mixed | LIBRARY_WARM_ONLY | r2 | 0.7104 | 45.047 | 5.631 | 5.540× | 32/32 |
| huginn | 8 | mixed | LIBRARY_WARM_ONLY | r3 | 0.6665 | 48.015 | 6.002 | 5.905× | 32/32 |
| huginn | 8 | mixed | MATCHED_CHAT_WARM | r4 | 0.5407 | 59.179 | 7.397 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | mixed | MATCHED_CHAT_WARM | r5 | 0.5538 | 57.777 | 7.222 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | mixed | MATCHED_CHAT_WARM | r6 | 0.5642 | 56.719 | 7.090 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | mixed | MATCHED_CHAT_WARM | r7 | 0.5557 | 57.582 | 7.198 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| huginn | 8 | mixed | MATCHED_CHAT_WARM | r8 | 0.5484 | 58.354 | 7.294 | NOT_AVAILABLE_matched_DP1 | 32/32 |
| ouro | 1 | hybrid | LIBRARY_WARM_ONLY | r1 | 2.9330 | 10.910 | 10.910 | 0.956× | 32/32 |
| ouro | 1 | mixed | LIBRARY_WARM_ONLY | r1 | 2.8044 | 11.410 | 11.410 | 1.000× | 32/32 |
| ouro | 2 | hybrid | LIBRARY_WARM_ONLY | r1 | 1.5222 | 21.022 | 10.511 | 1.842× | 32/32 |
| ouro | 2 | mixed | LIBRARY_WARM_ONLY | r1 | 1.4795 | 21.629 | 10.815 | 1.896× | 32/32 |
| ouro | 4 | hybrid | LIBRARY_WARM_ONLY | r1 | 0.9280 | 34.481 | 8.620 | 3.022× | 32/32 |
| ouro | 4 | mixed | LIBRARY_WARM_ONLY | r1 | 0.9334 | 34.284 | 8.571 | 3.005× | 32/32 |
| ouro | 8 | hybrid | LIBRARY_WARM_ONLY | r1 | 0.4846 | 66.037 | 8.255 | 5.787× | 32/32 |
| ouro | 8 | mixed | LIBRARY_WARM_ONLY | r1 | 0.4813 | 66.489 | 8.311 | 5.827× | 32/32 |

Makespan includes actual owned-engine drain after the final response.
TTFT and ITL use delivered frontend timestamps.
All-offered request records remain in `dp_request_metrics.csv`.
Ouro honors its registered chat-end token in the frontend.
Huginn uses its native EOS behavior.

The initial complete sweep has one cluster start per configuration.
The Huginn DP8 positive point receives two additional independent starts with crossed arm order.
That confirmation is not a full capacity experiment.
The work has no registered Poisson-rate or stability-qualified 90% SLO crossing.
Do not call these finite-burst rates sustainable goodput.

The Hybrid arm is the fixed event hybrid control.
It is not OURS_ADAPT_G_MIXED.
Any positive static-hybrid result remains a control result, not evidence of adaptive superiority over the best fixed hybrid.
DP scaling is generic replication and does not establish a new multi-GPU parallelism mechanism.

Huginn DP8, LIBRARY_WARM_ONLY: paired makespan speedup 1.1234×, 95% interval [1.0512, 1.1839].
This interval covers 3 starts and one repeated short-chat source family.

Huginn DP8, MATCHED_CHAT_WARM: paired makespan speedup 0.9927×, 95% interval [0.9439, 1.0254].
This interval covers 5 starts and one repeated short-chat source family.
