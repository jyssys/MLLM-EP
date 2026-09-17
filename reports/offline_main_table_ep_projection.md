# Offline main table draft: routed-MoE EP projection

Primary topology remains `DP1/TP1/SP1/EP{1,4,8}`. No TP8/SP8 timing was invented. Because the F1 n=32 gate failed, EP1 replay calibration and all n=128 common projections were not launched. The only comparable entries below are matched GSM8K IDs 0–31.

These are routed-MoE stage projections, **not E2E latency**.

| Method | EP | stage ms/request | speedup vs same-EP Vanilla | dispatch | expert | combine | max/mean | max/second | CV | wait fraction | remote logical bytes/request | label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Vanilla | 4 | 2002.752 | 1.000x | 310.642 | 1520.384 | 171.725 | 1.098 | 1.077 | 0.073 | 0.088 | 29,486,782,592 | SIMULATED-EP4-EP2-CALIBRATED |
| OURS-F0 | 4 | 1882.978 | 1.064x | 291.230 | 1430.953 | 160.795 | 1.099 | 1.077 | 0.073 | 0.088 | 27,095,757,440 | SIMULATED-EP4-EP2-CALIBRATED |
| OURS-F1 | 4 | 1940.082 | 1.032x | 300.160 | 1474.179 | 165.742 | 1.099 | 1.077 | 0.073 | 0.088 | 27,963,982,336 | SIMULATED-EP4-EP2-CALIBRATED |
| Vanilla | 8 | 1743.052 | 1.000x | 307.837 | 1265.254 | 169.961 | 1.223 | 1.098 | 0.125 | 0.177 | 43,110,431,872 | SIMULATED-EP8-EP2-CALIBRATED |
| OURS-F0 | 8 | 1640.239 | 1.063x | 288.775 | 1192.212 | 159.252 | 1.225 | 1.099 | 0.126 | 0.177 | 39,629,195,648 | SIMULATED-EP8-EP2-CALIBRATED |
| OURS-F1 | 8 | 1690.159 | 1.031x | 297.573 | 1228.470 | 164.116 | 1.225 | 1.099 | 0.126 | 0.178 | 40,901,247,232 | SIMULATED-EP8-EP2-CALIBRATED |
| TEAM-PORT | 1/4/8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | not run after stop |
| REFLEX | 1/4/8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | official code unavailable |

EP1 is intentionally N/A rather than an EP2/P estimate: the new EP1 path and replay code passed unit tests, but no GPU replay was run after the stop condition.
