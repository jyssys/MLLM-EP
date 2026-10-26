# Looped MoE expert-parallel discovery

Verdict: **NO_GO_A / NO_GO_B for the tested designs. No architecture GO.**
Production generalization remains **CONDITIONAL**.
The experiment ran the three exact checkpoints on TRUE EP4 and EP8.
All six configurations passed native-versus-EP engineering fidelity.
The tested recurrence-aware policies did not reach the 1.15× research gate.
No optimized A/B dispatcher receives a measured speedup claim.

| Model | EP | A vs best generic | B vs static, max | B vs generic, max | B load-only ideal |
| --- | --- | --- | --- | --- | --- |
| IFM Ouro | 4 | 1.000× | 1.068× | 1.002× | 1.083× |
| IFM Ouro | 8 | 1.000× | 1.094× | 1.002× | 1.112× |
| IFM Huginn | 4 | 1.000× | 1.055× | 1.004× | 1.074× |
| IFM Huginn | 8 | 1.000× | 1.067× | 1.007× | 1.092× |
| Foil-1 | 4 | 1.000× | 1.006× | 1.001× | 1.016× |
| Foil-1 | 8 | 1.000× | 1.009× | 1.008× | 1.022× |

Every ratio above is a modeled estimate, not measured E2E acceleration.
The B maxima can occur in different cells.
They are not ratios of independently selected maxima.
Column B versus generic uses matched model, degree, batch, prefix, and replica budget.

Read [report.md](report.md), [go_no_go.md](go_no_go.md), and [reproduce.md](reproduce.md).
See [methodology.md](methodology.md) for the frozen fidelity and timing limits.
All baseline numbers come from actual distributed GPU execution.
The native generation path recomputes the entire prefix without a KV cache.
These results do not establish a production DeepEP limit.
