# Cross-track comparison

| Dimension | Track A: temporal EP delta | Track B: approximate timestep PP |
|---|---|---|
| Primary substrate | live true EP4 DeepEP | TP4+EP4 quality emulator + measured PP topology/capacity |
| Core measured signal | 81--85% lag-1 same-rank cache hit | 0.95--0.997 boundary cosine, strongly boundary dependent |
| Ideal/gross opportunity | <=0.72% request E2E | 2.13x ideal no-dependency request ceiling |
| Best feasible/useful point | 0% with unfused codec | <=1.184x worst-task analytical upper at baseline bounded score |
| Quality evidence | codec hidden error only; trajectory not run after kill | full GSM8K/HumanEval trajectory emulator, benchmark execution, sequence/NFE |
| Live method | not implemented by gate | not implemented; exact PP request path unavailable and gate weak |
| Prior-art risk | very high: CompactFusion | very high: AsyncDiff, PipeFusion, DICE, ParaStep |
| EP specificity | high | medium for PP4; higher but unimplemented for PP2xEP2 |
| Decision | **NO-GO** | **CHARACTERIZATION-SIGNAL** |

Track A has a clean causal phenomenon but no economics.  Track B has a large
theoretical dependency ceiling, yet discrete language decisions convert stale
boundary error into more NFEs and trajectory drift.  The best shared policy
does not pass the practical continuation gate even before PP overhead.

Recommendation: do not combine the tracks and do not implement either as the
next main paper direction.  If a future runtime supplies exact LLaDA2 PP at low
engineering cost, Track B can be revisited on larger quality sets; Track A
should remain closed unless communication becomes a radically larger share.
