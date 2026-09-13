# Candidate oracle tournament

| candidate | scope | strongest measured/analytical result | direct request E2E | final gate |
|---|---|---|---:|---|
| routed EP + shared expert | same request, exact | full shared-work perfect ceiling 2.98--3.50%; partial pair sometimes positive | <=3.50% | KILL <5%, prior-art adjacent |
| remote dispatch + local expert | same request, exact | concurrent path is 0.020--0.092 ms slower at median | 0% | KILL contention |
| EP communication + attention | cross request, exact | dispatch pair negative in all six states; combine weak/mixed | 0% | KILL contention |
| complete-wave 3-stage pipeline | independent waves, exact | 0.106--0.186 ms/layer; service upper 4.35--4.59%, extreme upper 6.73% | 0% | CHARACTERIZATION only |
| phase-aware cross-state pairing | independent waves, exact | shape changes two-stage saving, but does not beat generic 3-stage | 0% demonstrated increment | KILL no direct scope/increment |
| next-step static preparation | same request, exact | recurring exact layout is input-dependent; static workspace persistent | <5% | KILL |
| completed-tile same-wave streaming | same wave, exact | not exposed by current fused boundary | not newly measured | KILL direct prior-art collision |

The strongest credible direct request oracle is the full shared-expert ceiling:
**3.50%**. The strongest independent-wave service upper is **6.73%**, obtained by
the deliberately unrealistic assumption that HumanEval's best sampled 3-stage
saving occurs at every layer and wave. Both fail the 8% promotion gate, and the
latter is not request latency.

Machine-readable scope and gate fields are in `CANDIDATE_ORACLES.csv`.
