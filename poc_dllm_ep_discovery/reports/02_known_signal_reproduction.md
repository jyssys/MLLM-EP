# Known-signal reproduction

## 1. Liveness collapse: reproduced

Median decision-live fraction falls from 0.904 to 0.181 on GSM8K (-80.0%) and from 0.783 to 0.172 on HumanEval (-78.1%). This is the strongest logical change.

## 2. Aggregate expert-support broadening: not reproduced

Across naturally shrinking ready pools, active routed experts fall rather than rise: 186 to 160 on GSM8K and 200 to 116 on HumanEval. Therefore the raw early/late trace does **not** establish the hypothesized `liveness down, active experts up` paradox.

A fixed-physical-M control tells a narrower story. At M=1024, the lowest-live wave activates 198 versus 173 experts on GSM8K (+14.5%) and 214 versus 200 on HumanEval (+7.0%). Thus lower logical liveness can coexist with broader support while physical row count is fixed. The expert-latency increase is only +6.21% and +1.69%, while whole-MoE latency is actually -1.00% and -0.32%. This is reproducible characterization, not large E2E headroom.

## 3. Tiny-expert growth: reproduced, but confounded

Natural early-to-late `<=4 row` active-expert fraction grows:

- GSM8K: 25.5% to 46.2% (+20.7 percentage points).
- HumanEval: 26.4% to 64.0% (+37.6 points).

At fixed M=1024, however, the lowest-live wave has **less** tiny-expert mass than the highest-live wave (GSM8K -13.4%; HumanEval -23.1%). The natural late-phase growth is therefore driven mainly by the shrinking physical ready pool, not by liveness itself causing a diffuse expert geometry at fixed work.

## 4. Coarse-stable / fine-volatile routing: reproduced

Lag-1 live-row routing becomes more overlapping late, but exact reuse remains low:

| dataset/phase | top-k overlap | exact top-k set | exact destination-rank set | rank-load cosine |
|---|---:|---:|---:|---:|
| GSM8K early | 57.9% | 4.8% | 45.9% | 0.9987 |
| GSM8K middle | 63.3% | 8.2% | 66.5% | 0.9997 |
| GSM8K late | 71.9% | 18.2% | 74.2% | 0.9991 |
| HumanEval middle | 60.2% | 7.3% | 64.3% | 0.9991 |
| HumanEval late | 65.2% | 7.5% | 74.0% | 0.9986 |

Coarse EP geometry is almost invariant even while exact expert identity changes substantially. This explains why a rank-level predictor can look easy while exact route-plan or expert-output reuse remains unsafe.

Evidence: `PHASE_MISMATCH_SUMMARY.csv`, `FIXED_M_CONTROLS.csv`, `TEMPORAL_ROUTE_STABILITY.csv`.

