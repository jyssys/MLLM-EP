
# Path-damage capture

Status: **MEASURED_TRUE_EP4_BOUNDED**

- Frozen cohort: 32 requests, 16 calibration / 16 held-out, four balanced domains.
- Exact selected-branch local labels: 9,312 paths across 1,164 token/layer groups and 32 requests.
- Functional drop-one labels: 256 paths across 32 requests.
- Intervention contract: one original selected path removed, no substitution, remaining original weights renormalized to sum one.
- Functional execution: TRUE TP2-DP2-EP4, same frozen teacher-forced natural continuation, remainder of the current step executed after intervention.
- Intervention marker audit: 256 probes; every materialized target fingerprint matched and exactly one physical rank owned/applied each dropped expert: **True**. Marker-count distribution across source/receiver ranks: `{'2': 8, '3': 120, '4': 128}`.

Full hidden-state dumps were not taken. The raw diagnostic tensors/logits are
kept locally under ignored `artifacts/runs/`; compact labels are versioned.

The first functional run completed cases 00--28 and then correctly rejected
case29 because the initially selected structural step 96 exceeded its 73-token
natural continuation. That failed run is preserved. A versioned plan applies a
damage-blind natural-length fallback to predeclared step 64 for case29, and a
resume run completed cases 29--31. No completed case was rerun or cherry-picked.
