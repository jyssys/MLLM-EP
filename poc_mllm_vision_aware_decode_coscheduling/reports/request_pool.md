# Frozen request pool

The routing-blind pool was frozen before this Stage-A scheduling analysis by requiring natural EOS/stop and at least 128 generated tokens. It contains **20** requests: 12 discovery and 8 held-out. The requested 24-request minimum could not be met from the existing bounded cohort, leaving a declared shortage of **4** rather than relaxing eligibility.

- Dataset counts: mathvision=3, mathvista=7, wemath=10
- Image hashes: unique across all selected requests
- Global wave size: 4
- Held-out waves: 2 per policy
- Decode trace: all 48 MoE layers and 22 checkpoints through step 128
- Selection used routing oracle: `false`

The discovery/held-out split is deterministic and image-disjoint. Policy assignments were frozen before any physical runtime timing. The compact source trace is committed at `artifacts/source_pool_trace.jsonl.gz` so the analysis can be reproduced without the prior 102-MiB trace.

The separated-signature alpha values were frozen in the predecessor study, and this scheduling split is new: its held-out set contains 6 requests from the predecessor discovery split and 2 from its held-out split. It is therefore held out from the current grouping analysis, but not a wholly virgin test of those inherited alpha values. This limitation cannot create the reported positive conclusion because S4 and S3/S2 are exactly tied.
