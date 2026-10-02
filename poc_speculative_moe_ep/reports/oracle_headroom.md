# Oracle Headroom

Measured clean TRUE EP4 on GPUs2,3,6,7, output cap64/EOS, first16 frozen requests (8+8). Route features come from a separate trace pass with identical request/prefix/draft IDs.

{
  "status": "OFFLINE_SAME_PREFIX_AVAILABLE_OPPORTUNITY_DIAGNOSTIC",
  "gamma0": "AR one target row, not one draft proposal",
  "common_opportunity_count": 106,
  "request_count": 16,
  "best_fixed_gamma": 8,
  "strongest_target_side_baseline": "acceptance",
  "target_side_gain_pct": {
    "mean": 0.04294250511069813,
    "ci95": [
      -0.13661324350225187,
      0.21642140227849385
    ],
    "independent_unit": "request",
    "n": 16,
    "bootstrap": 2000
  },
  "target_gain_vs_fixed_pct": {
    "mean": 0.4838670163909886,
    "ci95": [
      0.10910972798383146,
      1.0028258285963747
    ],
    "independent_unit": "request",
    "n": 16,
    "bootstrap": 2000
  },
  "target_gain_vs_expert_pct": {
    "mean": 12.622756470146385,
    "ci95": [
      6.7961094081289,
      19.500777151604687
    ],
    "independent_unit": "request",
    "n": 16,
    "bootstrap": 2000
  },
  "target_gain_vs_aggregate_oracle_pct": {
    "mean": -0.34443880553450623,
    "ci95": [
      -0.5473958133391489,
      -0.16606780221488712
    ],
    "independent_unit": "request",
    "n": 16,
    "bootstrap": 2000
  },
  "e2e_gain": null,
  "scope": "measured target-forward per progress on available common prefixes; not policy rollout; no E2E oracle",
  "warning": "trace features for clean pass; limited aligned opportunities; per-width native draft prefixes checked, not assumed"
}

Detailed alternatives stay in machine-readable JSON. Only prefixes actually shared by all4 native arms are eligible; token-prefix equality/draft nesting are checked, not assumed. gamma0=AR, not SDgamma1. Costs are real clean target labels; the analysis is not an adaptively executed policy. No oracle E2E is fabricated, and no missing alternatives are interpolated. Aggregate measured verification-cost oracle is a strongest relevant control and can already subsume communication.
