# Resource fallback acceptance

BF16 TP1 and TP2 paths loaded the unmodified checkpoint and generated all 192
frozen requests. TP1 used about 63.2 GiB including runtime/KV allocations;
TP2 used about 35.5 GiB per GPU. Semantic cohort status is
`LOW_RESOURCE_SEMANTIC_READY_NUMERIC_DRIFT_OBSERVED`.

- exact token sequence agreement: 90.104%
- parsed-answer agreement: 96.875%
- TP2-TP1 paired accuracy delta: 0.521pp,
  CI `[-1.0416666666666665, 2.604166666666667]`

Frozen trace prediction host invariance:

- host1 target EP2/4/8: {'2': True, '4': True, '8': True}
- host2 target EP2/4/8: {'2': True, '4': True, '8': True}

Host execution wall time and target EP predicted time remain separate. EP8
retains the extrapolation label on both hosts.
