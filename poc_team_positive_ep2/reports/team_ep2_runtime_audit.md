# TEAM true-EP2 runtime audit

## Verified topology

| Property | Verified value |
|---|---|
| Tensor parallelism | TP1 |
| Data parallelism | DP1 |
| Expert parallelism | EP2 |
| Rank 0 ownership | experts 0–63 in every layer |
| Rank 1 ownership | experts 64–127 in every layer |
| Source rank | rank 0 runs released router and TEAM masks |
| Dispatch | NCCL `all_to_all_single` of sorted token-expert hidden rows |
| Local execution | only resident owner experts execute |
| Combine | reverse NCCL `all_to_all_single`, weighted index-add on rank 0 |
| Non-MoE path | replicated; exact combined tensor broadcast after MoE |

This is **true ownership and remote sparse execution**, not a configuration flag:
the non-owner expert modules are replaced by parameter-free sentinels after
checkpoint load, remote assignment counts are nonzero, and each rank can only
execute its 64 resident experts.  The warm TEAM trace had 178,533 remote source
assignments in the fused-shape run.  Tests assert ownership disjointness and
dispatch/combine work conservation for every captured layer call.

## Semantics boundary

TEAM's decoded-token caching, hot/cold classification, speculative exploration,
limited activation, expert selection, and decoder policy were not altered.  The
port replaces only physical MoE execution after the official checkpoint is
loaded.

Direct layer-level correctness against the original full-expert implementation:

| Local expert path | Cosine | Relative L2 | Max abs | Router-logit max abs |
|---|---:|---:|---:|---:|
| Python reference | 0.99999994 | 0.0505% | 2.44e-4 | 0 |
| vLLM fused | 1.00000000 | 0.0553% | 2.44e-4 | 0 |

The small output difference is FP16 accumulation/reduction-order noise; route
logits are identical. This is stronger and more interpretable than requiring
long diffusion generations to be token-identical under numerically reordered
expert sums.

The backend is intentionally
`torch.distributed.nccl_all_to_all_single_reference`, not DeepEP.  It proves the
EP2 mechanism and preserves decisions, but it is not a production performance
claim.  DeepEP integration was not necessary to decide the residual oracle and
would have changed the task from bounded physical-execution extension to a
backend port.

## Memory

The Python-reference sharded model occupies about 32.1 GB per rank after
non-owned experts are removed.  The diagnostic vLLM-fused control occupies
about 61.1 GB because it stacks local fused weights before all Python parameter
references are released.  That temporary diagnostic memory is an
implementation artifact, not a required EP2 footprint.

## Evidence

- `plots/06_ep2_ownership.png`
- `analysis/ep2_work.csv`
- `analysis/ep2_communication.csv`
- `tests/test_analysis.py`
