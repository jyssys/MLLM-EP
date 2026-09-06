# Runtime comparison

| runtime | status | evidence |
|---|---|---|
| DeepEP V1 high-throughput | PASS | vLLM 0.20.0, `deep_ep 1.2.1+73b6ea4`; real child workers, dispatch tails |
| DeepEP V1 low-latency | FAIL/UNSUPPORTED | initialization assertion: `nvshmem_qp_depth >= (num_max_dispatch_tokens_per_rank + 1) * 2`; no unsafe patch used |
| DeepEP V2 | EXTERNALLY_BLOCKED | no V2 package/build in isolated validated environment; no baseline upgrade performed |
| allgather/reduce-scatter | UNAVAILABLE | not exposed by current vLLM configuration |

The result is not evidence for CROSS_BACKEND_EP or DEEPEP_FAMILY generality.
It remains current HT-path evidence only.
