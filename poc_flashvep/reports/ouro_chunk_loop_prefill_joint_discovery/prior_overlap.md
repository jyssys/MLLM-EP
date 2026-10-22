# Source audit and novelty limit

The audit used clean pinned checkouts before the G2 patch.
The RLT checkout remained unchanged. The experiment used a separate fork.

| Existing work | Verified mechanism | Boundary for this PoC |
| --- | --- | --- |
| RLT Atomic | `atomic_prefill=True` executes full recurrent depth per admitted token chunk | G4 is prior functionality, not a contribution |
| RLT Split | `SplitPrefillPolicy` provides a recurrent-depth scheduling boundary | G1 is prior functionality. It still obeys token chunk limits |
| RLT Mixed | `MixedPolicy` co-batches compatible decode and prefill recurrent rows | It remains a strong unmodified baseline |
| RLT runner | `ExecutionBatch.loop_count` represents consecutive recurrent calls | The primitive already exists. G2 frontier progression did not exist in the audited scheduler |
| CDB | Full-depth prompt prefill and recurrent decode scheduling | This is a source reference, not a measured competing engine in this study |
| Sarathi-Serve | Chunked prefill controls interference between prefill and decode | Smaller uninterrupted work alone is not a loop-specific novelty claim |

RLT pin: `4a350344027960a48384626aa646f1765fda6d4f`.
The relevant source locations are:

- [Configuration](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/config.py#L59): chunk size and Atomic execution settings.
- [Split and Mixed policies](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/scheduling_policy.py#L156): endpoint boundaries and mixed selection.
- [Execution-batch construction](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/worker/executor.py#L70): native loop-count assignment.
- [Existing split tests](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/tests/test_split_prefill.py): stock causal-frontier and staged-state behavior.

CDB pin: `abdfae0adbdd12be8c6123ccb34eee98a38afb39`.
Its [prefill implementation](https://github.com/kschwethelm/continuous-depth-batching/blob/abdfae0adbdd12be8c6123ccb34eee98a38afb39/src/looped_cdb/continuous_depth_batching/continuous_api.py#L980) performs full-depth prompt work.
We did not import a different exit or KV policy from that engine.

The [Sarathi-Serve paper page](https://www.usenix.org/conference/osdi24/presentation/agrawal) describes the established throughput/latency problem and chunked-prefill approach.
Its model-specific gains are not evidence for Ouro or RLT performance.

The audited RLT source exposed G1 and G4 policies, configurable token chunks, and Mixed batching.
It did not expose an SLO-aware Adaptive Joint C/G policy or G2 scheduler-frontier progression.
The present adapter fills the latter implementation gap only.
It does not establish priority over all later or unexamined work.

The measured fixed-G2 trade-off does not pass the registered continuation gate.
No Adaptive C-only/G-only/Joint comparison or production open-loop result was performed.
Therefore, this PoC does not support a new joint-scheduling contribution claim.
