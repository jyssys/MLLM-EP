# Prior-art and upstream collision matrix

| Track | Closest active work | Exact overlap | Remaining gap considered here |
|---|---|---|---|
| A: DCP × speculative decoding | vLLM #50391, #45425, #53673; backend DCP-varlen implementations | Exact verification semantics and CP-aware draft/target composition are active upstream work | Only a measured, nontrivial request-level benefit or missing cross-backend abstraction beyond the RFC could survive. Merely implementing one RFC strategy has no novelty. |
| B: PCP perpendicular DCP | vLLM #46358, #51429, #53573 and current PCP manager | Orthogonal PCP/DCP is an explicit active design | A real transition copy/duplication not already eliminated by canonical slot writes, plus a general zero-transition layout abstraction, would be required. |
| C: MoE-aware CP degree | CP deployment heuristics and existing parallelism planners | Length-/attention-aware CP selection is established; simple TP/CP load switching is crowded | A direct request-level regret caused specifically by MoE execution geometry after matching length, not recovered by a length threshold, could remain distinct. |

Audit links: [vLLM CP deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/), [DCP × speculation RFC #50391](https://github.com/vllm-project/vllm/issues/50391), [PCP perpendicular DCP RFC #46358](https://github.com/vllm-project/vllm/issues/46358), [draft/target parallelism RFC #53673](https://github.com/vllm-project/vllm/issues/53673), [MTP+DCP correctness #45425](https://github.com/vllm-project/vllm/issues/45425), [DCP-over-PCP validation #51429](https://github.com/vllm-project/vllm/issues/51429), and [PCP+DCP MLA collective correctness #53573](https://github.com/vllm-project/vllm/issues/53573).

The audit was adversarial: an open RFC is not proof that the implementation is solved, but it is enough to reject novelty based only on restating its problem and proposed exact strategies.
