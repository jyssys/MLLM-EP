# Source/config audit

Checkpoint configuration reports `qwen3_moe`, `Qwen3MoeForCausalLM`, hidden
size 2048, 48 decoder layers, 128 routed experts, top-8 routing,
`decoder_sparse_step=1`, and `mlp_only_layers=[]`.  Thus all 48 decoder layers
use routed MoE and EP8 places 16 local experts on each GPU.

The validated worker proof reports `DeepEPHTPrepareAndFinalize`,
`DeepEPHTAll2AllManager`, and `TritonExperts` with EP world size 8.  Four
ordinary driver processes are DP0–DP3 and each model worker is TP2, giving
TP2/DP4/EP8/PP1.  The linear map is `global_expert_id // 16`; no EPLB,
replication, routing policy, or scheduler change is active in KEEP.

The existing read-only hook measures per-rank CUDA-event durations around
DeepEP prepare/dispatch, TritonExperts, and finalize/combine and records local
16-expert histograms.  The TEMP wrapper calls the real router logits only
after the post-initialization arm file appears.  PERSIST is intentionally
out-of-band: official EPLB is run offline and the migration benchmark
broadcasts one actual local expert's two BF16 weight tensors over the EP group.

Because the capture does not retain token-level alternative routes, the TEMP
wrapper's invalid slots and output differences are treated as validity costs;
they are not hidden by a latency-only objective.
