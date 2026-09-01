# Source and semantics audit

The experiment was pinned to the repository working tree before the live runs.
The local vLLM installation reports `v0.20.0` and uses the V1 scheduler.  The
public references inspected for this audit are:

* <https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/>
* <https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/>
* <https://github.com/vllm-project/vllm/blob/main/vllm/config/parallel.py>
* <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_moe.py>
* <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl_moe.py>
* <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py>
* <https://github.com/deepseek-ai/DeepEP>

## Parallel groups

Local `vllm/config/parallel.py` defines `world_size_across_dp = TP * PP *
DP` and reads offline DP rank/size from `VLLM_DP_*`.  Local
`distributed/parallel_state.py` constructs TP groups from the TP dimension,
DP groups by transposing the DP/TP dimensions, and the EP group over
`DP * PCP * TP` ranks for an MoE model.  The experiment records all group rank
lists from `GroupCoordinator`; it does not infer them from GPU IDs.

For TP2/DP2/EP4, `use_sequence_parallel_moe` is true for DeepEP HT because it
requires EP, TP>1, DP>1.  For TP1/DP4/EP4 it is false because TP=1.  This is a
known topology confound and is reported with the results.  Both cases still
have an EP group of four ranks and 32 linear-placement experts per rank.

## Decoder boundary

Local `Qwen3MoeDecoderLayer.forward` executes input RMSNorm, self attention,
post-attention RMSNorm, and then `self.mlp(hidden_states)`.  The experiment
wraps the existing decoder layer only to record a layer-entry to MoE-entry
CUDA-event span; no tensor or route is changed.  Qwen3-VL's language decoder
uses the same Qwen3 MoE decoder layer implementation through its VL model
class.

## DeepEP HT synchronization

Local `deepep_ht.py` calls `get_dispatch_layout`, then `buffer.dispatch` with
`previous_event` and `async_finish` (disabled by DBO-off semantics only where
appropriate), and calls `buffer.combine` in `_finalize` with another
`previous_event`.  In the synchronous `prepare`/`finalize` path the returned
combined tensor is copied after `combine`; an `EventOverlap` can make the
current stream wait when asynchronous finalize is used.  Therefore this PoC
does not call a cross-GPU absolute CUDA timestamp subtraction.  It reports a
duration-based DP pre-MoE arrival-skew and a conservative EP completion-spread
wait proxy, with the DeepEP calls unchanged.

## Configuration proof

Each worker writes `topology_proof/rank*.json` with physical GPU, PID, global
rank, DP/TP/EP/PP rank and group rank lists.  Backend proof files from the
existing read-only hook record `DeepEPHTPrepareAndFinalize`,
`DeepEPHTAll2AllManager`, TritonExperts, and EP world size 4.
