# Source and runtime audit

- Model: local `Qwen3-VL-30B-A3B-Instruct` snapshot `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- Runtime: vLLM 0.20.0 V1, BF16, `TP2/DP2/EP4`, eager, chunked prefill (`max_num_batched_tokens=8192`), DBO off, prefix cache off.
- GPU visibility: `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical mapping recorded in each topology JSON).
- Runtime proof in `online_trace3/serving.log`, `online_trace_high2/run.log`, and `online_trace_real/run.log`: `Using DeepEPHTAll2AllManager`, EP world size 4, linear 32/128 experts, `Using TRITON Unquantized MoE backend`, `Using DeepEPHTPrepareAndFinalize`.
- `online_trace_real` uses local natural `skimage` images (astronaut and motorcycle); the earlier trace3/high2 runs used bounded PIL fixtures and are retained as separate controls.

## Boundary

`Qwen3MoeDecoderLayer.forward` (local source `.../vllm/model_executor/models/qwen3_moe.py`, class at line 364 and forward at line 416) performs attention then `self.mlp`. The fused-MoE runner obtains `topk_ids` from the router and invokes the unquantized method. The local read-only hook wraps that stock `apply` call with CUDA events and records the exact `topk_ids`; no route, tensor, placement, or scheduler operation is changed.

DeepEP HT is selected through `all2all_backend=deepep_high_throughput`; local `deepep_ht.py` dispatch starts at line 97 and finalization/combine at line 336. The path passes `previous_event` through the communication stream and uses asynchronous DeepEP prepare/finalize semantics. The online hook therefore treats the full stock apply interval as `T_MoE`; it does not claim that dispatch/expert/combine subspans are separately timestamped in the serving trace. Those subspans are available only in the separately labeled route-transfer replay.

## Feature caveat

Per-token fanout is computed exactly from the captured top-k expert IDs (`expert_id // 32`), yielding F1–F4. The hook's sender-destination matrix intentionally records a conservative local-source row; it is not a global cross-DP traffic matrix. Fanout conclusions are therefore not conflated with a complete sender matrix.
