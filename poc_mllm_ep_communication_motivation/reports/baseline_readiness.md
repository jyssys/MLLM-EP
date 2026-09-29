# Baseline readiness

Status: **PASS**

- Model revision: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`; BF16; shapes `{"hidden_size": 2048, "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8, "num_hidden_layers": 48}`.
- Physical GPUs: 2,3,6,7; topology `TP2-DP2-EP4-PP1`; UUIDs are retained in the machine-readable audit.
- Backend: TRUE sparse EP4 with `deepep_high_throughput`, linear expert placement, 32 experts/rank.
- Primary controls: chunked prefill off; `max_num_batched_tokens=20000`; `max_num_seqs=5`; DBO off; EPLB/redundancy off; prefix cache off; eager execution.
- Actual DeepEP prepare/finalize and sparse-expert calls are timed with deferred CUDA events. Logical BF16 payload accounting follows the installed backend's token-destination dedup contract.
- This is a motivation study; no communication method, scheduler policy, or speedup implementation is included.
