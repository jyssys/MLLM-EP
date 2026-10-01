# Pinned upstream semantics and port contract

Reference: [scale-snu/layered-prefill](https://github.com/scale-snu/layered-prefill/tree/053f80e5201a7c0ab56e468e3d578907a2ca9cc3),
commit `053f80e5201a7c0ab56e468e3d578907a2ca9cc3`.
README, scheduler.py, model_runner.py and qwen3_moe.py were inspected before coding.
The upstream model uses TP; it does not supply this TRUE EP4 port.

The upstream scheduler maintains per-sequence stage queues. It selects decode
every iteration and advances a prefill sequence through contiguous layer groups.
The runner retains both hidden and residual tensors. Its model executes decode
alone before/after the active prefill range and a mixed batch inside that range.
We preserve this order in the installed vLLM Qwen3-VL model, using the real paged
attention/KV cache and DeepEP kernels. Intermediate prefill outputs live on GPU
between engine iterations. The scheduler defers prefill sampling until the last
stage, leaving decode sampling enabled each iteration.

The 16-cap auto rule uses 1/2/4/8/16 stages at <=512/1024/2048/4096/otherwise
scheduled prefill tokens. For 2029 prefill tokens the exact upstream rule gives
**4**, not 8, stages. Controlled G4/G8/G16 partitions are separate arms.
All ranks execute 48 decoder-layer MoE collectives in the same order: inactive
prefill ranges contain only decode rows; active ranges contain full prefill rows
plus decode. DP token-size metadata changes with that row selection. No token
chunking is used to implement LP.

Implementation scope: `CONTROLLED_NATIVE_LP_EP4_HARNESS`. Native stage execution
must pass correctness before any serving-latency results are accepted. No
post-hoc Layered replay is an admissible substitute.
