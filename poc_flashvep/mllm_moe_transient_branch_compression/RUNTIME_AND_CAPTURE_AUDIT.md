# Runtime and capture audit

## Baseline

- Model: Qwen3-VL-30B-A3B-Instruct, BF16.
- Intended topology: TP2 / DP2 / EP4, DeepEP high-throughput, four physical H100 GPUs 4–7.
- Successful live-capture environment: `/home/esjung/.venvs/flashvep-deepep-v020`
  with vLLM `0.20.0+cu129` and DeepEP/NVSHMEM libraries from the pinned v0.2.0
  environment.  Package metadata identifies DeepEP as `1.2.1+73b6ea4`.
  Two earlier attempts from the generic conda environment failed
  before model execution because DeepEP was absent; these are port failures,
  not method results.
- Model snapshot: `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
- DBO: disabled in the established capture harness.

The installed source selects `DeepEPHTPrepareAndFinalize` from
`vllm/model_executor/layers/fused_moe/all2all_utils.py` and the local expert
implementation is `TritonExperts` in `fused_moe.py`. A fresh runtime proof must
confirm DP=2, EP enabled, DeepEP HT, and these concrete classes before new
measurements are admitted.

The fresh proof now exists.  `capture_ep_branches.py` launches two DP engine
processes with `VLLM_DP_SIZE=2`, `tensor_parallel_size=2`,
`enable_expert_parallel=True`, `all2all_backend=deepep_high_throughput`, and
`enable_dbo=False`.  All four worker proof files report EP ranks 0--3,
`ep_world_size=4`, `DeepEPHTAll2AllManager`,
`DeepEPHTPrepareAndFinalize`, `TritonExperts`, and visible devices
`4,5,6,7`.  The two driver outputs are both `ok: true` and returned the same
one-token greedy output for every captured request.

## Capture semantics

The reused capture patches the router and modular fused-expert call without
changing the stock output. For sampled tokens it records top-k expert IDs and
weights, pre-expert hidden input, one raw output per selected expert, and the
stock combined output. Raw branch outputs are reconstructed across the four EP
ranks and validated by recomputing the weighted sum. Observer-heavy captures
are used only for geometry and quality; they are never used as clean timing.

The CPU pilot used the prior 24-image, 25%-token capture.  Fresh evidence now
captures every prompt token for nine images over natural, fine-grained, and
chart/document categories at 336/448/672-pixel edges and layers
4/12/24/36/44/47.  Weighted branch reconstruction is checked independently for
every sample/layer.  A second exact replay loads the released checkpoint expert
weights and reproduces the captured vLLM branch outputs. Across all 27 sampled
request/layer validation rows, median relative-L2 is 0, worst p99 relative-L2
is `1.56e-8`, and minimum cosine is 0.9999983.
