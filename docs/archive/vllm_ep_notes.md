# vLLM Expert Parallel Notes

Date: 2026-06-25

## Installed Version

- Package: `vllm`
- Version: `0.20.0+cu129`
- Import version: `0.20.0`
- Install path: `/usr/local/lib/python3.12/dist-packages/vllm`

## Qwen3-VL-MoE Implementation Path

Local source files inspected:

- `vllm/model_executor/models/qwen3_vl_moe.py`
- `vllm/model_executor/models/qwen3_moe.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`
- `vllm/model_executor/model_loader/default_loader.py`

Qwen3-VL-MoE uses the vLLM model implementation
`Qwen3VLMoeForConditionalGeneration`. The language model reuses Qwen3-MoE
blocks:

```text
Qwen3-VL-MoE
  -> Qwen3MoeSparseMoeBlock
  -> ReplicatedLinear gate
  -> FusedMoE experts
```

The sparse block constructs `FusedMoE` with:

- `num_experts=config.num_experts`
- `top_k=config.num_experts_per_tok`
- `intermediate_size=config.moe_intermediate_size`
- `renormalize=config.norm_topk_prob`
- `gate=self.gate`
- `enable_eplb=parallel_config.enable_eplb`

For Qwen3-VL-30B-A3B-Instruct this means:

- layers: 48
- global experts: 128
- top-k: 8
- placement target for this phase: linear expert placement, 16 experts per rank

## Expert Parallel Settings

vLLM 0.20 exposes the following relevant flags:

- `--enable-expert-parallel` / `-ep`
- `--expert-placement-strategy {linear,round_robin}`
- `--all2all-backend {allgather_reducescatter,deepep_high_throughput,deepep_low_latency,flashinfer_all2allv,flashinfer_nvlink_one_sided,flashinfer_nvlink_two_sided,mori,naive,nixl_ep,pplx}`
- `--enable-ep-weight-filter`
- `--enable-eplb`
- `--enable-return-routed-experts`

For vanilla measurement, do not enable EPLB. Use static linear placement:

```bash
--tensor-parallel-size 8 \
--enable-expert-parallel \
--expert-placement-strategy linear \
--all2all-backend allgather_reducescatter \
--enable-return-routed-experts
```

For the Python `LLM` API, the equivalent constructor arguments are:

```python
LLM(
    model="models/Qwen3-VL-30B-A3B-Instruct",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    expert_placement_strategy="linear",
    all2all_backend="allgather_reducescatter",
    enable_return_routed_experts=True,
)
```

Important nuance: in vLLM's `FusedMoEParallelConfig`, EP is formed by flattening
the active parallel ranks. With `tensor_parallel_size=8` and
`enable_expert_parallel=True`, the MoE layers use `EP={8, rank}` and TP is
effectively disabled inside MoE. This is the practical single-node offline API
path for 8-way EP. A pure attention-DP / MoE-EP setup would use data-parallel
ranks, but the local offline `LLM` constructor rejects single-process DP usage;
that path is more natural through `vllm serve` with DP workers.

## EP Sharding Verification Points

`FusedMoE.determine_expert_map()` implements static expert placement.
For `ep_size=8`, `global_num_experts=128`, `expert_placement_strategy="linear"`:

- rank 0 owns experts 0-15
- rank 1 owns experts 16-31
- rank 2 owns experts 32-47
- rank 3 owns experts 48-63
- rank 4 owns experts 64-79
- rank 5 owns experts 80-95
- rank 6 owns experts 96-111
- rank 7 owns experts 112-127

The model loader has an optional `enable_ep_weight_filter` path that skips
non-local expert tensors before reading from disk. The CLI documents that this
has no effect on 3D fused-expert checkpoints, but Qwen3-VL-MoE stores packed
expert tensors in the HF checkpoint and vLLM remaps them into FusedMoE expert
weights during loading. The hard gate remains empirical: rank memory must be
well below the full replicated DeepSpeed run (`57.94 GB` per rank).

## Routing Extraction Path

vLLM 0.20 has a built-in routed expert capture path:

- constructor flag: `enable_return_routed_experts=True`
- CLI flag: `--enable-return-routed-experts`
- worker: `init_routed_experts_capturer()`
- module: `RoutedExpertsCapturer`
- output field: `CompletionOutput.routed_experts`

The output shape is documented in `vllm/outputs.py`:

```text
routed_experts: np.ndarray | None  # [seq_len, layer_num, topk]
```

This captures top-k expert ids only. It does not expose router softmax scores.
For Phase 2-A:

- P1 trace can use `routed_experts` directly.
- M2 per-expert/per-rank load can use `routed_experts` directly.
- P2 gating scores still require a separate vLLM internal hook or a slower HF
  single-rank simulation path to recover router logits. Do not patch fused MoE
  kernels for this phase unless EP sanity has passed.

During the Phase 2-A profiling run, the local vLLM 0.20 source was checked
again for gating-score exposure. The built-in capture path is:

```text
RoutedExpertsCapturer.capture(layer_id, topk_ids)
CompletionOutput.routed_experts  # [seq_len, layer_num, topk]
```

`topk_weights` and router softmax values exist inside the fused MoE routing path,
but they are not exported through `CompletionOutput` or the routed-expert shared
memory buffer. Therefore `outputs/calibration/gating_score_todo.md` records P2
as TODO for a deeper vLLM hook or Phase 2-B implementation. P1/P3 were completed
with actual vLLM EP routed expert ids and no kernel patching.

## Gate Status

Step 1 confirms that vLLM has native Qwen3-VL-MoE `FusedMoE` and EP support, and
that routed expert ids can be returned without custom kernel hooks. Step 2 must
prove that the model actually loads as 8-way EP by rank memory and can run a
dummy multimodal prefill without OOM/NCCL errors.

## Runtime Notes From Sanity

The initial routed-experts attempt with default KV cache sizing failed during
engine initialization after `RoutedExpertsCapturer` startup. Root cause is
consistent with the capture buffer scaling with total KV-cache slots:

```text
shape = [max_num_kv_tokens, 48 layers, topk 8] int32
```

With automatic KV allocation, vLLM reported about 2.5M KV tokens, which implies
multi-GB shared memory for routed-expert capture. For profiling/sanity runs,
set a bounded KV cache:

```bash
--kv-cache-memory-bytes 1073741824
```

With this bound, vLLM reported 43,680 KV tokens and routed-experts capture
initialized successfully.

Confirmed 8-way EP sanity command shape:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 scripts/vllm_ep_sanity.py \
  --kv-cache-memory-bytes 1073741824 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --output outputs/vllm_ep_sanity.json
```

Observed gate evidence:

- vLLM worker names: `Worker_TP0_EP0` ... `Worker_TP7_EP7`
- vLLM log: `EP Rank 0/8`, `Local/global number of experts: 16/128`
- vLLM loader log: `EP weight filter: ep_size=8, ep_rank=0, loading 16/128 experts`
- model memory: about `11.8 GiB` per rank, far below the earlier DeepSpeed full
  replica level of `57.94 GiB`
- routed-experts output: `[79, 48, 8]`, int32, ids in `[0, 127]`
