# Environment, model, and runtime audit

## Immutable identifiers

- Model: `inclusionAI/LLaDA2.0-flash`
- Downloaded revision: `744c3f8c6c8317d2377d6d16d8a3d4be2caef563`
- Official dInfer commit: `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f`
- Primary runtime target: dInfer LLaDA2 SGLang path with SGLang `0.5.3.post1`
- GPU allow-list: physical GPUs 0, 1, 2, 3 only
- Isolated Python: `/home/esjung/.venvs/llada2-flash-sglang-053`
- PyTorch/CUDA: `2.8.0+cu128` / CUDA 12.8 runtime
- Transformers: `4.57.0`
- DeepEP package: `1.2.1+73b6ea4`
- Audit-only current SGLang main: `78da62519012a06833ae37ea9a214d101c2963b8`

## Verified checkpoint facts

The downloaded config reports BF16, 32 layers, hidden size 4096, 256 routed
experts, top-8 routing, one shared expert, MoE intermediate size 1024, eight
routing groups with top four groups selected, 32 attention heads, four KV
heads, and maximum context length 32768. Layer 0 is dense and layers 1--31
are sparse. The 42 safetensor shards total 205,779,410,432 bytes.

Safetensor-header accounting gives 102,889,705,216 parameters. Routed experts
alone contribute 99,857,989,632 parameters (199.716 GB BF16, 97.1% of the
checkpoint). Attention is 2.416 GB, embeddings/LM head 1.288 GB, shared experts
0.780 GB, the dense layer MLP 0.226 GB and routers 0.065 GB. This is therefore
a genuine capacity-driven sparse deployment problem: neither a full BF16
single-GPU model nor an all-experts-per-rank implementation fits an 80 GB H100.

## Official 4-GPU path is not true sparse EP

The official benchmark derives `tp_size` from the number of GPUs and defaults
`ep_size` to one. Its `ServerArgs` originally passed `tp_size=4` but did not
pass an MoE A2A backend. Therefore the documented four-GPU command is TP4;
four GPUs alone are not evidence of EP4.

The model does contain a distinct true sparse path. When SGLang's MoE A2A
backend is `deepep`, `LLaDA2SparseMoeBlock.forward_deepep` computes router
top-k, invokes `DeepEPDispatcher.dispatch`, executes only local fused experts,
and invokes `DeepEPDispatcher.combine`. With TP-world size four this path
forces EP size four and assigns 64 routed experts per rank. This still requires
live runtime/profiler validation before it is accepted as the headline EP4
substrate.

The bounded EP path is more precisely **dense TP4 + routed-expert EP4**. Dense
attention, the dense first layer, LM head and other dense projections retain
the official TP4 placement. In sparse layers the replicated logical token
matrix is row-partitioned, each rank routes its rows, DeepEP normal/HT mode
dispatches to 64 owner-local experts, BF16 Triton fused MoE executes locally,
DeepEP combines, and the source-token rows are gathered for the next dense
TP4 operation. DP is one and no sequence parallelism is enabled.

The 64-expert ownership is contiguous: rank 0 owns global experts 0--63,
rank 1 owns 64--127, rank 2 owns 128--191 and rank 3 owns 192--255. The one
shared expert is replicated in full on each rank in the EP path; it is not
sent through DeepEP. This is an explicit design choice needed because routed
EP and dense TP share the same four-process group in the official manual
runner.

## Minimal substrate fixes

The official SGLang checkpoint loader computed a rank-local filtered state
dictionary but then accidentally retained the unfiltered shard. The local
audit clone changes this one statement to retain only rank-local expert
weights. A second loader defect gated routed-expert loading on the presence of
global expert zero; consequently only rank 0 loaded any routed experts. The
bounded fix tests whether the rank-local expert dictionary is nonempty. This
is a substrate correctness repair, not a method optimization. The benchmark
clone also exposes SGLang's DeepEP backend/mode and CUDA-graph switch.

Live evidence after the fix:

- the loader sees 6,305 tensors/rank in EP4 versus 24,161/rank in TP4;
- a DeepEP identity test dispatches 75.1% of assignments remotely and every
  receiver observes only local IDs 0--63;
- median primitive dispatch/combine is 0.1625/0.2435 ms;
- layer 1 TP4 versus EP4 routed output relative L2 is 0.373% against an
  independent checkpoint reference;
- final layer hidden relative L2 is 4.16%, cosine 0.999146, attributable to
  repeated BF16 reduction-order differences rather than missing experts;
- all eight bounded GSM8K answers remain benchmark-equivalent in the initial
  topology check.

Current SGLang main was audited separately. It has newer native dLLM/FDFO and
DeepEP support, but its tested main requires the CUDA-13-era torch/kernel stack
that is incompatible with this host's driver/toolchain. The executable
official dInfer 0.5.3 substrate is therefore used rather than silently calling
an environment failure a method result.

## Live-gate disposition

All five substrate gates passed. The isolated environment executes both TP4
and true routed EP4; ownership, remote dispatch, owner-local execution and
combine were observed; bounded GSM8K/HumanEval scores match; and both paths fit
at about 57 GiB/rank without CPU expert offload. Performance conclusions and
observer-tax boundaries are reported separately.
