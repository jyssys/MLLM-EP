# Runtime and ownership audit

The target is the **standalone text MoE Qwen/Qwen3-30B-A3B**, not Qwen3-VL. Its local frozen revision is `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. The draft is the existing-runtime-compatible dense Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`. No model was trained.

The original `/home/esjung/MLLM-EP-github` has unrelated staged/dirty work. An independent worktree and new branch were created without editing its index or working files. The zero-new-commit branch was based on `dd73f832e5bd0e835f8d5bd4a6e94e82ff5b1ab2`, containing the validated native TRUE-EP infrastructure. Existing root HEAD `f0dc8372` alone lacked those committed PoC assets. This baseline choice is explicit; no user branch was reset.

## Actual software and hardware

GPU venv: `/home/esjung/.venvs/flashvep-deepep-v020/bin/python`. Package metadata: vLLM **0.20.0+cu129**, Torch **2.11.0+cu129**, CUDA **12.9**, NCCL **2.28.9**, DeepEP **1.2.1+73b6ea4**, Triton **3.6.0**, Transformers **5.14.1**. DeepEP source SHA is `73b6ea4a439ba03a695563f9fd242c8e4b02b37c`. The initial environment JSON records `vllm.__version__=0.20.0` and unavailable `deep_ep.__version__`; package metadata resolves those fields without modifying the initial artifact. No runtime upgrade/install was performed. CPU analysis uses the already installed Anaconda base packages.

| Physical GPU | Logical CUDA / EP rank | UUID | GPU |
|---|---|---|---|
| 2 | 0 | GPU-24200107-8a7f-de46-1bc8-b81f8d3af13e | H100 80GB HBM3 |
| 3 | 1 | GPU-17488c15-2d4c-5d9e-d503-29b0d959a8a8 | H100 80GB HBM3 |
| 6 | 2 | GPU-e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 | H100 80GB HBM3 |
| 7 | 3 | GPU-4cc26b88-19fc-1988-f9e0-17858aa7a99b | H100 80GB HBM3 |

`CUDA_VISIBLE_DEVICES=2,3,6,7`. Each GPU reports 81,559 MiB. `nvidia-smi topo -m` reports NV18 links between every selected pair on the shared eight-GPU fabric. Other GPU processes are read-only audit targets and never terminated. No pure NVLink bandwidth claim is made.

## Loaded target and topology

- Actual config: 48 sparse layers, d=2048, MoE m=768, 128 routed experts, K8, normalized selected routing weights, SiLU, BF16.
- Loaded local expert tensors on every EP rank: w1 `[32,1536,2048]`, w2 `[32,2048,768]`, BF16: **9,437,184 bytes per expert**. Runtime expert maps are asserted against contiguous linear32 placement, not inferred solely from the config.
- TRUE TP2/DP2/EP4/PP1: TP groups [0,1] and [2,3]; DP peer groups [0,2] and [1,3]; EP all four. TP×DP is four GPUs; EP is not an extra multiplier.
- Actual backend assertions: `DeepEPHTPrepareAndFinalize` and `TritonExperts`; silent backend fallback aborts capture. Expert owner proofs and global group records are retained in raw/exported evidence.
- EPLB off; redundant experts zero; DBO off; prefix caching off; eager; async scheduling off; Flash Attention; chunked prefill off; max_num_seqs=1; max_num_batched_tokens=4096; max_model_len=4096; KV reservation 2 GiB.
- One real request on DP0, DP1 runtime dummy execution. DP1 is **not a replicated user request**. User-only and executed/dummy/padded counters are separate, and M1 sees actual executed expert work.
- Native draft TP2 equals target TP2 because the installed DraftModelProposer requires it. No draft process uses another physical GPU.

## Measurement boundaries and repairs

The existing external-DP wave driver, scoped GPU transaction, port selection and destination-deduplicated route accounting are reused, unchanged. New hooks are scoped to this independent PoC. Natural greedy generation uses EOS and at most64 tokens in the discovery/physical runs, 16 in initial smoke; no 32768 cap.

Initial failures were concrete and repaired: GPU venv lacks pyarrow, so parquet preprocessing uses the existing CPU interpreter; this Transformers chat-template call returns BatchEncoding, so input_ids are explicitly extracted; callable worker RPC is rejected by installed vLLM serialization policy, so a registered **string RPC method** drains events. No unsafe serialization option/environment upgrade was added. Failed versioned logs remain local and compact failure evidence is exportable.

Deferred CUDA events bracket target `_prepare`, `_fused_experts`, `_finalize`. These are dispatch/expert/combine **runtime composites**, not isolated wire transfer. Full router projection is not independently timed: router-selection events are reported with that exact label. One rank maximizing cumulative48-layer MoE duration supplies **all** D/E/C components. The complete target forward separately uses one maximum complete-forward envelope, ending after final norm/LM head and before sampling; component maxima from different ranks are never added.

There is no per-layer global host synchronization or `.item()` for event collection. Route clones are diagnostic-only; CPU downloads/synchronize happen in a flush RPC after the wave. Clean has no route dump, bypass no timing/route hook work. GPU events can include runtime producer gaps/stream waits; **Dispatch+Combine is not a decomposition into pure payload movement**.

The first implementation only recorded DP-wave drain wall. Native RequestOutput completion can happen sooner. The repaired boundary is wave-ready before admission → actual finished RequestOutput, including prefill/draft/controller but excluding subsequent DP drain and flush RPC. Old AR/gamma2 trace request E2E is N/A, never filled with DP-wave wall.

Every GPU launch stops only the matching user-owned burner, rejects competing selected-GPU processes and has bounded process-group cleanup and restoration. The server's burner script defaults to **0,1,4,5**, so this PoC explicitly supplies `VLLM_UTILIZE_GPUS=2,3,6,7` both for transactions and final restoration.
