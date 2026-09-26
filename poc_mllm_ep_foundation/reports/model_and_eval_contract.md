# Model and evaluation contract

Status: **PASS (contract and image plumbing)**  
Evidence date: 2026-09-24

## Frozen model contract

| Field | Verified value |
|---|---|
| Model | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| Revision | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| Dtype | BF16 |
| Hidden / routed intermediate | 2048 / 768 |
| Routed experts / selected experts | 128 / 8 |
| Decoder layers | 48 |
| Vision patch / spatial merge | 16 / 2 |
| Shared expert | absent |
| Fused gate/up tensor | `[128, 2048, 1536]` |
| Down tensor | `[128, 768, 2048]` |
| Router tensor | `[128, 2048]` |

The values were checked against the local config, the checkpoint tensor index,
and the installed vLLM implementation. No LLaDA2 512-wide or historical
1024-wide calibration is accepted by the new cost-model signature.

## Processor and image contract

- PIL performs EXIF transpose and RGB conversion exactly once.
- The Qwen processor owns resizing; `qwen-vl-utils` does not pre-resize the
  same image.
- The visual placeholder span is checked against
  `t*h*w/(spatial_merge_size**2)`.
- Generated output is sliced after the prompt tokens. The 12-request reference
  and EP runs had zero prompt leaks and zero empty outputs.
- Blank and deliberately wrong-image controls changed the answer as expected;
  therefore the runner did not silently ignore the image.

## Decoding lanes

`OFFICIAL_REPRODUCTION` is pinned to `do_sample=true`, seed 3407,
temperature 0.7, top-p 0.8, sampling top-k 20, repetition penalty 1.0,
presence penalty 1.5, EOS stop, and maximum 32768 new tokens. The bounded
four-request execution capped output at 128 tokens and is explicitly a
protocol deviation, not a reported-score reproduction.

`DETERMINISTIC_PARITY` uses greedy decoding and is reserved for runtime and
trace comparisons. Sampling top-k 20 is unrelated to MoE expert top-k 8.

## Runtime and repository

- Python 3.12.13, PyTorch 2.11.0+cu129, Transformers 5.14.1, vLLM
  0.20.0+cu129.
- The checkpoint's generation metadata records Transformers 4.56.0; this
  metadata/runtime difference is preserved rather than hidden.
- Audit commit: `f0dc8372ec0d1d2e10347c82e630b84430c5dc5f`.
- The worktree already contained 1,611 dirty entries. This package is isolated
  under `poc_mllm_ep_foundation/`; unrelated files were not modified.

Machine-readable source: `artifacts/manifest/environment.json` and
`configs/frozen_protocol.json`.
