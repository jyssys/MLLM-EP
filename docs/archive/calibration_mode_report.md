# Calibration / MODE Reference Report

Date: 2026-06-24

## Scope

This report covers the Phase 1 calibration-statistics update requested after the initial implementation. The code remains CPU-only, pure PyTorch, and dummy-input driven. No real Qwen3-VL forward pass, DeepSpeed execution, quantization workflow, speed measurement, or accuracy evaluation was added.

## MODE Sources Reviewed

- MODE repository: https://github.com/MingZwhy/MODE
- Frequency recorder: https://github.com/MingZwhy/MODE/blob/main/mllm_quant/moe_freq/record_freq.py
- Calibration data guide: https://github.com/MingZwhy/MODE/blob/main/docs/data.md
- Calibration data preparation script: https://github.com/MingZwhy/MODE/blob/main/scripts/prepare_calib_data.sh
- Qwen3-VL-MoE adapter: https://github.com/MingZwhy/MODE/blob/main/mllm_quant/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py

Quantization-only files such as sensitivity collection, ILP bit allocation, and quantize scripts were intentionally ignored.

## What Was Adopted

`calib/collect_stats.py` now mirrors MODE's frequency-recorder design at the tensor-logic level:

- Counts expert routing frequency by modality: total, text, image/vision.
- Splits image tokens into dominant/key and redundant groups from attention.
- Supports MODE-style `attn_mode="adaptive"` as the default, with `dominant_ratio=0.2` for top-20% key vision tokens.
- Excludes pre-image prompt tokens and optional special/template token ids from text-to-image attention scoring.
- Treats redundant image routing stats as all non-dominant image tokens, matching MODE's actual MoE hook aggregation behavior.
- Adds expert centroids by accumulating hidden-state means for every routed expert. MODE tracks frequency only; this centroid path is our MACS-style addition for future redundant-token analysis.

The real hook structure is represented as metadata through `build_mode_hook_plan()`: input/generate capture, per-layer self-attention capture, and MoE block routing capture. It deliberately does not register hooks against an actual model in Phase 1.

## Calibration Data Format

`load_sharegpt4v_records()` and `resolve_calibration_image_path()` support the ShareGPT4V/COCO-style schema used by MODE:

```json
{
  "image": "coco/train2017/000000000009.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion"},
    {"from": "gpt", "value": "Answer"}
  ]
}
```

Image paths are resolved relative to a caller-provided calibration image root, matching the MODE convention of `--calib_image_folder data`.

## Qwen3-VL-MoE Hook Notes

`docs/model_arch.md` was updated with MODE-specific observations:

- MODE's Qwen3-VL-MoE sparse block uses a bias-free `gate`, then softmax/top-k routing.
- The sparse block returns router logits, and MODE can recompute routing from `gate(hidden_states)` if needed.
- MODE's adapter also uses interleaved M-RoPE via `mrope_section` / `apply_interleaved_mrope`, consistent with the Hugging Face architecture notes.

## Tests Added

New dummy tests cover:

- Adaptive attention split with `dominant_ratio=0.2`.
- Exclusion of pre-image/system text and special tokens from key-token scoring.
- Redundant tokens as all non-key vision tokens.
- Attention-driven calibration aggregation, including key/redundant expert counts and centroids.
- `collect_calibration_stats()` with attention payloads.
- ShareGPT4V calibration record loading and image-path resolution.

Verification:

```text
python3 -m pytest -q
22 passed in 1.26s
```

## Phase 2 Boundaries Preserved

The following remain TODO-only or metadata-only:

- Real Qwen3-VL forward hooks.
- GPU execution.
- de-RoPE and CLS importance terms.
- Redundant-token rerouting.
- DeepSpeed expert-parallel integration.
- Speedup, throughput, accuracy, and hyperparameter tuning.
