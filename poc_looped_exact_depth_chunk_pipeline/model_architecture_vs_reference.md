# Checkpoint architecture and published description

| Item | Pinned checkpoint / loaded runtime | Published card |
|---|---|---|
| Stored layers | 48 | 24 |
| Recurrent passes | 4 | 4 |
| Logical attention executions | 192 | Not consistent with the card's 24-layer entry |
| Hidden width | 2048 | 2048 |
| Q/KV heads | 16/16 MHA | MHA |
| FFN width | 5632 | SwiGLU |
| Context | Config permits 65,536 positions. This campaign caps total positions at 32,768. | 32K SFT |
| Early exit | Native vLLM executes all four passes. Config threshold is 1.0. | Threshold 1.0 means all steps. |

The card/config layer discrepancy is a provenance limitation. It is not permission to change the loaded model.
All physical experiments retain the pinned 48-layer weights and 192 cache slots.
No smaller model, reduced recurrence, quantization, or quality-changing exit rule replaces native R4.

The card recommends Transformers 4.54.1 for its HF example.
The existing reference environment uses the checkpoint's cache-fixed code with Transformers 4.57.1.
The optimized native engine uses vLLM 0.26.0 and Transformers 5.16.1.
This campaign does not downgrade either shared environment. The independent implementation check reports actual discrepancies.

The official repository address returned HTTP 404 during this audit.
The project page links to [LoopLM](https://github.com/rkstgr/LoopLM), whose README describes a reimplementation.
The audit pins that linked repository at `8d79e31a871e23291a1afb21bf87d33116e8eb26`.
It is not the original private evaluation harness. This campaign does not substitute its evaluator.
The paper documents an in-house prompt/harness and LLM judge. Its complete evaluator is not reproduced here.
The public-score comparison therefore remains `REFERENCE_PROTOCOL_NOT_IDENTICAL`.

The installed native Ouro class does not declare `SupportsPP`.
Its model forward does not consume incoming intermediate hidden states as a generic PP stage.
The ordinary physical-layer partition is also not the requested recurrent-depth partition.
This audit does not count a PP flag or `make_layers` helper as working pipeline serving.
Generic PP comparisons therefore use physical component calibration and remain `UNVALIDATED` as integrated serving results.

Sources: [pinned configuration](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/config.json),
[pinned card](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/README.md),
[paper evaluation protocol](https://arxiv.org/html/2510.25741v2).
