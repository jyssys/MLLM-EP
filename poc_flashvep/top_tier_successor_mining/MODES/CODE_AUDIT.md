# MoDES — source audit

Official `ModelTC/MoDES`, commit `933b3ccac3c23e01c763771f1a7d60c4e7ed13a4`.

| File | Behavior / reproduction control |
|---|---|
| `get_layer_importance_ddp.py` | Full teacher logits, answer-token mask, each layer/modality ablation, KL sum divided by token count then normalized across layers. |
| `grid_search_tau_ddp.py` | Caches exact tau pair evaluations; carries frontier pointer across text thresholds; skips infeasible upper endpoint. |
| `models/qwen3.py` | Top-k probabilities renormalized first; layer importance multiplies decision scores only; dropped contribution gets zero weight; no post-drop renormalization. |
| `models/kimi.py` | Separate model adaptation, shared-expert handling and sentinels; must audit independently before cross-model use. |
| `models/utils.py` | Threshold-and-mask helper supports invalid expert IDs; scalar scaling respects token mask. |
| `tasks/gqa.py` | GQA image-ID join, short-answer evaluation, full-answer calibration. |

Current Transformers 5.14.1 has changed router return values and packed expert
weight layout relative to this source. Avoid wholesale replacement of modern
model forward methods. Port only scoring/masking and prove no-op equivalence.

Potential code/paper discrepancy: Qwen's layer skip copies its normalized MLP
input into the skipped output positions, and the decoder then adds its residual.
This differs from zeroing a skipped expert branch. Test both official and intended
ablation semantics before using either to claim a method limitation.

Minimal faithful fast port can use existing grouped GEMM or DeepEP sentinel
support; no new kernel is necessary. HF quality timing is not EP serving timing.
