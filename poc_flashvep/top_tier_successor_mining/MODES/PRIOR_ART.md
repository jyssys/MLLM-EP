# MoDES adversarial screen

[MoDES](https://arxiv.org/abs/2511.15690) already establishes modality/layer-aware
skipping across Qwen/Kimi and other MLLMs. Generic MLLM transfer, early-layer
sensitivity and different Vision/Text budgets are not successor novelty.

[AnyExperts](https://arxiv.org/abs/2511.18314) explicitly addresses semantic
importance and stronger OCR/NLP sensitivity, with trained variable expert budgets.
[MACS](https://arxiv.org/abs/2605.05225) covers semantic/modal capacity allocation
and rerouting in multimodal EP. Training-free versus trained is an implementation
distinction, not by itself a sufficient independent research mechanism.

The remaining question is a material failure after faithful calibration and
simple recalibration/threshold controls, with a measured quality-efficiency gap.
The current pilot and small monotonicity reversals do not establish it.
