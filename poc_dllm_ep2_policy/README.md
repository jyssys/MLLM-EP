# dLLM EP2 denoising-adaptive EP policy PoC

This directory is independent of the earlier EP4 temporal PoC.  It implements
the mechanism-screening contract in
`poc_flashvep/reports/dllm_ep2_denoising_adaptive_ep_policy_poc_spec.md`
(SHA-256 `58635263c9a69c1bd42d73b16691b1a43b0a5ccf1a9d1ad874cd8b7dd19d3d34`).

The live runner refuses any visibility other than physical GPUs 6 and 7.  EP2
results are never promoted beyond `EP2 HOLD-FOR-EP4`.

Final outcome: **EP2 NO-GO**. See
`reports/dllm_ep2_denoising_adaptive_ep_policy_poc.md` and the reproducible
derived artifacts under `results/ep2_20260911_170000/analysis/`.
