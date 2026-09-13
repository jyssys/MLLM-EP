# Track A6 — prior-art attack

The broad idea is already close to existing diffusion communication work.

- [CompactFusion](https://arxiv.org/abs/2507.17511) explicitly transmits
  compressed step-wise activation residuals and adds error feedback for
  parallel diffusion inference.  This directly collides with novelty framed as
  “send delta instead of full activation.”
- [DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html)
  already exploits temporal activation/routing consistency in diffusion MoE
  and introduces conditional communication.
- Generic activation/MoE communication quantization covers the FP8/INT8
  encoding ingredient.
- TEAM/Epoch remove or avoid work; they are not the same transport protocol,
  but they reduce the remaining communication mass and make this result less
  durable.

The residual gap would have been a destination-rank cache-coherence protocol
under changing dLLM MoE routes, with trajectory validation.  That gap is
mechanistically distinct, but the measured request economics eliminate it
before novelty matters.

Final prior-art conclusion: high collision risk plus sub-1% gross E2E
headroom.  Do not continue Track A on this substrate.
