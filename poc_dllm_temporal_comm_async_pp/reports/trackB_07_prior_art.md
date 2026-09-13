# Track B7 — prior-art attack

- [AsyncDiff](https://arxiv.org/abs/2406.06911) already partitions diffusion
  models across devices and breaks sequential denoising dependencies by using
  temporally similar hidden state.  It reports 2.7x on four A5000 GPUs with
  negligible image-quality degradation.  “Apply AsyncDiff to dLLM” is not a
  sufficient novelty claim.
- [PipeFusion](https://arxiv.org/abs/2405.14430) pipelines layer/patch work and
  uses one-step stale features across diffusion steps.  It is a direct
  mechanism neighbor even though the data modality and scheduling units differ.
- [DICE](https://openaccess.thecvf.com/content/ICCV2025/html/Luo_DICE_Staleness-Centric_Optimizations_for_Parallel_Diffusion_MoE_Inference_ICCV_2025_paper.html)
  is closer still: MoE diffusion, interweaved parallelism, layer-selective
  synchronization, and conditional communication.  It directly attacks
  staleness/overlap and reports 1.26x.
- [ParaStep](https://arxiv.org/abs/2505.14741) uses reuse-then-predict to
  parallelize diffusion denoising with step-wise communication.
- The current [vLLM LLaDA2 operator documentation](https://github.com/vllm-project/dllm-plugin/blob/main/docs/OPERATOR_LLaDA2.md)
  explicitly leaves PP unsupported, so a correct language-dLLM PP substrate is
  an engineering gap but not by itself a new inference principle.

A defensible residual research question would couple discrete unmask/acceptance
trajectory, boundary-specific error amplification, periodic exact reset, and a
hybrid PPxEP topology.  Our data establish the error structure but not a strong
speed/quality frontier.  The prior-art risk is high and the measured gate is
weak.
