# Prior boundary

- [FlashBlock](https://proceedings.mlr.press/v306/chen26g.html) caches
  block-external attention across diffusion steps. It already identifies
  cross-step attention redundancy. This PoC cannot claim that observation.
- [LoSA](https://proceedings.mlr.press/v306/xi26d.html) combines stable-token
  reuse with sparse attention. This PoC does not approximate masks or reuse
  stale attention outputs for changing queries.
- [Native vLLM DiffusionGemma](https://vllm-project.github.io/2026/06/10/diffusion-gemma)
  already caches committed prefix KV. Its prefix and canvas paths differ.
  Its architecture also mixes sliding and full attention. Those facts cannot
  be transferred blindly to LLaDA2. DiffusionGemma is optional, not a rescue
  requirement after a primary hard stop.
- [Ring Attention](https://arxiv.org/abs/2310.01889) already partitions
  attention work across devices. Multi-GPU attention itself is not novelty.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) uses exact,
  stable softmax accumulation. Distributed max/sum/output merging is established.
  A prefix/canvas split does not introduce new softmax mathematics.
- [Official LLaDA2-mini weights](https://huggingface.co/inclusionAI/LLaDA2.0-mini)
  advertise32K context. [Mini-preview](https://huggingface.co/inclusionAI/LLaDA2.0-mini-preview)
  advertises4K. The loaded original config, not the older experiment setting,
  determines the trained limit here.
- Existing dInfer caches prefix projections. Its native SGLang path still
  creates expanded/cache-scattered KV tensors. We must measure those costs.
  Their presence alone does not establish an attainable multi-GPU speedup.

Potential novelty remains unproven. It would require an exact execution change
that improves physical four-GPU E2E beyond strong native caching and attention.
Structural KV bytes and Amdahl ceilings cannot establish that result.

The source audit pins the existing dInfer checkout and its dirty patch hash.
No external source file is edited. Prior PoC timing on other GPUs is not
substituted for measurements on4/5/6/7.
