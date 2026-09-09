"""Layer/MoE-only sampled diagnostic on the validated full observer seams."""
import observer


def install():
    observer.install()
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    # No attention/router CUDA events; retain layers, full forward and DeepEP
    # dispatch/expert/combine timing at explicit sparse sampling intervals.
    Qwen3MoeAttention.forward = Qwen3MoeAttention.forward.__wrapped__
    BaseRouter.select_experts = BaseRouter.select_experts.__wrapped__
    original = GPUModelRunner._model_forward

    def forward(self, *args, **kwargs):
        context = getattr(observer._TLS, "context", None)
        if context and not context["selected"]:
            # Collective ordinal and local DP step can select different samples.
            # Infer phase from existing CPU metadata even on those MoE samples.
            flags = []
            for rid in context["scheduled"]:
                index = self.input_batch.req_id_to_index.get(rid)
                if index is not None:
                    flags.append(int(self.input_batch.num_computed_tokens_cpu[index])
                                 < int(self.input_batch.num_prompt_tokens[index]))
            if flags:
                context["phase"] = "prefill" if all(flags) else "decode" if not any(flags) else "mixed"
        return original(self, *args, **kwargs)
    GPUModelRunner._model_forward = forward
