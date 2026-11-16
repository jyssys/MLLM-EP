"""Use the standard TP worker with the existing exact normalization dispatch."""
import json
import os

from vllm.v1.worker.gpu_worker import Worker


class ExactTPWorker(Worker):
    def init_device(self):
        super().init_device()
        from .common import bind_previous
        bind_previous()
        from ..dense_ar_slo_layer_pd.phase_stable_norm import install
        config = json.loads(open(os.environ['DENSE_CONFIG']).read())
        role = 'tp_rank' + str(self.rank)
        config = dict(config, phase_mask_norm_roles=(role,))
        install(config, role)

    def duetserve_report(self):
        import torch
        runner = self.model_runner
        return dict(rank=self.rank, local_rank=self.local_rank,
            pid=os.getpid(), session=os.getsid(0),
            uuid=str(torch.cuda.get_device_properties(self.device).uuid),
            parameter_elements=sum(p.numel() for p in runner.model.parameters()),
            weight_bytes=sum(p.numel() * p.element_size() for p in runner.model.parameters()),
            KV_blocks=runner.kv_cache_config.num_blocks,
            KV_tensor_bytes=sum(t.numel() * t.element_size() for t in runner.kv_caches),
            compile_mode=str(self.vllm_config.compilation_config.mode),
            cudagraph_mode=str(self.vllm_config.compilation_config.cudagraph_mode),
            tensor_parallel_size=self.parallel_config.tensor_parallel_size,
            BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            KV_dtype=str(runner.kv_cache_dtype),
            scope='Actual TP rank storage and configuration. No communication time is inferred.')
