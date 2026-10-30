"""One-variable diagnostic execution variants. The default runtime is untouched."""
import hashlib
import os
from pathlib import Path


def install():
    variant = os.environ.get('B_REPAIR_VARIANT')
    if not variant:
        return
    import torch
    from dataclasses import asdict, replace
    from ..looped_pd_abc import worker
    from .common import write
    if getattr(worker, '_B_repair_installed', False):
        return
    worker._B_repair_installed = True
    accepted = ('graph_off', 'async_off', 'streams_off', 'bf16_reduction_off',
                'fixed_decode_core', 'head_fixed', 'decode_core_and_head', 'all_linear_fixed', 'canonical_all')
    if variant not in accepted:
        raise ValueError('Unregistered diagnostic repair variant: ' + variant)
    original = worker.configuration

    def configuration(model, width, chunk, graphs=False, asynchronous=False):
        if variant == 'graph_off':
            graphs = False
        if variant == 'async_off':
            asynchronous = False
        value = original(model, width, chunk, graphs, asynchronous)
        if variant == 'streams_off':
            value['execution_config'] = replace(value['execution_config'], multi_stream=False)
        folder = os.environ.get('B_REPAIR_RUN_DIR')
        if folder:
            write(Path(folder) / f'repair_effective_configuration_{os.getpid()}.json', dict(
                variant=variant, execution=asdict(value['execution_config']),
                scheduler=asdict(value['scheduler_config']), cache=asdict(value['cache_config']),
                exit=asdict(value['exit_config'])))
        return value

    worker.configuration = configuration
    if variant == 'bf16_reduction_off':
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    usage = dict(core_bucket_calls=0, boundary_bucket_calls=0, fixed_head_python_calls=0, fixed_linear_python_calls=0,
                 scope='Python construction/select calls, including capture. Not a kernel replay count.')
    if variant in ('fixed_decode_core', 'decode_core_and_head', 'canonical_all'):
        from vllm_rlt.worker import cuda_graph
        from vllm_rlt.config import cuda_graph_rows
        original_select = cuda_graph.RecurrentGraphs.select
        def fixed_select(self, hidden, batch):
            # Packed prompt attention retains its original path. Only independent
            # single-query rows use the existing masked, bucketed graph implementation.
            if batch.cu_seqlens_q is not None:
                return original_select(self, hidden, batch)
            self.capture_sizes = (cuda_graph_rows(self.execution),)
            usage['core_bucket_calls'] += 1
            return cuda_graph.DeviceRecurrentGraphs.select(self, hidden, batch)
        cuda_graph.RecurrentGraphs.select = fixed_select
    if variant == 'canonical_all':
        from .repair_coda import install as coda_install
        coda_install(usage)
    if variant in ('head_fixed', 'decode_core_and_head'):
        import torch.nn.functional as functional
        original_linear = functional.linear
        def fixed_head_linear(inputs, weight, bias=None):
            if inputs.ndim != 2 or not 0 < len(inputs) <= 8 or weight.shape[0] not in (49152, 65536):
                return original_linear(inputs, weight, bias)
            usage['fixed_head_python_calls'] += 1
            padded = inputs.new_zeros((8, inputs.shape[-1]))
            padded[:len(inputs)].copy_(inputs)
            return original_linear(padded, weight, bias)[:len(inputs)]
        functional.linear = fixed_head_linear
    if variant in ('all_linear_fixed', 'canonical_all'):
        import torch.nn.functional as functional
        original_linear = functional.linear
        def fixed_linear(inputs, weight, bias=None):
            if inputs.ndim != 2 or len(inputs) == 0:
                return original_linear(inputs, weight, bias)
            usage['fixed_linear_python_calls'] += 1
            block = 8 if len(inputs) <= 8 else 256
            outputs = []
            for offset in range(0, len(inputs), block):
                rows = inputs[offset:offset + block]
                padded = inputs.new_zeros((block, inputs.shape[-1]))
                padded[:len(rows)].copy_(rows)
                outputs.append(original_linear(padded, weight, bias)[:len(rows)])
            return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        functional.linear = fixed_linear
    folder = os.environ.get('B_REPAIR_RUN_DIR')
    if folder:
        write(Path(folder) / f'repair_override_{os.getpid()}.json', dict(
            variant=variant, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            repair_sources_SHA256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ('repair_hooks.py', 'repair_coda.py')},
            BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            weights_and_tensor_dtype='unchanged BF16', fixed_depth='unchanged',
            scope='Diagnostic variant applied to both control and B. No performance or quality acceptance.'))
        from multiprocessing.util import Finalize
        Finalize(None, lambda: write(Path(folder) / f'repair_usage_{os.getpid()}.json', usage), exitpriority=15)
