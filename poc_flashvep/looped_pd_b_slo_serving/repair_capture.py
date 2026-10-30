"""Capture actual target-request operands. This diagnostic changes timing only."""
import os
from pathlib import Path


def install():
    target = os.environ.get('B_REPAIR_CAPTURE_REQUEST')
    if not target:
        return
    import torch
    from multiprocessing.util import Finalize
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.worker.executor import ModelRuntime
    if getattr(ModelRuntime, '_B_capture_installed', False):
        return
    ModelRuntime._B_capture_installed = True
    records = []
    weights = {}
    prefixes = []
    seed = int(os.environ['B_REPAIR_CAPTURE_SEED']) if 'B_REPAIR_CAPTURE_SEED' in os.environ else None
    matching_ids = []
    execute = ModelRunner._execute_batch
    forward = ModelRuntime.forward

    def traced_execute(self, batch, prepared):
        self.runtime._B_current_batch = batch
        try:
            return execute(self, batch, prepared)
        finally:
            self.runtime._B_current_batch = None

    def traced_forward(self, batch):
        context = getattr(self, '_B_current_batch', None)
        selected = []
        row = 0
        if context is not None:
            for span in context.spans:
                request = span.request
                if seed is not None and request.sampling_params.seed == seed and request.request_id not in matching_ids:
                    matching_ids.append(request.request_id)
                # PD workers address requests by transfer UUID. The first same-seed
                # UUID belongs to warmup. The second belongs to the measured trace.
                selected_id = matching_ids[1] if len(matching_ids) >= 2 else None
                if request.request_id == target or (seed is not None and request.request_id == selected_id):
                    last = span.token_start + span.token_count - 1
                    step = last - len(span.request.prompt_token_ids) + 1
                    if step >= int(os.environ.get('B_REPAIR_CAPTURE_MIN_STEP', '-1000000')):
                        selected.append((row + span.token_count - 1, step, span.depth))
                row += span.token_count
        saved_inputs = [(row, step, depth, batch.hidden[row].detach().clone() if batch.hidden is not None else None)
                        for row, step, depth in selected]
        if os.environ.get('B_REPAIR_CAPTURE_KV') == '1' and context is not None:
            for span in context.spans:
                if (len(matching_ids) >= 2 and span.request.request_id == matching_ids[1]
                        and span.depth == 2 and context.operation.value == 'recurrent'
                        and span.token_start - len(span.request.prompt_token_ids) + 1 == 125):
                    cache = self.cache
                    allocation = cache._get_allocation(span.request.request_id)
                    blocks = allocation.block_tables[cache._plane(span.depth)]
                    block_count = (span.token_start + cache.block_size - 1) // cache.block_size
                    indices = torch.tensor(blocks[:block_count], device=cache.device, dtype=torch.long)
                    tensors = []
                    for tensor in (cache.key_cache, cache.value_cache):
                        pages = tensor.index_select(0, indices).permute(1, 0, 2, 3, 4)
                        tensors.append(pages.reshape(cache.num_layers, -1, *pages.shape[-2:])[:, :span.token_start].clone())
                    prefixes.append(dict(depth=span.depth, position=span.token_start,
                                         prompt_length=len(span.request.prompt_token_ids),
                                         key=tensors[0], value=tensors[1],
                                         scope='Actual causal prefix before the current R3 write. Current position excluded.'))
        result = forward(self, batch)
        for row, step, depth, inputs in saved_inputs:
            records.append(dict(operation=context.operation.value, step=step, depth=depth,
                                loop_count=context.loop_count, row=row, rows=context.num_rows,
                                requests=[span.request.request_id for span in context.spans],
                                input_hidden=inputs,
                                hidden=result.hidden[row].detach().clone() if result.hidden is not None else None,
                                logits=result.logits[row].detach().clone() if result.logits is not None else None))
            if result.logits is not None:
                weights['lm_head'] = self.model.lm_head.weight.detach()
        return result

    ModelRunner._execute_batch = traced_execute
    ModelRuntime.forward = traced_forward

    def finish():
        folder = Path(os.environ['B_REPAIR_RUN_DIR'])
        from .common import write
        write(folder / f'capture_status_{os.getpid()}.json', dict(records=len(records), matching_ids=matching_ids,
                                                               target_external_id=target, target_seed=seed))
        if not records:
            return
        torch.cuda.synchronize()
        if 'lm_head' in weights:
            # Reuse the actual target hidden row and original resident BF16 weight.
            # This is an operator diagnostic after generation, not a serving run.
            hidden = next(row['input_hidden'] for row in reversed(records) if row['logits'] is not None)
            weight = weights['lm_head']
            baseline = torch._C._nn.linear(hidden.reshape(1, -1), weight)[0]
            diagnostic = []
            for size in range(1, 9):
                inputs = hidden.new_zeros((size, hidden.numel()))
                inputs[0].copy_(hidden)
                actual = torch._C._nn.linear(inputs, weight)[0]
                fixed = torch.nn.functional.linear(inputs, weight)[0]
                top, ids = actual.float().topk(2)
                diagnostic.append(dict(rows=size, builtin_exact_to_one_row=bool(torch.equal(actual, baseline)),
                                       builtin_max_absolute=float((actual.float() - baseline.float()).abs().max()),
                                       patched_exact_to_one_row=bool(torch.equal(fixed, baseline)),
                                       builtin_top1=int(actual.argmax()), builtin_top_margin=float(top[0] - top[1])))
            write(folder / f'actual_head_geometry_{os.getpid()}.json', dict(
                variants=diagnostic, hidden_source='Actual target-request coda input after generation',
                weights='Original resident BF16 LM-head weights',
                evidence='MEASURED_OPERATOR_NUMERICAL_DIAGNOSTIC_NOT_E2E_SPEEDUP'))
        for record in records:
            for key in ('input_hidden', 'hidden', 'logits'):
                if record[key] is not None:
                    record[key] = record[key].cpu()
        for prefix in prefixes:
            prefix['key'] = prefix['key'].cpu()
            prefix['value'] = prefix['value'].cpu()
        torch.save(dict(records=records,
                        KV_prefixes=prefixes,
                        weight_metadata={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                                         for key, value in weights.items()},
                        scope='Actual request operands. Instrumented timing is not a serving measurement.'),
                   folder / f'actual_operands_{os.getpid()}.pt')
    Finalize(None, finish, exitpriority=16)
