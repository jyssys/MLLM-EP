"""Heavy prompt-completion state checks before the first sampled output token."""
import hashlib
import os
import time


def install():
    if os.environ.get('RCC_STATE_PROBE') != '1':
        return
    import torch
    from multiprocessing.util import Finalize
    from vllm_rlt.worker.model_runner import ModelRunner
    from vllm_rlt.request import Operation
    from .common import ROOT, write
    if getattr(ModelRunner, '_rcc_state_probe_installed', False):
        return
    ModelRunner._rcc_state_probe_installed = True
    records = []
    execute = ModelRunner._execute_batch

    def byte_hash(tensor):
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest(), len(raw)

    def check(self, batch, prepared):
        if batch.operation == Operation.CODA:
            hidden = prepared.forwards[0].forward.hidden
            offset = 0
            for span in batch.spans:
                request = span.request
                n = len(request.prompt_token_ids)
                if span.token_start == n-1 and request.sampling_params.seed in (2026103017, 2026103018):
                    cache = self.cache_manager
                    allocation = cache._get_allocation(request.request_id)
                    recurrent = set(getattr(self.model, 'recurrent_kv_layers', range(cache.num_layers)))
                    coda = set(getattr(self.model, 'coda_kv_layers', ()))
                    planes = []
                    for depth, table in enumerate(allocation.block_tables):
                        blocks = torch.tensor(table[:(n+cache.block_size-1)//cache.block_size], device=cache.device)
                        for layer in range(cache.num_layers):
                            if depth and layer not in recurrent:
                                continue
                            valid = n-1 if layer in coda else n
                            for name, tensor in (('K', cache.key_cache), ('V', cache.value_cache)):
                                gathered = tensor.index_select(0, blocks)[:,layer].flatten(0,1)[:valid]
                                sha, size = byte_hash(gathered)
                                planes.append(dict(depth=depth, layer=layer, kind=name, valid_tokens=valid,
                                                   sha256=sha, bytes=size))
                    sha, size = byte_hash(hidden[offset])
                    records.append(dict(seed=request.sampling_params.seed, prompt_tokens=n, host_time=time.monotonic(),
                                        phase='Before first CODA, after full fixed-depth prompt completion',
                                        hidden_sha256=sha, hidden_bytes=size, planes=planes,
                                        request_id=request.request_id))
                offset += span.token_count
        return execute(self, batch, prepared)

    ModelRunner._execute_batch = check

    def finish():
        write(ROOT/'raw'/os.environ['RCC_RUN']/f'prompt_state_probe_{os.getpid()}.json',
              dict(records=records, evidence='MEASURED_HEAVY_CORRECTNESS',
                   scope='Only two seeds. Coda KV excludes its not-yet-written final prompt row.'))

    Finalize(None, finish, exitpriority=19)
