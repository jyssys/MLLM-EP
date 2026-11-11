"""Diagnostic exact KV and hidden hashes before final destination release."""
import hashlib
import os
from .common import ROOT, write


def install(cfg):
    assert cfg.get('state_probe') and not cfg.get('clean_confirmation')
    import torch
    from vllm_rlt.engine.stage_engine import StageEngine
    from ..looped_pd_headroom_recovery.state_probe import plane
    finish = StageEngine._finish
    close = StageEngine.close
    rows = []
    marker = ROOT / 'raw' / cfg['name'] / 'measurement_begin.json'

    def finished(engine, request, reason):
        index = request.sampling_params.seed - cfg['seed']
        if engine.stage_config.produces_tokens and marker.exists() and 0 <= index < cfg['count']:
            cache, runner = engine.cache_manager, engine.model_runner
            assert cache.has_allocation(request.request_id), 'Final state must precede KV release'
            assert len(request.generated_token_ids) == cfg['cap'], 'The exactness request did not complete'
            runner.synchronize()
            allocation = cache._get_allocation(request.request_id)
            recurrent = set(engine.model.recurrent_kv_layers)
            valid = len(request.prompt_token_ids) + cfg['cap'] - 1
            combined = hashlib.sha256()
            planes = []
            for depth in range(cfg['depth']):
                blocks = torch.tensor(allocation.block_tables[depth][:(valid+cache.block_size-1)//cache.block_size],
                    device=cache.device, dtype=torch.long)
                for layer in range(cache.num_layers):
                    if depth and layer not in recurrent:
                        continue
                    assert allocation.written[depth][layer].prefix >= valid, 'A final valid KV plane is incomplete'
                    for kind, tensor in (('K', cache.key_cache), ('V', cache.value_cache)):
                        raw = plane(tensor, blocks, layer).flatten(0, 1)[:valid].contiguous().view(torch.uint8).cpu().numpy()
                        data = memoryview(raw).cast('B')
                        combined.update(data)
                        planes.append(dict(depth=depth, layer=layer, kind=kind, valid_tokens=valid,
                            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            hidden = request.hidden_state.contiguous().view(torch.uint8).cpu().numpy()
            rows.append(dict(seed=request.sampling_params.seed, output_ids=list(request.generated_token_ids),
                KV_sha256=combined.hexdigest(), planes=planes,
                hidden_sha256=hashlib.sha256(memoryview(hidden).cast('B')).hexdigest()))
        return finish(engine, request, reason)

    def closed(engine):
        try:
            return close(engine)
        finally:
            if engine.stage_config.produces_tokens:
                write(ROOT / 'raw' / cfg['name'] / f'post_decode_state_{os.getpid()}.json', dict(rows=rows,
                    scope='Diagnostic hashes of every semantically valid final Decode KV plane and final recurrent hidden state. Synchronized before Native release. Not clean timing.'))
    StageEngine._finish, StageEngine.close = finished, closed
