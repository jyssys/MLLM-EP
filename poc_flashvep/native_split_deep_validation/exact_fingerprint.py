"""Reduce diagnostic gathers while preserving every hashed byte and its order."""
import hashlib
import os


def raw(value):
    import torch
    return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def partial_fingerprint(connector, transfer):
    import torch
    cache = connector.cache
    allocation = cache._get_allocation(transfer.tid)
    sha, size = hashlib.sha256(), 0
    for depth, layer, end in connector.layout(transfer):
        written = allocation.written[depth][layer].prefix
        if written < end:
            raise RuntimeError(f'KV frontier is not valid: {depth}/{layer} {written}<{end}')
        blocks = torch.tensor(allocation.block_tables[depth][:(end + cache.block_size - 1) // cache.block_size],
                              device=cache.device)
        for tensor in (cache.key_cache, cache.value_cache):
            payload = raw(tensor[:, layer].index_select(0, blocks).flatten(0, 1)[:end])
            sha.update(payload)
            size += len(payload)
    hidden = raw(connector.hidden[transfer.slot, :connector.hidden_elements(transfer)])
    return dict(KV_sha256=sha.hexdigest(), KV_bytes=size,
                hidden_sha256=hashlib.sha256(hidden).hexdigest(), hidden_bytes=len(hidden))


def native_fingerprint(connector, transfer):
    import torch
    cache, model = connector.cache, connector.engine.model
    n = len(transfer.request.prompt_token_ids)
    recurrent = set(getattr(model, 'recurrent_kv_layers', range(cache.num_layers)))
    coda = set(getattr(model, 'coda_kv_layers', ()))
    allocation = cache._get_allocation(transfer.tid)
    cut = getattr(connector, 'cut', cache.max_loops)
    sha, total = hashlib.sha256(), 0
    for depth in range(cut):
        for layer in range(cache.num_layers):
            if depth and layer not in recurrent:
                continue
            if layer in coda and hasattr(connector, 'cut'):
                continue
            end = n - 1 if layer in coda else n
            blocks = torch.tensor(allocation.block_tables[depth][:(end + cache.block_size - 1) // cache.block_size],
                                  device=cache.device)
            values = [tensor[:, layer].index_select(0, blocks).contiguous().view(torch.uint8).cpu().numpy()
                      for tensor in (cache.key_cache, cache.value_cache)]
            for page, position in enumerate(range(0, end, cache.block_size)):
                count = min(cache.block_size, end - position)
                for value in values:
                    payload = value[page, :count].tobytes()
                    sha.update(payload)
                    total += len(payload)
    hidden = raw(connector.hidden[transfer.slot])
    return dict(KV_sha256=sha.hexdigest(), KV_valid_bytes=total,
                hidden_sha256=hashlib.sha256(hidden).hexdigest(), hidden_bytes=len(hidden))


def install(cfg):
    if not cfg.get('verify_transfer') or not cfg.get('deep_fast_verification', True):
        return
    from ..adaptive_recurrence_cut.connector import CutConnector
    from ..looped_pd_abc import verify
    from .common import ROOT, write
    CutConnector.fingerprint = partial_fingerprint
    verify.fingerprint = native_fingerprint
    write(ROOT / 'raw' / cfg['name'] / f'deep_exact_fingerprint_{os.getpid()}.json', dict(
        layer_first_gather=True, native_block_K_V_order_preserved=True,
        full_prompt_state_probe_layer_first=True,
        full_prompt_probe='Reuses the existing looped_pd_headroom_recovery.state_probe. No replacement is required.',
        partial_plane_K_V_order_preserved=True, all_valid_bytes_retained=True,
        scope='Heavy diagnostic only. Same SHA256 and byte order. Does not run in clean serving.'))
