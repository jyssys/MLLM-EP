"""Preserve the legacy byte stream with bounded plane-wise GPU readbacks."""
import hashlib


def fingerprint(connector,transfer):
    import torch
    cache=connector.cache
    model=connector.engine.model
    n=len(transfer.request.prompt_token_ids)
    recurrent=set(getattr(model,'recurrent_kv_layers',range(cache.num_layers)))
    coda=set(getattr(model,'coda_kv_layers',()))
    allocation=cache._get_allocation(transfer.tid)
    cut=getattr(connector,'cut',cache.max_loops)
    h=hashlib.sha256()
    total=0
    for depth in range(cut):
        for layer in range(cache.num_layers):
            if depth and layer not in recurrent:continue
            if layer in coda and hasattr(connector,'cut'):continue
            end=n-1 if layer in coda else n
            pages=(end+cache.block_size-1)//cache.block_size
            blocks=torch.tensor(allocation.block_tables[depth][:pages],device=cache.key_cache.device,dtype=torch.long)
            arrays=[tensor[:,layer].index_select(0,blocks).contiguous().view(torch.uint8).cpu().numpy()
                for tensor in (cache.key_cache,cache.value_cache)]
            # Keep the original depth/layer/page/K/V byte order, including
            # the partial final page and the Coda's missing final position.
            for page,p in enumerate(range(0,end,cache.block_size)):
                count=min(cache.block_size,end-p)
                for array in arrays:
                    data=array[page,:count].tobytes()
                    h.update(data)
                    total+=len(data)
    hidden=connector.hidden[transfer.slot].contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(KV_sha256=h.hexdigest(),KV_valid_bytes=total,
        hidden_sha256=hashlib.sha256(hidden).hexdigest(),hidden_bytes=len(hidden))
