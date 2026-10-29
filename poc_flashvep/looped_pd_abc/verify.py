"""Heavy, explicit byte checks. These checks never run in clean serving trials."""
import hashlib
import os
from pathlib import Path
import torch
from .common import append

def fingerprint(connector,transfer):
 cache=connector.cache;model=connector.engine.model;n=len(transfer.request.prompt_token_ids)
 recurrent=set(getattr(model,'recurrent_kv_layers',range(cache.num_layers)))
 coda=set(getattr(model,'coda_kv_layers',()))
 allocation=cache._get_allocation(transfer.tid);cut=getattr(connector,'cut',cache.max_loops)
 h=hashlib.sha256();total=0
 for depth in range(cut):
  for layer in range(cache.num_layers):
   if depth and layer not in recurrent:continue
   if layer in coda and hasattr(connector,'cut'):continue
   end=n-1 if layer in coda else n
   for p in range(0,end,cache.block_size):
    block=allocation.block_tables[depth][p//cache.block_size];count=min(cache.block_size,end-p)
    for tensor in (cache.key_cache,cache.value_cache):
     data=tensor[block,layer,:count].contiguous().view(torch.uint8).cpu().numpy().tobytes()
     h.update(data);total+=len(data)
 hidden=connector.hidden[transfer.slot].contiguous().view(torch.uint8).cpu().numpy().tobytes()
 return dict(KV_sha256=h.hexdigest(),KV_valid_bytes=total,hidden_sha256=hashlib.sha256(hidden).hexdigest(),hidden_bytes=len(hidden))

def check(connector,request):
 transfer=connector.transfers[request.request_id]
 actual=fingerprint(connector,transfer);expected=getattr(transfer,'expected_fingerprint',None)
 row=dict(tid=transfer.tid,model=connector.engine.model.config.to_dict()['model_type'],expected=expected,actual=actual,
  passed=actual==expected,attempt_id=os.environ.get('ABC_ATTEMPT_ID'),evidence='MEASURED_NIXL_BYTE_IDENTITY_HEAVY_CORRECTNESS_ONLY')
 append(Path(os.environ['ABC_RUN_DIR'])/('KV_checks-'+str(os.getpid())+'.jsonl'),row)
 if actual!=expected:raise RuntimeError('NIXL KV or hidden byte fingerprint mismatch')
