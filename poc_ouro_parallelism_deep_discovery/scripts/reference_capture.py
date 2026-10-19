"""Independent staged-reference capture at the frozen native hardware input."""
import argparse
import re
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports,load_engine
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import numpy as np
import torch

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=MODELS,required=True)
    p.add_argument('--native',type=Path,required=True);args=p.parse_args()
    out=Path(os.environ['LOOPED_KV_OUTPUT'])
    workload=read(args.native/'workload.json')['inputs'][0]
    capture=read(args.native/'parity_capture.json')['evidence'][0]
    N=1024;teacher=workload['teacher_ids'][:4]
    model,tok,engine=load_engine(args.size,'fixed',graphs=False,batch=1,blocks=192,max_model_len=2048)
    logits=[];hidden=[];index=[0];tables=[]
    def head_hook(module,arguments,value):
        hidden.append(arguments[0].reshape(-1,2048)[-1:].clone())
        logits.append(value.reshape(-1,49152)[-1:].clone())
        if len(logits)==4:
            table=next(iter(engine.cache.cache_allocator.block_table.values()))
            tables.append(list(table))
    handle=model.lm_head.register_forward_hook(head_hook)
    originals={name:getattr(engine.runner,name) for name in ['_sample','_sample_pregathered','_run_coda_forward']}
    def force(tensor):
        # The native output staging buffer can exceed the single active row.
        # Force only its first active slot. Do not alter padded request slots.
        assert tensor.numel()>=1 and index[0]<len(teacher)
        tensor.reshape(-1)[0]=teacher[index[0]];index[0]+=1
    def sample(*values,**kwargs):
        originals['_sample'](*values,**kwargs);force(values[-1])
    def pregather(*values,**kwargs):
        originals['_sample_pregathered'](*values,**kwargs);force(values[-1])
    def coda(*values,**kwargs):
        result,staging=originals['_run_coda_forward'](*values,**kwargs);force(result)
        return result,staging
    engine.runner._sample=sample;engine.runner._sample_pregathered=pregather;engine.runner._run_coda_forward=coda
    result=engine.generate_batch([workload['prompt_ids'][str(N)]],max_new_tokens=4,eos_token_id=None,warmup=False)[0]
    torch.cuda.synchronize()
    assert result.generated_tokens==teacher and len(hidden)==len(logits)==index[0]==4
    assert len(tables)==1
    positions=capture['positions'];indices=torch.tensor([tables[0][p//16]*16+p%16 for p in positions],device='cuda')
    dims=torch.tensor([0,1,2,3,124,125,126,127],device='cuda');kv=[]
    for name in capture['KV_names']:
        logical=int(re.search(r'layers\.(\d+)\.',name).group(1))
        k=engine.cache.key_cache[logical].index_select(0,indices)[:,:2].index_select(-1,dims)
        v=engine.cache.value_cache[logical].index_select(0,indices)[:,:2].index_select(-1,dims)
        kv.append(torch.cat([k,v],dim=-1).float().cpu().numpy())
    path=out/f'reference_base_{args.size}_R4.npz'
    np.savez_compressed(path,logits=torch.cat(logits).float().cpu().numpy(),
        hidden=torch.cat(hidden).float().cpu().numpy(),KV=np.stack(kv))
    handle.remove()
    write(out/'status.json',dict(success=True,model=MODELS[args.size]['id'],revision=MODELS[args.size]['revision'],
        size=args.size,R=4,source_native=str(args.native),positions=positions,KV_names=capture['KV_names'],
        path=str(path),sha256=sha(path),teacher_ids=teacher,physical_gpu=validate_mask()[0],
        evidence_class='QUALITY_ONLY',scope='Four controlled teacher-forced positions. Actual staged reference model with selected KV features. Not task accuracy.'))
    print('REFERENCE_CAPTURE',args.size)

if __name__=='__main__':main()
