"""Real-weight projection search. This is not a full recurrent or serving timer."""
import argparse,math,statistics,os,sys,time
from .common import *
from .lt import Plan,library


def percent(values,q):
    values=sorted(values);x=(len(values)-1)*q;a=int(x);b=math.ceil(x)
    return values[a]+(values[b]-values[a])*(x-a)


def main():
    p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args()
    initialize();cfg=json.loads((ROOT/'configs'/(args.name+'.json')).read_text());folder=ROOT/'raw'/args.name
    import torch
    import torch.nn.functional as F
    torch.set_grad_enabled(False);torch.set_num_threads(1);torch.cuda.set_device(0)
    from ..looped_pd_abc.common import MODELS,MODEL_PINS
    from vllm_rlt.models import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(MODELS[cfg['model']],device='cuda:0',dtype=torch.bfloat16)
    groups={}
    for name,module in model.named_modules():
        core=('transformer.core_block.' in name or name=='transformer.adapter') if cfg['model']=='huginn' else 'model.layers.' in name
        if not core or not isinstance(module,torch.nn.Linear):continue
        role=('down' if name.endswith('down_proj') or '.mlp.proj' in name else 'adapter' if name.endswith('adapter') else
            'up' if '.mlp.' in name else 'attention')
        key=(role,tuple(module.weight.shape))
        if key not in groups:groups[key]=dict(name=name,module=module,multiplicity=0)
        groups[key]['multiplicity']+=1
    if cfg.get('smoke'):groups={k:v for k,v in groups.items() if k[0] in ('down','adapter')}
    results=[];memory=[]
    def exact(a,b):
        return dict(equal=torch.equal(a,b),mismatched=int((a!=b).sum().item()),
            max_abs=float((a.float()-b.float()).abs().max().item()))
    def fixed(x,w):
        outputs=[]
        for left in range(0,len(x),256):
            live=x[left:left+256];pad=x.new_zeros((256,x.shape[1]));pad[:len(live)].copy_(live)
            outputs.append(F.linear(pad,w)[:len(live)])
        return torch.cat(outputs) if len(outputs)>1 else outputs[0]
    def timing(fn,repeats=5,inner=8):
        for _ in range(3):fn()
        torch.cuda.synchronize();samples=[]
        for _ in range(repeats):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(inner):fn()
            b.record();b.synchronize();samples.append(a.elapsed_time(b)/inner)
        return dict(p50_ms=percent(samples,.5),p95_ms=percent(samples,.95),samples_ms=samples)
    def graph_timing(fn,repeats=9,inner=16):
        for _ in range(3):fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        result=timing(g.replay,repeats=repeats,inner=inner)
        del g;torch.cuda.synchronize();return result
    for (role,shape),group in groups.items():
        w=group['module'].weight;n,k=shape
        for m in cfg['rows']:
            torch.manual_seed(cfg['selection_seed']);x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
            sensitive=role in ('down','adapter')
            reference=lambda:fixed(x,w) if sensitive else F.linear(x,w)
            expected=reference().clone();native=lambda:F.linear(x,w)
            item=dict(role=role,module=group['name'],weight_shape=list(shape),rows=m,
                multiplicity_per_loop=group['multiplicity'],qualified_padding=sensitive,
                reference_geometry='256-row slices' if sensitive else 'Native unpadded',
                qualified_reference=timing(reference),unpadded=timing(native),
                unpadded_exact_selection=exact(expected,native()),candidates=[])
            plan=Plan(m,n,k,workspace=cfg['workspace_bytes'],maximum=cfg['maximum_candidates'])
            y=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
            for index in range(len(plan)):
                candidate=plan.metadata(index)
                fn=lambda index=index:plan.run(index,x,w,y)
                try:
                    fn();torch.cuda.synchronize();candidate['selection_exact']=exact(expected,y)
                    candidate['selection_timing']=timing(fn,repeats=3,inner=5)
                    candidate['selection_graph']=graph_timing(fn,repeats=3,inner=8)
                except RuntimeError as error:
                    candidate['error']=str(error)
                item['candidates'].append(candidate)
            viable=[c for c in item['candidates'] if 'selection_graph' in c]
            ordered=sorted(viable,key=lambda c:c['selection_graph']['p50_ms'])
            shortlist=ordered[:cfg['validate_fastest']]
            shortlist += sorted([c for c in viable if c['selection_exact']['equal']],key=lambda c:c['selection_graph']['p50_ms'])[:cfg['validate_fastest']]
            chosen={c['index']:c for c in shortlist}
            for index,candidate in chosen.items():
                candidate['heldout_exact']=[]
                for seed in cfg['validation_seeds']:
                    torch.manual_seed(seed);x.copy_(torch.randn_like(x))
                    expected=reference().clone();plan.run(index,x,w,y);torch.cuda.synchronize()
                    candidate['heldout_exact'].append(dict(seed=seed,**exact(expected,y)))
                candidate['operator_qualified']=candidate['selection_exact']['equal'] and all(v['equal'] for v in candidate['heldout_exact'])
                candidate['heldout_graph']=graph_timing(lambda index=index:plan.run(index,x,w,y))
            item['reference_graph']=graph_timing(reference)
            item['unpadded_graph']=graph_timing(native)
            candidates=list(chosen.values())
            item['unpadded_heldout_exact']=[]
            for seed in cfg['validation_seeds']:
                torch.manual_seed(seed);x.copy_(torch.randn_like(x))
                item['unpadded_heldout_exact'].append(dict(seed=seed,**exact(reference(),native())))
            if candidates:item['fastest_tested']=min(candidates,key=lambda c:c['heldout_graph']['p50_ms'])['index']
            valid=[c for c in candidates if c['operator_qualified']]
            item['best_operator_qualified']=min(valid,key=lambda c:c['heldout_graph']['p50_ms'])['index'] if valid else None
            item['scope']='Real checkpoint weight, seeded operands, isolated projection CUDA Event enclosures. Operator equality is not model-state qualification. Finite heuristic search, not a global optimum. Graph calls exclude plan creation and workspace allocation.'
            results.append(item)
            write(folder/'projection_results_partial.json',dict(config=cfg,results=results))
            print('CELL',cfg['model'],role,shape,'M',m,'candidates',len(plan),'exact',len(valid),'ref_ms',item['reference_graph']['p50_ms'],flush=True)
            plan.close();del plan,y,x,expected
            torch.cuda.synchronize()
        memory.append(dict(role=role,shape=shape,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))
    metadata=dict(torch=torch.__version__,CUDA=torch.version.cuda,python=sys.version,
        GPU=torch.cuda.get_device_name(),properties=str(torch.cuda.get_device_properties(0)),
        cublasLt_version=library().lt_version(),preferred_BLAS=str(torch.backends.cuda.preferred_blas_library()),
        BF16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        matmul_precision=torch.get_float32_matmul_precision(),model_pin=MODEL_PINS[cfg['model']],
        libraries=[line.strip() for line in Path('/proc/self/maps').read_text().splitlines() if 'libcublas' in line])
    write(folder/'projection_results.json',dict(config=cfg,results=results,metadata=metadata,memory=memory,
        source_hashes={str(p.relative_to(REPO)):digest(p) for p in SOURCE.rglob('*') if p.suffix in ('.py','.cc')}))
    print('COMPLETE',args.name,flush=True)


if __name__=='__main__':main()
