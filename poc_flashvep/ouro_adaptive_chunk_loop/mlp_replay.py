"""Same-input, same-weight BF16 GEMM geometry diagnostic."""
import json
import re
import torch
from .common import ROOT,write,digest

def find(tensors,kind,position):
    for key,tensor in tensors.items():
        match=re.match(rf'^t0_d2_l0_{kind}_p(\d+)_(\d+)$',key)
        if match and int(match[1])<=position<=int(match[2]):
            return tensor[position-int(match[1])]
    raise KeyError((kind,position))

def difference(reference,candidate):
    delta=reference.float()-candidate.float()
    return dict(exact=torch.equal(reference,candidate),max_abs=float(delta.abs().max()),
        relative_L2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)))

@torch.inference_mode()
def run_job(model,job,target):
    folder=ROOT/'raw'/'p1_attention_boundary_graph0_v1'
    position=61
    if (folder/'atomic_selected_tensors.pt').exists():
        source={mode:torch.load(folder/f'{mode}_selected_tensors.pt',map_location='cpu',mmap=True,weights_only=True) for mode in ['atomic','grouped']}
        inputs={m:find(t,'post_attention_norm',position) for m,t in source.items()}
        expected={m:find(t,'MLP',position).to('cuda') for m,t in source.items()}
        sources={m:dict(index_sha256=digest(folder/f'{m}_tensor_index.json'),tensor_file=str(folder/f'{m}_selected_tensors.pt')) for m in source}
    else:
        witness=ROOT/'portable_mlp_witness.pt'
        tensors=torch.load(witness,map_location='cpu',weights_only=True)
        inputs={m:tensors['input'] for m in ('atomic','grouped')}
        expected={m:tensors[m].to('cuda') for m in ('atomic','grouped')}
        sources=dict(portable_witness=str(witness),sha256=digest(witness))
    assert torch.equal(inputs['atomic'],inputs['grouped']),'Same-input control failed'
    x=inputs['atomic'].to('cuda')
    mlp=model.model.layers[0].mlp
    previous=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        reference=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x.float(),mlp.gate_proj.weight.float()))
            *torch.nn.functional.linear(x.float(),mlp.up_proj.weight.float()),mlp.down_proj.weight.float())
        rows=[]
        for count in [1,24,48,64,128,203,1024]:
            values=mlp(x.reshape(1,-1).repeat(count,1))
            native=values[0]
            rows.append(dict(rows=count,reference='Same captured BF16 input and pretrained MLP weights',
                versus_captured_atomic=difference(expected['atomic'],native),
                versus_captured_grouped=difference(expected['grouped'],native),
                versus_FP32_diagnostic=difference(reference,native)))
        exact_originals=next(r for r in rows if r['rows']==1024)['versus_captured_atomic']['exact'] and next(r for r in rows if r['rows']==48)['versus_captured_grouped']['exact']
        report=dict(position=position,zero_based_recurrent_depth=2,physical_layer=0,same_input_exact=True,
            captured_output_difference=difference(expected['atomic'],expected['grouped']),geometry=rows,
            original_geometries_reproduce_exactly=exact_originals,
            conclusion='BF16_GEMM_BATCH_GEOMETRY_REPRODUCED_FOR_THIS_CASE' if exact_originals else 'BATCH_GEOMETRY_NOT_FULLY_ATTRIBUTED',
            scope='Operator replay only. No FP32 serving substitution. No task-accuracy claim.',
            sources=sources)
        write(target/'mlp_geometry_replay.json',report)
        print(report['conclusion'],flush=True)
        return report
    finally:
        torch.backends.cuda.matmul.allow_tf32=previous

def register():
    path=ROOT/'p1_mlp_replay_plan.json'
    assert not path.exists()
    write(path,dict(gpus=[0],jobs=[dict(id='p1_MLP_same_input_geometry_v1',kind='mlp_replay',gpu=0,split='DIAGNOSTIC',timeout_s=600)]))

def export_witness():
    folder=ROOT/'raw'/'p1_attention_boundary_graph0_v1'
    source={m:torch.load(folder/f'{m}_selected_tensors.pt',map_location='cpu',mmap=True,weights_only=True) for m in ('atomic','grouped')}
    inputs={m:find(t,'post_attention_norm',61) for m,t in source.items()}
    assert torch.equal(inputs['atomic'],inputs['grouped'])
    data=dict(input=inputs['atomic'].clone(),atomic=find(source['atomic'],'MLP',61).clone(),grouped=find(source['grouped'],'MLP',61).clone())
    path=ROOT/'portable_mlp_witness.pt'
    torch.save(data,path)
    write(ROOT/'portable_mlp_witness.json',dict(path=path.name,sha256=digest(path),position=61,depth=2,layer=0,
        scope='Three captured BF16 vectors only. No model weights. Replays one diagnosed operator case.'))

if __name__=='__main__':
    if '--export-witness' in __import__('sys').argv:
        export_witness()
    else:
        register()
