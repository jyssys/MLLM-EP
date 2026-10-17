"""Certify official-forward hook parity, cache indexing, and calibration inputs."""
import gc
import importlib
import os
import subprocess
import time
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from poc_async_recurrent_pp.contract import *
from poc_async_recurrent_pp.emulator import StageEmulator, cache_signature, generate


def token_ids(tokenizer, row, suffix):
    return tokenizer.apply_chat_template([dict(role='user',content=row['question']+suffix)],
        tokenize=True,add_generation_prompt=True)


def load():
    assert set(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) <= {'4','5','6','7'}
    model = AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,
        torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.config.num_hidden_layers == 24
    assert model.config.total_ut_steps == 4 and model.config.early_exit_threshold == 1
    return model, tokenizer


@torch.inference_mode()
def main():
    output = Path(os.environ['LOOPED_KV_OUTPUT'])
    torch.manual_seed(20261017)
    model, tokenizer = load()
    remote = importlib.import_module(type(model).__module__)
    assert sha(remote.__file__) == sha(MODEL/'modeling_ouro.py')
    write(output/'environment.json',dict(model_revision=REVISION,source_sha256=sha(remote.__file__),
        source_file=remote.__file__,torch=torch.__version__,transformers=transformers.__version__,
        cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),
        mask=os.environ['CUDA_VISIBLE_DEVICES'],logical0_UUID=str(torch.cuda.get_device_properties(0).uuid),
        GPU=subprocess.check_output(['nvidia-smi','--id=4,5,6,7','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),config=model.config.to_dict(),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        source_hashes={name:sha(MODEL/name) for name in ['config.json','modeling_ouro.py','configuration_ouro.py','tokenizer.json','tokenizer_config.json','model.safetensors']},
        weight_patch=None,source_patch=None))
    data = read(ROOT/'outputs/manifests/data.json')
    suffix = data['prompt_suffix']
    parity = []
    for row in data['smoke']:
        ids = token_ids(tokenizer,row,suffix)
        inp = torch.tensor([ids],device='cuda')
        reference = model(inp,use_cache=True,exit_at_step=3,logits_to_keep=1)
        emulator = StageEmulator(model, capture=True)
        observed = model(inp,use_cache=True,exit_at_step=3,logits_to_keep=1)
        emulator.close()
        bitwise = torch.equal(reference.logits,observed.logits)
        kv_equal = all(torch.equal(a,b) for a,b in zip(reference.past_key_values.key_cache,observed.past_key_values.key_cache))
        kv_equal &= all(torch.equal(a,b) for a,b in zip(reference.past_key_values.value_cache,observed.past_key_values.value_cache))
        assert bitwise and kv_equal and len(emulator.captures)==16
        signature = cache_signature(observed.past_key_values)
        parity.append(dict(id=row['id'],logits_bitwise=bitwise,all_KV_bitwise=kv_equal,
            calls=len(emulator.captures),cache=signature))
        del reference,observed
    # Compare actual official generate to the custom exact greedy loop.
    natural = []
    for row in data['smoke'][:4]:
        ids=token_ids(tokenizer,row,suffix)
        inp=torch.tensor([ids],device='cuda')
        official=model.generate(inp,attention_mask=torch.ones_like(inp),max_new_tokens=2048,
            do_sample=False,pad_token_id=tokenizer.eos_token_id,eos_token_id=tokenizer.eos_token_id,exit_at_step=3)
        official_ids=official[0,len(ids):].tolist()
        hooks=StageEmulator(model)
        instrumented=generate(model,tokenizer,ids,2048,hooks)
        hooks.close()
        assert official_ids==instrumented['ids']
        natural.append(dict(id=row['id'],prompt_ids=ids,official_ids=official_ids,**instrumented,
            official_generate_parity=True))
        print('natural parity',row['id'],len(official_ids),flush=True)
    write(output/'reference_parity.json',dict(smoke=parity,natural=natural,pass_gate=True))
    # Fit scalar drift predictors on independent train examples only.
    numerator=np.zeros((3,3),dtype=np.float64)
    denominator=np.zeros((3,3),dtype=np.float64)
    w0xy=np.zeros(3);w0xx=np.zeros(3)
    sum_x=np.zeros((3,2048));sum_y=np.zeros((3,2048));count=0
    observations=[]
    for row in data['calibration']:
        inp=torch.tensor([token_ids(tokenizer,row,suffix)],device='cuda')
        past=None
        hook=StageEmulator(model,capture=True)
        for step in range(17):
            hook.captures=[]
            result=model(inp,past_key_values=past,use_cache=True,exit_at_step=3,logits_to_keep=1)
            past=result.past_key_values
            states={(v['round'],v['stage']):v['value'].numpy().astype(np.float64) for v in hook.captures}
            # Last prompt position plus 16 free-running calibration positions.
            for r in range(1,4):
                drift=states[r,0]-states[r-1,0]
                for j in range(1,4):
                    delta=states[r,j]-states[r-1,j]
                    numerator[r-1,j-1]+=np.sum(delta*drift)
                    denominator[r-1,j-1]+=np.sum(drift*drift)
                    observations.append(dict(id=row['id'],step=step,round=r,stage=j,
                        stale_relative_L2=float(np.linalg.norm(delta)/(np.linalg.norm(states[r,j])+1e-12)),
                        adjacent_cosine=float(np.sum(states[r,j]*states[r-1,j])/(np.linalg.norm(states[r,j])*np.linalg.norm(states[r-1,j])+1e-12))))
            x=states[0,0].reshape(-1)
            for j in range(1,4):
                y=states[0,j].reshape(-1)
                w0xy[j-1]+=np.sum(x*y);w0xx[j-1]+=np.sum(x*x)
                sum_x[j-1]+=x;sum_y[j-1]+=y
            count+=1
            token=int(result.logits[0,-1].argmax())
            inp=torch.tensor([[token]],device='cuda')
        hook.close()
        del past,result
        gc.collect()
        print('calibration',row['id'],flush=True)
    beta=w0xy/(w0xx+1e-12)
    coefficients=dict(alpha=(numerator/(denominator+1e-12)).tolist(),w0_beta=beta.tolist(),
        w0_bias=((sum_y-beta[:,None]*sum_x)/count).tolist(),
        calibration_ids=[r['id'] for r in data['calibration']],observed_positions=count,
        fit='Scalar least squares. W0 adds calibration-only mean residual.',
        arithmetic='Prediction FP32, cast to BF16. Model weights/compute BF16 unchanged.')
    write(ROOT/'outputs/manifests/predictor.json',coefficients)
    write(output/'activation_observations.json',observations)
    write(output/'status.json',dict(pass_gate=True,n_reference_smoke=16,n_natural_parity=4,
        calibration_requests=8,calibration_positions=count,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        note='Baseline task-quality pilot remains a separate required check.'))
    print('BASELINE_PARITY_PASS',flush=True)


if __name__=='__main__':
    main()
