import argparse
from collections import Counter
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.distributed as dist

from .common import *
from ..looped_moe_headroom.adapters import Adapter,quality
from .runtime import PhysicalEP

def forward(a,ep,ids):
    ep.reset()
    return a.forward(ids)

def greedy(a,ep,ids,cap=8):
    out=[]
    for _ in range(cap):
        logits=forward(a,ep,ids) if ep else a.forward(ids)
        token=logits[:,-1].argmax(-1,keepdim=True)
        out.append(token[0].item())
        ids=torch.cat((ids,token),dim=-1)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    rank=args.rank
    world=len(config['gpus'])
    target=ROOT/'raw'/config['id']
    target.mkdir(parents=True,exist_ok=True)
    request=target/f'request.rank{rank}.json'
    assert not request.exists(),'Preserve every existing attempt'
    write(request,dict(config=config,rank=rank,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),
        start=time.time(),source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}))
    try:
        torch.set_num_threads(1)
        a=Adapter(config['model'])
        metadata=a.metadata()
        metadata['eos_token_id']=a.eos()
        write(target/f'native_metadata.rank{rank}.json',metadata)
        with torch.inference_mode():
            ids=a.encode('The capital of France is Paris. A scientist measures the temperature and records the result. '*30,32)
            reference=a.forward(ids)
            repeat=a.forward(ids)
            assert torch.equal(reference,repeat),'Native repeated logits differ'
            reference_cpu=reference.cpu()
            del reference,repeat
            native_tokens=greedy(a,None,ids,config.get('quality_tokens',8))
            if config.get('phase') in ('baseline','quality'):
                from .measure import prepare_references,prepare_shape_references
                engineering_references=prepare_references(a,config.get('quality_examples',32))
                engineering_references+=prepare_shape_references(a,config)
        dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{args.port}',rank=rank,world_size=world,
            timeout=datetime.timedelta(seconds=240),device_id=torch.device('cuda:0'))
        with torch.inference_mode():
            ep=PhysicalEP(a)
            write(target/f'ownership.rank{rank}.json',ep.ownership)
            dist.barrier()
            ep.trace=True
            actual=forward(a,ep,ids)
            traces=ep.resolve()
            ep.trace=False
            reference=reference_cpu.to('cuda')
            info=quality(actual,reference,ids)
            info.update(relative_l2=float((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-9)),
                max_scaled_abs=float((actual.float()-reference.float()).abs().max()/reference.float().abs().max().clamp_min(1e-9)),
                max_absolute_error=float((actual.float()-reference.float()).abs().max()),
                native_repeat_exact=True,observed_site_calls=dict(ep.calls))
            del reference,reference_cpu,actual
            ep_tokens=greedy(a,ep,ids,config.get('quality_tokens',8))
            info.update(native_tokens=native_tokens,ep_tokens=ep_tokens,token_exact=native_tokens==ep_tokens,
                thresholds=dict(relative_l2=.02,max_scaled_abs=.04,top1_agreement=.99,absolute_delta_nll=.02),
                topology='TP1 replicated dense / source-SP / actual ragged physical EP / DP1 / PP1',
                ep_degree=world,rank=rank,physical_gpu=config['gpus'][rank],gpu_uuid=a.device_uuid,
                original_native_backend=a.backend,learned_dtype='BF16',native_backend_qualification='See previous FA4 study',
                resident_before_sharding=ep.start_memory,resident_after_sharding=ep.resident_after_shard,
                routed_weight_bytes=sum(t['bytes'] for row in ep.ownership for t in row['tensors']),
                nccl_version=torch.cuda.nccl.version(),torch=torch.__version__,cuda=torch.version.cuda)
            expected={site:a.loops if site in a.recurrent else 1 for site in a.sites}
            assert dict(ep.calls)==expected,(dict(ep.calls),expected)
            info['pass_fidelity']=info['relative_l2']<=.02 and info['max_scaled_abs']<=.04 and info['top1_agreement']>=.99 and abs(info['delta_nll'])<=.02 and info['token_exact']
            write(target/f'fidelity.rank{rank}.json',info)
            write(target/f'trace.rank{rank}.json',traces)
            assert info['pass_fidelity'],'BLOCKED_BACKEND_FIDELITY'
            if config.get('phase')=='ready_probe':
                from .ready_probe import observe
                observe(a,ep,target,config)
            if config.get('phase')=='steady_generation':
                from .steady import measure
                measure(a,ep,target,config)
            if config.get('phase') in ('baseline','quality'):
                from .measure import run,validate_references
                validate_references(a,ep,target,engineering_references)
                del engineering_references
                if config.get('phase')=='baseline':run(a,ep,target,config)
                elif config.get('profile',True):
                    from .measure import profile_forward,shaped_inputs
                    from .operator_controls import micro
                    example,requests=shaped_inputs(a,1,128)
                    ep.trace=True
                    ep.reset(requests)
                    a.forward(example)
                    ep.resolve()
                    ep.trace=False
                    micro(a,ep,target,1,128)
                    profile_forward(a,ep,target)
            for ownership in ep.ownership:
                block=a.sites[ownership['site']]
                owner=block.experts if a.name.startswith('ifm') else block
                assert all(getattr(owner,t['name']).data_ptr()==t['pointer'] for t in ownership['tensors'])
            write(target/f'complete.rank{rank}.json',dict(status='PASS',rank=rank,ep=world,model=config['model'],
                expert_storage_unchanged=True))
            dist.barrier()
        dist.destroy_process_group()
        print('PASS',config['id'],rank,flush=True)
    except BaseException as error:
        write(target/f'failure.rank{rank}.json',dict(error=repr(error),trace=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
        raise

if __name__=='__main__':main()
