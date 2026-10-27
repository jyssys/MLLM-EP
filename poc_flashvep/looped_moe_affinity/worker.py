import argparse
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

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--rank',type=int,required=True);p.add_argument('--port',type=int,required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());rank=args.rank
    target=ROOT/'raw'/config['id'];target.mkdir(parents=True,exist_ok=True)
    request=target/f'request.rank{rank}.json';assert not request.exists()
    write(request,dict(config=config,rank=rank,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),
        time=time.time(),sources={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}))
    try:
        torch.set_num_threads(1)
        if config['phase']=='foil_official':
            from .quality import official_foil
            official_foil(config,target)
        elif config['phase']=='official_native':
            sys.path.insert(0,str(XLLM))
            from xllm.paper_part1.eval_native import main as official_main
            sys.argv=['eval_paper_part1.py','--artifact',str(MODEL_ASSETS/config['model']),
                '--tasks-root',str(DATA/'paper_part1'),'--output',str(target/'official'),
                '--tasks',*config['tasks']]
            result=official_main();assert result in (None,0),result
        else:
            from .adapter import Adapter
            a=Adapter(config['model'])
            write(target/f'metadata.rank{rank}.json',a.metadata())
            with torch.inference_mode():
                ids=a.encode('The capital of France is Paris. A scientist records a measurement. '*10,32)
                logits=a.forward(ids)
                assert torch.isfinite(logits).all()
                repeat=a.forward(ids)
                assert torch.equal(logits,repeat)
                write(target/f'native_smoke.rank{rank}.json',dict(status='PASS',shape=list(logits.shape),repeat_exact=True,
                    backend=a.backend,relative_l2=0.,scope='Smoke only. Not benchmark accuracy.'))
                if config['phase']=='profile':
                    del logits,repeat
                    from .measure import references
                    shape_reference=references(a,config)
                    logits=a.forward(ids);repeat=a.forward(ids)
                if len(config['gpus'])>1:
                    reference=logits.detach().cpu();del logits,repeat
                    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{args.port}',rank=rank,world_size=len(config['gpus']),
                        timeout=datetime.timedelta(seconds=300),device_id=torch.device('cuda:0'))
                    from .ep import ExpertParallel
                    ep=ExpertParallel(a)
                    a.model.register_forward_pre_hook(lambda module,inputs:ep.reset(ep.request_ids))
                    actual=a.forward(ids)
                    reference=reference.to('cuda')
                    l2=float((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-9))
                    exact=torch.equal(actual,reference)
                    write(target/f'fidelity.rank{rank}.json',dict(relative_l2=l2,logits_exact=exact,backend=a.backend,
                        recurrence_calls=dict(ep.calls),ep=len(config['gpus']),status='PASS' if l2<=.02 else 'FAIL'))
                    assert l2<=.02
                    write(target/f'ownership.rank{rank}.json',ep.ownership)
                    del reference,actual
                    if config['phase']=='ep_eval':
                        from .quality import evaluate_ifm,evaluate_foil
                        (evaluate_foil if config['model']=='foil1' else evaluate_ifm)(a,ep,target,config)
                    elif config['phase']=='profile':
                        from .measure import measure
                        measure(a,ep,target,config,shape_reference)
                    elif config['phase']=='nll':
                        from .nll import evaluate
                        evaluate(a,ep,target,config)
                    elif config['phase']=='paired_verify':
                        from .paired_verify import verify
                        verify(a,ep,target,config)
                    elif config['phase']=='attribution':
                        from .e2e_attribution import measure
                        measure(a,ep,target,config)
                    elif config['phase']=='generic_controls':
                        from .generic_controls import measure
                        measure(a,ep,target,config)
                    dist.barrier();dist.destroy_process_group()
                elif config['phase']=='foil_eval':
                    from .quality import evaluate_foil
                    evaluate_foil(a,None,target,config)
                elif config['phase']=='routes':
                    from .collect_routes import collect
                    collect(a,target,config)
                elif config['phase']=='nll':
                    from .nll import evaluate
                    evaluate(a,None,target,config)
                elif config['phase']=='native_profile':
                    from .native_measure import measure
                    measure(a,target,config)
                elif config['phase']=='native_memory_probe':
                    from .native_memory_probe import measure
                    measure(a,target,config)
                elif config['phase']=='attribution':
                    from .e2e_attribution import measure
                    measure(a,None,target,config)
        write(target/f'complete.rank{rank}.json',dict(status='PASS'))
        print('PASS',config['id'],rank,flush=True)
    except BaseException as error:
        write(target/f'failure.rank{rank}.json',dict(error=repr(error),trace=traceback.format_exc()))
        print(traceback.format_exc(),flush=True);raise

if __name__=='__main__':main()
