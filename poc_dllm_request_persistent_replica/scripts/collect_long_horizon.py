"""Reuse validated worker AST: change only frozen cap, manifest and watchdog.

Model loading, DeepEP sparse dispatch/combine, exact source partition, native
decoder, timing lanes and Recorder are the previous implementation unchanged.
No external source file is modified, no runtime replication method is added.
"""
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import torch.multiprocessing as mp
from poc_dllm_request_persistent_replica.contract import *


class AmendWorker(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='generate':
            for key in node.keywords:
                if key.arg=='gen_length':
                    key.value=ast.Call(func=ast.Attribute(value=ast.Name(id='row',ctx=ast.Load()),attr='get',ctx=ast.Load()),
                        args=[ast.Constant('generation_budget'),ast.Constant(64)],keywords=[])
        if isinstance(node.func,ast.Attribute) and node.func.attr=='alarm' and node.args and isinstance(node.args[0],ast.Constant) and node.args[0].value==120:
            node.args[0]=ast.Constant(300)
        if isinstance(node.func,ast.Name) and node.func.id=='generate' and len(node.args)>=4 and isinstance(node.args[3],ast.JoinedStr):
            # Tiny deterministic probe: same warm prompt/block32, bypass vs
            # trace; retain native64 cap for warmups, not long cohort.
            node.args[2]=ast.IfExp(test=ast.Compare(left=ast.Name(id='n',ctx=ast.Load()),ops=[ast.Eq()],comparators=[ast.Constant(1)]),
                                 body=ast.Constant('trace'),orelse=ast.Constant('bypass'))
        return node


def worker(rank,port,args):
    import poc_dllm_ep_comm.scripts.run_dllm_ep4 as old
    import poc_dllm_ep_comm.contract as shared_contract
    # Pure lossless JSON formatting, no change to captured values or model.
    def compact_write(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,separators=(',',':'))+'\n')
    old.ROOT=ROOT;old.write=compact_write;shared_contract.write=compact_write
    source=inspect.getsource(old.worker)
    tree=AmendWorker().visit(ast.parse(source));ast.fix_missing_locations(tree)
    exec(compile(tree,str(COMM/'scripts/run_dllm_ep4.py')+'[bounded-long-cap]','exec'),old.__dict__)
    old.worker(rank,port,args)


def physical_main(args):
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    try:mp.spawn(worker,args=(port,args),nprocs=4,join=True)
    except BaseException as error:
        new_json(Path(args.output)/'status.json',dict(ok=False,status='RUNTIME_UNAVAILABLE',error=str(error)))
        raise
    new_json(Path(args.output)/'status.json',dict(ok=True,status='COMPLETE',mask=MASK))


def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');p.add_argument('--output');
    p.add_argument('--limit',type=int,default=32);p.add_argument('--secondary',action='store_true');a=p.parse_args()
    if a.physical:return physical_main(a)
    out=ROOT/'raw/long_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=MASK,PYTHONPATH=str(PROJECT)+':'+str(DINFER/'python'),
        DLLM_PERSIST_OUTPUT=str(out),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        SGL_ENABLE_JIT_DEEPGEMM='false',SGLANG_ENABLE_JIT_DEEPGEMM='false',
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='1024',OMP_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1')
    for k in list(env):
        if k.startswith(('LLADA_EP_TRACE','LLADA_BLOCK_TRACE','LLADA_DENOISE_TRACE','FLASHVEP_','SD_EP_','AGENT_EP_')):env.pop(k)
    with (out/'full.log').open('x') as log:
        code=subprocess.call([PYTHON,'-m','poc_dllm_request_persistent_replica.scripts.gpu_session','--timeout','3600',
                              PYTHON,'-m','poc_dllm_request_persistent_replica.scripts.collect_long_horizon','--physical',
                              '--output',str(out),'--limit',str(a.limit)],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('LONG_CAPTURE_EXIT',code,out,flush=True);raise SystemExit(code)


if __name__=='__main__':main()
