import argparse
import os
import subprocess
from poc_dllm_ep_comm.contract import ROOT,PROJECT,PYTHON,DINFER,MASK
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--limit',type=int,default=32)
    p.add_argument('--secondary',action='store_true');a=p.parse_args();out=ROOT/'raw'/a.name
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=MASK,PYTHONPATH=str(PROJECT)+':'+str(DINFER/'python'),
        DLLM_COMM_OUTPUT=str(out),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        SGL_ENABLE_JIT_DEEPGEMM='false',SGLANG_ENABLE_JIT_DEEPGEMM='false',
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='1024',OMP_NUM_THREADS='4',
        TORCHDYNAMO_DISABLE='1')
    for k in list(env):
        if k.startswith(('LLADA_EP_TRACE','LLADA_BLOCK_TRACE','LLADA_DENOISE_TRACE','FLASHVEP_','SD_EP_','AGENT_EP_')):env.pop(k)
    command=[PYTHON,'-m','poc_dllm_ep_comm.scripts.run_dllm_ep4','--output',str(out),'--limit',str(a.limit)]
    if a.secondary:command.append('--secondary')
    with (out/'full.log').open('x') as log:
        code=subprocess.call([PYTHON,'-m','poc_dllm_ep_comm.scripts.gpu_session','--timeout','2400',*command],
            cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('EXIT',code,'LOG',out/'full.log');raise SystemExit(code)
if __name__=='__main__':main()
