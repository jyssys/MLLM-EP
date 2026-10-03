"""One bounded session: revalidate existing TRUE EP4 runner, then measure P2P."""
import os
import subprocess
from poc_dllm_block_local_replica.contract import *


def main():
    out=ROOT/'raw/physical_revalidation'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=MASK,PYTHONPATH=str(PROJECT)+':'+str(DINFER/'python'),
        DLLM_REPLICA_OUTPUT=str(out),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        SGL_ENABLE_JIT_DEEPGEMM='false',SGLANG_ENABLE_JIT_DEEPGEMM='false',
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK='1024',OMP_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1')
    for k in list(env):
        if k.startswith(('LLADA_EP_TRACE','LLADA_BLOCK_TRACE','LLADA_DENOISE_TRACE','FLASHVEP_','SD_EP_','AGENT_EP_')):env.pop(k)
    with (out/'full.log').open('x') as log:
        command=[PYTHON,'-m','poc_dllm_block_local_replica.scripts.gpu_session','--timeout','900',
                 PYTHON,'-m','poc_dllm_block_local_replica.scripts.physical_work','--output',str(out)]
        code=subprocess.call(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('PHYSICAL_EXIT',code,out,flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
