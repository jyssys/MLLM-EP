"""Reuse exact-mask burner restoration and output-scoped orphan cleanup."""
import argparse
import os
from poc_dllm_multigpu_e2e_atlas.contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    import poc_dllm_attn_moe_pipeline.scripts.gpu_session as guard
    import poc_multibd_moe_ep.scripts.run_guarded as cleanup
    guard.PROJECT=cleanup.PROJECT=PROJECT;guard.MASK=cleanup.MASK=MASK
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False);cleanup.ROOT=out
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(PROJECT)+':'+str(DINFER/'python'),POC_MULTIBD_OUTPUT=str(out)+'/worker',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for key in list(env):
        if key.startswith(('LLADA_EP_','LLADA_BLOCK_TRACE','LLADA_DENOISE_TRACE','FLASHVEP_','SD_EP_','AGENT_EP_')):
            env.pop(key)
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    write(out/'run_contract.json',dict(command=command,mask=MASK,
        source_sha256={str(s.relative_to(ROOT)):sha(s) for s in ROOT.rglob('*.py')},owner_marker=env['POC_MULTIBD_OUTPUT']))
    with (out/'console.log').open('w') as log:
        try:
            code=guard.guarded_run(command,env,log,out,a.timeout);status=dict(success=code==0,code=code)
        except (Exception,KeyboardInterrupt) as error:status=dict(success=False,error=repr(error))
        finally:cleanup.finish_scoped_cleanup(out)
    write(out/'guard_status.json',status);print(status,flush=True)
    raise SystemExit(0 if status['success'] else 1)


if __name__=='__main__':main()
