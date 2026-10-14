"""Use existing exact-mask burner restoration and scoped orphan cleanup."""
import argparse
import os
from poc_dllm_prefix_canvas_cp.contract import *


def finish_this_run(cleanup,output):
    # The inherited helper matches ROOT-relative ownership markers. Narrow
    # ROOT to this output, not the entire PoC, before looking for orphans.
    cleanup.ROOT=Path(output)
    cleanup.finish_scoped_cleanup(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    import poc_dllm_attn_moe_pipeline.scripts.gpu_session as guard
    import poc_multibd_moe_ep.scripts.run_guarded as cleanup
    guard.PROJECT=cleanup.PROJECT=PROJECT;guard.MASK=cleanup.MASK=MASK;cleanup.ROOT=ROOT
    output=ROOT/a.output;output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=str(PROJECT)+':'+str(DINFER/'python'),POC_MULTIBD_OUTPUT=str(output)+'/worker',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for k in list(env):
        if k.startswith(('LLADA_EP_TRACE','LLADA_BLOCK_TRACE','LLADA_DENOISE_TRACE','FLASHVEP_','SD_EP_','AGENT_EP_')):env.pop(k)
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    write(output/'run_contract.json',dict(command=command,mask=MASK,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py')},
        owner_marker=env['POC_MULTIBD_OUTPUT'],restoration='only exact pre-existing same-user mask'))
    with (output/'console.log').open('w') as log:
        try:code=guard.guarded_run(command,env,log,output,a.timeout);status=dict(success=code==0,code=code)
        except (Exception,KeyboardInterrupt) as e:status=dict(success=False,error=repr(e))
        finally:finish_this_run(cleanup,output)
    write(output/'guard_status.json',status);print(status,flush=True)
    raise SystemExit(0 if status['success'] else 1)


if __name__=='__main__':main()
