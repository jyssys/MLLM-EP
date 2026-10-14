"""Reuse scoped watchdog/restoration, without editing a prior PoC."""
import argparse
import os
from poc_looped_moe_ep.contract import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--timeout',type=int,default=600)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    import poc_dllm_attn_moe_pipeline.scripts.gpu_session as guard
    import poc_multibd_moe_ep.scripts.run_guarded as cleanup
    guard.PROJECT=cleanup.PROJECT=PROJECT
    guard.MASK=cleanup.MASK=MASK;cleanup.ROOT=ROOT
    output=ROOT/a.output
    assert not output.exists(),'Preserve every completed/failed run; use a new output directory'
    output.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=MASK,OMP_NUM_THREADS='2',
        PYTHONPATH=str(PROJECT)+':/home/esjung/external/Diffulex-multibd-poc',
        POC_MULTIBD_OUTPUT=str(output),TOKENIZERS_PARALLELISM='false',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    with (output/'console.log').open('w') as log:
        try:
            code=guard.guarded_run(command,env,log,output,a.timeout)
            status=dict(success=code==0,exit_code=code)
        except (Exception,KeyboardInterrupt) as e:
            status=dict(success=False,error=repr(e))
        finally:cleanup.finish_scoped_cleanup(output)
    write(output/'guard_status.json',status)
    print(status,flush=True)
    raise SystemExit(0 if status['success'] else 1)


if __name__=='__main__':main()
