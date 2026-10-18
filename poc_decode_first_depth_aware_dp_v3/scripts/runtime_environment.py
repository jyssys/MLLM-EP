"""Record the reused runtime without allocating a CUDA context."""
import platform
import subprocess
from poc_decode_first_depth_aware_dp.runtime import configure_imports
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    configure_imports()
    import torch,transformers,tokenizers
    write(ROOT/'artifacts/runtime_environment.json',dict(
        python=platform.python_version(),torch=torch.__version__,CUDA=torch.version.cuda,
        transformers=transformers.__version__,tokenizers=tokenizers.__version__,
        model=MODEL_ID,revision=REVISION,model_config=read(MODEL/'config.json'),
        model_config_sha256=sha(MODEL/'config.json'),runtime=CDB_REVISION,
        CDB_dirty=subprocess.check_output(['git','status','--porcelain'],cwd=CDB,text=True),
        independent_DP4=True,per_process_logical_device=0,physical_GPUs=[4,5,6,7],
        prefix_caching=False,graphs_decode=True,graphs_prefill=False,
        online_gate=True,synthetic_exit_replay=False,async_batching=False,
        min_depth=2,max_depth=4,q=.5,cache_policy='last_exited',cache_slots=96,
        local_batch_cap=16,main_cache_blocks=2048,block_size=16,max_model_len=8192,
        gate_heads_unused=['preloop_exit_gate','lookahead_exit_gate']))

if __name__=='__main__':main()
