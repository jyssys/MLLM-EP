"""Record source identity and current hardware without changing execution."""
import argparse
import datetime
import importlib.metadata
import subprocess
from poc_decode_first_depth_aware_dp.contract import *
from poc_decode_first_depth_aware_dp.runtime import configure_imports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists()
    configure_imports()
    import torch,transformers,tokenizers,flash_attn_interface
    sources={}
    for name,repo,revision in [('CDB',CDB,CDB_REVISION),('vllm_rlt_audit',RLT,RLT_REVISION)]:
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==revision
        assert subprocess.run(['git','diff','--quiet'],cwd=repo).returncode==0
        sources[name]=dict(path=str(repo),revision=revision,
            python_sha256={str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob('*.py'))},
            upstream_files_modified=False)
    write(args.output,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,tokenizers=tokenizers.__version__,
        flash_attention_module=flash_attn_interface.__file__,flash_attention_version=importlib.metadata.version('flash-attn-3'),
        transformers_module=transformers.__file__,sources=sources,
        GPU_inventory=subprocess.check_output(['nvidia-smi','--id=4,5,6,7',
            '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        checkpoint_hashes=read(ROOT/'artifacts/checkpoint.json'),
        compatibility_metadata={'pad_token_id':None,'meaning':'Restores the official Transformers 4.57 default missing in the 5.9 config object. No weight or token modification.'},
        unused_gate_heads='Official CDB defines untrained lookahead/preloop heads. They are not selected or executed. Original trained gate identity is asserted.',
        CP_TP_EP='All model workers are TP1/EP1/CP1. Four quality shards are not a DP scheduling benchmark.',
        original_root_spec_overwritten=False))
    print('SOURCE_AUDIT_RECORDED')


if __name__=='__main__':main()
