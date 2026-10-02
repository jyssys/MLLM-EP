"""Read-only audit; original dirty worktree remains untouched."""
import importlib.metadata
import json
import subprocess
from pathlib import Path
from poc_agentic_ep.contract import ROOT,MASK,TARGET,TARGET_REV,SPEC,sha,write


def command(*args):return subprocess.check_output(args,text=True)


def main():
    path=ROOT/'results/environment.json'
    if path.exists():raise FileExistsError(path)
    original=Path('/home/esjung/MLLM-EP-github')
    def diff_hash(cached=False):
        import hashlib
        return hashlib.sha256(subprocess.check_output(['git','diff',*(['--cached'] if cached else []),'--binary'],cwd=original)).hexdigest()
    config=json.loads((Path(TARGET)/'config.json').read_text())
    assert (config['num_hidden_layers'],config['hidden_size'],config['num_experts'],config['num_experts_per_tok'],config['moe_intermediate_size'])==(48,2048,128,8,768)
    packages={n:importlib.metadata.version(n) for n in ('torch','vllm','deep_ep','transformers','triton')}
    write(path,dict(mask=MASK,model=TARGET,revision=TARGET_REV,config=config,
        git_sha=command('git','rev-parse','HEAD').strip(),packages=packages,
        nvidia_smi=command('nvidia-smi'),topology=command('nvidia-smi','topo','-m'),
        gpu_map=command('nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'),
        process_snapshot=command('nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory','--format=csv,noheader'),
        installed_gpu_runner_sha256=sha('/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py'),
        original_worktree=dict(staged_sha256=diff_hash(True),unstaged_sha256=diff_hash(False)),
        spec_path=str(SPEC),spec_sha256=sha(SPEC),
        frozen_analysis_plan=dict(primary='new_prefill',secondary='teacher_forced_decode32',
            independent_unit='trajectory',bootstrap=2000,seed=3407,
            hot_expert_M=[4,8,16,32],replica_slots_per_layer_per_rank=[1,2,4],
            memory_GiB_per_rank=[.5,1,2,4],tool_gaps_ms=[50,100,250,500,1000],
            gate1='adjacent-control cosine >=.05 OR coverage>=5pp, CI>0 for both controls',
            gate2='previous-global coverage@16 >=5pp, CI>0',
            gate3='oracle remote useful bytes >=15% at feasible budget; structural only, no speedup',
            gate4='previous remote useful bytes>=10% OR oracle recovery>=50%',
            gate5='useful budget preparation conservative serial-copy estimate<=500ms',
            topology='existing TP2-DP2-EP4; spec TP1 if compatible is a recommendation, no runtime rewrite')))
    print('AUDIT_SAVED')


if __name__=='__main__':main()
