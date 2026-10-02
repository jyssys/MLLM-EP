"""Read-only runtime audit; freeze data before observing generation/routes."""
import json
import os
import random
import subprocess
from pathlib import Path

from poc_speculative_moe_ep.contract import ROOT, SPEC, TARGET, DRAFT, TARGET_REV, DRAFT_REV, MASK, write, sha


def main():
    import torch
    import vllm
    import deep_ep
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi, hf_hub_download
    envpath = ROOT/'artifacts/environment.json'
    if (ROOT/'artifacts/frozen_pool.json').exists():
        raise FileExistsError('cohort already frozen')
    tc = json.loads((Path(TARGET)/'config.json').read_text())
    dc = json.loads((Path(DRAFT)/'config.json').read_text())
    expected = dict(num_hidden_layers=48, hidden_size=2048, num_experts=128,
                    num_experts_per_tok=8, moe_intermediate_size=768)
    for k, v in expected.items():
        assert tc[k] == v, (k, tc[k])
    tt, dt = (AutoTokenizer.from_pretrained(p, local_files_only=True) for p in (TARGET,DRAFT))
    assert tt.get_vocab() == dt.get_vocab(), 'tokenizer ID mapping mismatch'
    examples = ['def f(x):\n    return x + 1', 'Solve: 6 apples cost $3.50.', '<|im_start|>assistant\n']
    assert all(tt.encode(x) == dt.encode(x) for x in examples)
    smi = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True)
    mapping = []
    for logical, physical in enumerate(map(int, MASK.split(','))):
        row = next(x for x in smi.splitlines() if int(x.split(',')[0])==physical)
        mapping.append(dict(logical_cuda=logical, intended_ep_rank=logical, physical_gpu=physical,
                            nvidia_smi=row))
    versions = dict(torch=torch.__version__, cuda=torch.version.cuda, vllm=vllm.__version__,
                    deep_ep=getattr(deep_ep,'__version__','UNAVAILABLE'), nccl=list(torch.cuda.nccl.version()))
    source = Path(vllm.__file__).parent
    files = ['config/speculative.py','v1/spec_decode/draft_model.py','v1/sample/rejection_sampler.py',
             'v1/worker/gpu_model_runner.py','v1/core/sched/scheduler.py']
    audit = dict(git_base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        spec_path=str(SPEC), spec_sha256=sha(SPEC), versions=versions, target_config=tc,draft_config=dc,
        physical_logical_mapping=mapping, topology='TP2-DP2-EP4-PP1',
        topo=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        process_snapshot=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True),
        source_hashes={str(source/p):sha(source/p) for p in files}, tokenizer_equal=True,
        target_revision=TARGET_REV,draft_revision=DRAFT_REV, tokenizer_vocab_entries=len(tt),
        tokenizer_file_hashes={p:{f:sha(Path(p)/f) for f in ('tokenizer.json','tokenizer_config.json')} for p in (TARGET,DRAFT)})
    if not envpath.exists():write(envpath,audit)
    else:assert json.loads(envpath.read_text())['tokenizer_file_hashes']==audit['tokenizer_file_hashes']
    write(ROOT/'artifacts/sd_config.json', dict(target=TARGET,draft=DRAFT,
        target_revision=TARGET_REV,draft_revision=DRAFT_REV, draft_tp=2, target_tp=2,
        gamma=[0,2,4,8], temperature=0., output_cap=64, eos_enabled=True,
        rule='native vLLM greedy longest-prefix accept + target correction/bonus',
        exact_kv_rollback='native vLLM scheduler/GPUModelRunner', seed=3407,
        expert_top_k=8, eplb=False,redundant_experts=0,dbo=False,prefix_caching=False,eager=True,
        async_scheduling=False, backend='deepep_high_throughput',
        discovery_gate_requests=16, intended_pool=96, request_timeout_sec=180,
        initial_smoke_tokens=16, new_online_policy=False))
    def read_parquet(path):
        # Use existing CPU environment; do not mutate the validated GPU venv.
        code='import json,sys,pyarrow.parquet as pq; print(json.dumps(pq.read_table(sys.argv[1]).to_pylist()))'
        return json.loads(subprocess.check_output(['/home/esjung/anaconda3/bin/python','-c',code,str(path)],text=True))
    gsm = read_parquet('/home/esjung/.cache/huggingface/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet')
    info = HfApi().dataset_info('openai/openai_humaneval')
    parquet = next(x.rfilename for x in info.siblings if x.rfilename.endswith('.parquet'))
    human = read_parquet(hf_hub_download('openai/openai_humaneval',parquet,repo_type='dataset',revision=info.sha))
    frozen = []
    for dataset, rows, count, revision in [('gsm8k',gsm,64,'740312add88f781978c0658806c59bc2815b9866'),('humaneval',human,32,info.sha)]:
        indices=list(range(len(rows)));random.Random(3407).shuffle(indices)
        for idx in indices[:count]:
            row=rows[idx]
            content = ('Solve this problem. Give a concise solution.\n'+row['question']) if dataset=='gsm8k' else ('Complete the following Python function. Return code only.\n'+row['prompt'])
            encoded=tt.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            ids=list(encoded['input_ids']) if hasattr(encoded,'keys') else list(encoded)
            frozen.append(dict(request_id=f'{dataset}:{idx}',dataset=dataset,source_id=row.get('task_id',str(idx)),
                source_revision=revision,prompt_token_ids=ids,prompt_tokens=len(ids)))
    # Balanced interleave; the first 16 are the frozen initial evidence gate.
    a=[r for r in frozen if r['dataset']=='gsm8k'];b=[r for r in frozen if r['dataset']=='humaneval']
    ordered=[r for pair in zip(a,b) for r in pair]+a[len(b):]
    write(ROOT/'artifacts/frozen_pool.json',dict(seed=3407,rows=ordered,selection='seeded before generation/routes; no answer selection'))
    print(json.dumps(dict(versions=versions,tokenizer_equal=True,pool=len(ordered),human_revision=info.sha)))


if __name__=='__main__':main()
