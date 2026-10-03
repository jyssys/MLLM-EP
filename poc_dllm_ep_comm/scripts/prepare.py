"""Freeze inputs/configuration before any acceptance or routing observations."""
import json
import subprocess
import random
from poc_dllm_ep_comm.contract import ROOT,PROJECT,DINFER,MODEL,SPEC,REVISION,write,sha,MASK

def main():
    destination=ROOT/'results/frozen_manifest.json'
    if destination.exists():raise FileExistsError(destination)
    source=PROJECT.parent/'MLLM-EP-llada2-flash-100b/poc_llada2_flash_ep/data/bounded_eval_32'
    rows=[];sources={}
    for domain in ('gsm8k','humaneval'):
        path=source/(domain+'_32.json');data=json.loads(path.read_text())['details']
        order=list(range(len(data)));random.Random(3407).shuffle(order)
        sources[domain]=dict(path=str(path),sha256=sha(path),origin='openai/'+('gsm8k' if domain=='gsm8k' else 'openai_humaneval'))
        for n,index in enumerate(order[:16]):
            rows.append(dict(request_id=f'{domain}_{n:02d}',domain=domain,source_index=index,prompt=data[index]['prompt']))
    write(destination,dict(rows=rows,sources=sources,selection='metadata-only seeded permutation3407, first16/domain; no result selection',
        model='inclusionAI/LLaDA2.0-mini',revision=REVISION,dtype='BF16',batch=1,
        primary=dict(block_length=32,threshold=.9,temperature=0,generation_budget=64,cache='prefix',early_stop=True,
            decoder='native ThresholdParallelDecoder',maximum_unroll=1,expected_tpf=15,remasking='low_confidence',
            source='pinned dInfer README LLaDA2-mini command; bounded generation, eager graph-off characterization'),
        secondary=dict(block_length=64,requests_per_domain=4,selection='first4 of frozen primary/domain; same threshold'),
        metrics=dict(primary='all actual runtime logical routed D+C bytes; live-only masked output positions before forward; one-shot final accepted output routes',
            eos_sensitivity='also report final-returned-position-only live bytes, excluding final EOS/tail; and all materialized accepted-position denominator',
            timing_validity='timing versus bypass overhead <=10%, exact token IDs and NFE parity required',
            bootstrap='2000 request bootstrap, paired block configurations grouped by original problem ID',
            gates=dict(residual_no_go=1.5,residual_go=2.0,comm_no_go=.10,comm_go=.15))))
    def command(args):return subprocess.check_output(args,text=True)
    files=['README.md','python/dinfer/model/modeling_llada2_moe_sglang.py','python/dinfer/decoding/generate_uniform.py',
        'python/dinfer/decoding/parallel_strategy.py','python/dinfer/decoding/utils.py']
    write(ROOT/'results/environment.json',dict(gpu_mask=MASK,gpus=command(['nvidia-smi','--id='+MASK,
        '--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu','--format=csv,noheader']),
        topology=command(['nvidia-smi','topo','-m']),gpu_processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),
        git_sha=command(['git','rev-parse','HEAD']).strip(),model_config=json.loads((MODEL/'config.json').read_text()),
        model_file_hashes={p.name:sha(p) for p in MODEL.glob('*') if p.is_file() and p.suffix!='.safetensors'},
        dinfer_upstream=command(['git','-C',str(DINFER),'rev-parse','HEAD']).strip(),
        dinfer_loaded_source_hashes={f:sha(DINFER/f) for f in files},spec_sha256=sha(SPEC),
        original_dirty_staged_sha='7018be31bd8a517574711055a72afbd6ae90737e4a585578c98ec255dbf7e1d9',
        original_dirty_unstaged_sha='f56c2cbd47c4995364f072907ad7d886c935c3afe55d90e175a65abf0a4d941e'))
    write(ROOT/'results/runtime_inventory.json',dict(selected='existing SGLang/dInfer DeepEP normal BF16 bridge, LLaDA2-mini cached16B',
        alternatives={'LLaDA-MoE-7B':'exists; older native EP4 naive broadcast runner replicated four user inputs and cross-topology BF16 drift; not selected',
            'SDAR-30B':'checkpoint and TEAM environment exist; assignment-level/reference EP path, larger than mini',
            'LLaDA2-Flash-100B':'existing validated DeepEP bridge reused, same mini model implementation, avoiding100B loading',
            'Epoch':'prior paper audited, no full reimplementation or measured comparison',
            'agentic_EP':'destination-dedup counting convention reused; Qwen timing/shape caches NOT reused'},
        no_environment_upgrade=True,no_framework_rewrite=True))
    print('FROZEN',len(rows))

if __name__=='__main__':main()
