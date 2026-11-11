"""Retain the actual 4096-token context limit at the last natural candidate."""
import hashlib,json,random,zipfile
from .common import *


def main():
    initialize()
    from transformers import AutoTokenizer
    from ..looped_pd_atomic_prefill.common import MODEL_PATHS
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL_PATHS['huginn']),trust_remote_code=True,local_files_only=True)
    archive=Path('/home/esjung/.cache/huggingface/hub/datasets--THUDM--LongBench/blobs/cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64')
    pools={split:json.loads((ROOT/'data'/f'natural_{split}_huginn_L3900.json').read_text()) for split in ('calibration','discovery')}
    targets={r['source_SHA256']:r for pool in pools.values() for r in pool};found={}
    with zipfile.ZipFile(archive) as z:
        for path in z.namelist():
            if not any(path.endswith('/'+t+'.jsonl') or path==t+'.jsonl' for t in ('gov_report','qmsum','narrativeqa','musique')):continue
            for line in z.read(path).decode().splitlines():
                row=json.loads(line);content='Document:\n'+row['context']+'\nQuestion:\n'+row.get('input','')
                sha=hashlib.sha256(content.encode()).hexdigest()
                if sha not in targets:continue
                ids=tokenizer.apply_chat_template([dict(role='user',content=content)],tokenize=True,add_generation_prompt=True,return_dict=False)
                assert len(ids)>=4080
                old=targets[sha];assert ids[:3900]==old['input_ids']['huginn']
                found[sha]=dict(old,input_ids=dict(huginn=ids[:4080]),length=4080,
                    context_contract=dict(actual_input_tokens=4080,max_output_tokens=16,actual_total=4096,
                        verified_limit=4096,chat_special_tokens_already_in_input_ids=True))
            if len(found)==len(targets):break
    assert len(found)==152
    for split,pool in pools.items():
        path=ROOT/'data'/f'natural_{split}_huginn_L4080.json';rows=[found[r['source_SHA256']] for r in pool]
        if path.exists():assert json.loads(path.read_text())==rows
        else:write(path,rows)
    reference='edge_huginn_L4080_native_reference_state8'
    cfg=json.loads((ROOT/'configs/reserve_huginn_L3900_native_reference_state8.json').read_text())
    cfg.update(name=reference,length=4080,state_capacity_tokens=4080,chunk=4080,token_budget=8160,
        D_token_budget=4080,P_prefill_graph_rows=8160,D_prefill_graph_rows=4080,kv_blocks=23296,
        dataset='natural_calibration_huginn_L4080.json',prompt_ids=[r['prompt_id'] for r in pools['calibration'][:8]],
        phase='NATURAL_CONTEXT_EDGE_NATIVE_EXACT_STATE',required_qualification=None,
        qualification_fresh_native=True,state_hash_defer_until_close=False,state_pinned_slots=6)
    for key in ('campaign_state_reference','qualification_output_reference','qualification_state_subset_reference'):
        cfg.pop(key,None)
    cfg['comparison_reference']=None
    save(cfg)
    parent='reserve_capacity_huginn_L3900_B12_start1';cfg=json.loads((ROOT/'configs'/(parent+'.json')).read_text())
    name='edge_capacity_huginn_L4080_B12_start1';cfg.update(name=name,length=4080,
        state_capacity_tokens=4080,chunk=4080,token_budget=8160,D_token_budget=4080,kv_blocks=23296,
        capacity_native_prefill_geometry=True,
        P_prefill_graph_rows=8160,D_prefill_graph_rows=4080,
        dataset='natural_calibration_huginn_L4080.json',reference=reference,required_qualification=reference)
    save(cfg)
    write(ROOT/'state/huginn_context_edge_plan.json',dict(jobs=[reference,name],state8=reference,capacity=name,
        scope='Last legal homogeneous input candidate. Output remains sixteen. Actual input already contains chat special tokens. No context extension, filler, model change, or relaxed SLO.'))
    print(reference,name)


if __name__=='__main__':main()
