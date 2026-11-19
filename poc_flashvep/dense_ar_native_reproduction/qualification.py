"""Qualify standard continuous workers before any Main performance sweep."""
import json
import time
from .common import ROOT,write,digest,save,initialize
from .run import execute


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def configuration(name,method,dataset,count=8,**overrides):
    value=dict(name=name,method=method,entry_module='poc_flashvep.dense_ar_native_reproduction.serving',
        gpus=[0,1],dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=count,
        token_budget=8192,max_num_seqs=1024,batch_invariant=False,kv_layout='HND',output_cap=2048,
        arrival_offsets=[0.]*count,serial=True,state_capture=True,correctness_only=True,
        protocol='OFFICIAL_PROXY',wall_timeout=1800,serving_timeout=1200)
    value.update(overrides);return value


def main():
    initialize();data=json.loads((ROOT/'data/longbench_v1_mixed200.json').read_text())
    # Fixed task and length coverage. Selection does not read generated outputs.
    selected=[]
    for task in ('hotpotqa','musique','narrativeqa','gov_report','qmsum'):
        selected.append(next(r for r in data if r['task']==task))
    for row in sorted(data,key=lambda r:r['input_tokens']):
        if row not in selected:selected.append(row);break
    for row in sorted(data,key=lambda r:r['input_tokens'],reverse=True):
        if row not in selected:selected.append(row);break
    selected.append(next(r for r in data if r not in selected))
    path=ROOT/'data/qualification8.json'
    if path.exists():assert json.loads(path.read_text())==selected
    else:write(path,selected)
    names=[]
    for label,method,extra in (
        ('Standard','Native',dict(reference_single=True,gpus=[1])),
        ('Native','Native',{}),
        ('Agg','Agg',dict(serial=False,arrival_offsets=[float(i) for i in range(8)])),
    ):
        name=f'qual_{label}_LongBench_N8_B8192_S1024_HND_BI0_v2'
        assert execute(save(configuration(name,method,path.name,**extra))),name
        names.append(name)
    teacher={r['prompt_id']:r for r in records(ROOT/'raw'/names[0]/'requests.jsonl')}
    independent=json.loads((ROOT/'raw'/'standard_HND_BI0_LongBench200_natural_C1_cap2048_v1'/'natural_reference.json').read_text())
    gold={r['prompt_id']:r for r in independent['rows']};result={}
    state=json.loads((ROOT/'raw'/names[0]/'prompt_state_single.json').read_text())['requests']
    for name in names:
        folder=ROOT/'raw'/name;rows=records(folder/'requests.jsonl')
        outputs=[dict(prompt_id=r['prompt_id'],reference=r['output_ids']==gold[r['prompt_id']]['output_ids']) for r in rows]
        states=[]
        for file in folder.glob('prompt_state_*.json'):
            role=file.stem.removeprefix('prompt_state_')
            for rid,value in json.loads(file.read_text())['requests'].items():
                if not rid.startswith('q'):continue
                expected=state[rid]
                states.append(dict(request_id=rid,role=role,KV=value['KV']==expected['KV'],
                    hidden=value.get('final_hidden_SHA256')==expected.get('final_hidden_SHA256') if role!='decode' else None))
        lifetime={p.name:json.loads(p.read_text())['passed'] for p in folder.glob('KV_lifetime_*.json')}
        result[name]=dict(outputs=outputs,states=states,lifetime=lifetime,
            passed=len(outputs)==8 and all(r['reference'] for r in outputs) and all(r['KV'] and r['hidden'] is not False for r in states) and bool(lifetime) and all(lifetime.values()))
    write(ROOT/'analysis/standard_Native_Agg_N8_qualification_v1.json',dict(time=time.time(),results=result,evidence='CORRECTNESS_ONLY'))
    print('QUALIFICATION',json.dumps({k:v['passed'] for k,v in result.items()}),flush=True)


if __name__=='__main__':main()
