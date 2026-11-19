"""Test the official invariant numerical path, without legacy norm adapters."""
import json
from .common import ROOT,write,save
from .qualification import configuration,records
from .run import execute


def main():
    names=[]
    for label,method,extra in (
        ('Standard','Native',dict(reference_single=True,gpus=[1])),
        ('Native','Native',{}),
        ('Agg','Agg',dict(serial=False,arrival_offsets=[float(i) for i in range(8)])),
    ):
        name=f'qual_{label}_LongBench_N8_B8192_S1024_HND_BI1_v1'
        assert execute(save(configuration(name,method,'qualification8.json',batch_invariant=True,**extra))),name
        names.append(name)
    teacher={r['prompt_id']:r for r in records(ROOT/'raw'/names[0]/'requests.jsonl')}
    state=json.loads((ROOT/'raw'/names[0]/'prompt_state_single.json').read_text())['requests'];results={}
    for name in names:
        folder=ROOT/'raw'/name;rows=records(folder/'requests.jsonl');checks=[]
        for row in rows:
            rid=f'q{row["sample_id"]}'
            values=dict(request_id=rid,output=row['output_ids']==teacher[row['prompt_id']]['output_ids'])
            for file in folder.glob('prompt_state_*.json'):
                role=file.stem.removeprefix('prompt_state_');captured=json.loads(file.read_text())['requests']
                if rid not in captured:continue
                values[role+'_KV']=captured[rid]['KV']==state[rid]['KV']
                if role!='decode':values[role+'_hidden']=captured[rid]['final_hidden_SHA256']==state[rid]['final_hidden_SHA256']
            checks.append(values)
        lifetime={p.name:json.loads(p.read_text())['passed'] for p in folder.glob('KV_lifetime_*.json')}
        native='qual_Native_' in name
        complete_states=all(sum(k.endswith('_KV') for k in r)==(2 if native else 1) and
            sum(k.endswith('_hidden') for k in r)==1 for r in checks)
        passed=len(checks)==8 and len({r['request_id'] for r in checks})==8 and complete_states and all(all(v for k,v in r.items() if k!='request_id') for r in checks) and all(lifetime.values()) and len(lifetime)==(1 if 'qual_Standard_' in name else 2)
        results[name]=dict(passed=passed,checks=checks,lifetime=lifetime)
    write(ROOT/'analysis/official_invariant_N8_qualification_v1.json',dict(results=results,
        passed=all(r['passed'] for r in results.values()),evidence='CORRECTNESS_ONLY',
        scope='Official batch-invariant numerical family. Standard full-model execution. No repository normalization or token correction. Distinct from stock BI0 performance.'))
    print('INVARIANT_QUALIFICATION',json.dumps({k:v['passed'] for k,v in results.items()}),flush=True)


if __name__=='__main__':main()
