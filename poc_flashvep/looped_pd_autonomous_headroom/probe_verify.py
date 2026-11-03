"""Compare real D-local probes with exact native-PD prompt references."""
import copy
from .common import *
from .run import execute
from .campaign import gate


def main():
    while time.time()<GPU_DEADLINE-900:
        p=ROOT/'state/probe_followup.json'
        if p.exists() and json.loads(p.read_text())['stage'] in ('PROBE_RESULTS_AVAILABLE','FAILED_REVIEW_REQUIRED'):break
        time.sleep(5)
    results=[]
    for model in ('ouro','huginn'):
        name=f'D_probe_{model}_tokens16_n24';folder=ROOT/'raw'/name
        probe=folder/'D_probe_requests.jsonl'
        if not probe.exists():continue
        rows=[json.loads(s) for s in probe.read_text().splitlines()]
        if not rows:continue
        c=copy.deepcopy(json.loads((ROOT/'configs'/(name+'.json')).read_text()))
        c.pop('D_probe_tokens',None)
        c.update(name=f'probe_reference_{model}_tokens16',length=16,cap=1,count=len(rows),
            state_capacity_tokens=16,seed=rows[0]['seed'],request_caps=[1]*len(rows),arrival_offsets=[0.]*len(rows),
            phase='correctness',verify_transfer=True,state_probe=True,dataset=f'probe_reference_{model}.json')
        data=[dict(prompt_id=f'probe_reference_{i}',input_ids={model:r['input_ids']}) for i,r in enumerate(rows)]
        c['prompt_ids']=[r['prompt_id'] for r in data]
        write(ROOT/'data'/c['dataset'],data);write(ROOT/'configs'/(c['name']+'.json'),c)
        okay=execute(c['name'])
        if okay:
            ref=[json.loads(s) for s in (ROOT/'raw'/c['name']/'requests.jsonl').read_text().splitlines()]
            ref={r['sample_id']:r for r in ref if not r.get('warmup')}
            okay=gate(c['name']) and all(ref[i]['output_ids']==r['output_ids'] and ref[i]['exit_depths']==r['exit_depths'] for i,r in enumerate(rows))
        before=json.loads((folder/'before_measured.json').read_text());after=json.loads((folder/'after_measured.json').read_text())
        requests=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()];requests=[r for r in requests if not r.get('warmup')]
        expected=sum((r['input_tokens']+r['output_tokens']-1)*c['depth'] for r in requests)+sum(len(r['input_ids'])*c['depth'] for r in rows)
        observed=sum(v['issued_loop_rows']-before[k]['issued_loop_rows'] for k,v in after.items())
        result=dict(model=model,probe_reference_exact=okay,completed_probes=len(rows),
            expected_recurrent_rows=expected,observed_recurrent_rows=observed,full_work=expected==observed,
            scope='Correctness only. Same-window P saturation and Decode SLO are separate gates.')
        write(folder/'probe_correctness.json',result);results.append(result)
    write(ROOT/'state/probe_verification_results.json',results)


if __name__=='__main__':main()
