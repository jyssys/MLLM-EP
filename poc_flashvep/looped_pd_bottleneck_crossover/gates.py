"""Require actual output, valid-cache, hidden-state, and work accounting evidence."""
import sys
from .common import *

def check(name):
    folder=ROOT/'raw'/name;status=json.loads((folder/'status.json').read_text())
    assert status['status']=='COMPLETE',status
    cohorts=[json.loads(s) for s in (folder/'cohorts.jsonl').read_text().splitlines()]
    assert cohorts and all(r.get('native_output_parity') for r in cohorts)
    assert all(r['full_depth_rows']==r['full_depth_rows_observed'] for r in cohorts)
    states={}
    for path in folder.glob('prompt_states_*.jsonl'):
        for s in path.read_text().splitlines():
            r=json.loads(s);key=(r['seed'],r['prompt_tokens'])
            value=dict(hidden=r['hidden_sha256'],planes=r['planes'])
            states.setdefault(key,[]).append(value)
    matched=0
    for key,values in states.items():
        if len(values)<2:continue
        assert all(v==values[0] for v in values),f'Prompt state mismatch: {key}'
        matched+=1
    assert matched>=sum(r['batch'] for r in cohorts),'Missing native-versus-barrier state evidence'
    result=dict(status='PASS_INTERNAL_BARRIER_CORRECTNESS',name=name,model=cohorts[0]['model'],
        compared_output_sequences=sum(r['batch'] for r in cohorts),matched_prompt_state_groups=matched,
        scope='Finite same-engine barrier gate. Not original-HF equivalence or downstream accuracy.',time=time.time())
    write(ROOT/'state'/f'{name}_gate.json',result);return result

def native_pd(model):
    name=f'gate_{model}_native_pd_v1';folder=ROOT/'raw'/name
    assert json.loads((folder/'status.json').read_text())['status']=='COMPLETE'
    rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines() if not json.loads(s)['warmup']]
    ref=ROOT/'raw'/f'gate_{model}_barrier_v3'/'references.jsonl'
    reference=json.loads(ref.read_text().splitlines()[0])['outputs']
    expected={int(k.rsplit('_q',1)[1]):v for k,v in reference.items()}
    assert len(rows)==2
    for r in rows:
        assert r['output_ids']==expected[r['sample_id']]['output_ids']
        assert r['exit_depths']==expected[r['sample_id']]['depths']
    checks=[json.loads(s) for p in folder.glob('KV_checks-*.jsonl') for s in p.read_text().splitlines()]
    assert len(checks)>=2 and all(r['passed'] for r in checks)
    def state_map(directory):
        out={}
        for p in directory.glob('prompt_states_*.jsonl'):
            for s in p.read_text().splitlines():
                r=json.loads(s);out[(r['seed'],r['prompt_tokens'])]=(r['hidden_sha256'],r['planes'])
        return out
    native=state_map(ROOT/'raw'/f'gate_{model}_barrier_v3');remote=state_map(folder)
    for r in rows:
        key=(r['seed'],r['input_tokens']);assert remote.get(key)==native.get(key) and key in remote,'Native-PD prompt state differs from local native state'
    result=dict(status='PASS_NATIVE_PD_TRANSPORT',model=model,output_sequences=2,valid_KV_hidden_byte_checks=len(checks),
        scope='Small initial transport gate. Not full qualification or HF equivalence.',time=time.time())
    write(ROOT/'state'/f'{model}_native_pd_gate.json',result);return result

def qualification(model,count=32,folder_name=None,native_name=None):
    folder=ROOT/'raw'/(folder_name or f'qualification_{model}_pd32_v1')
    native=ROOT/'raw'/(native_name or f'qualification_{model}_native_reference32_v1')
    for p in (folder,native):assert json.loads((p/'status.json').read_text())['status']=='COMPLETE'
    reference=json.loads((native/'references.jsonl').read_text().splitlines()[0])
    expected={int(k.rsplit('_q',1)[1]):v for k,v in reference['outputs'].items()}
    rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines() if not json.loads(s)['warmup']]
    assert len(rows)==count
    for r in rows:
        assert r['output_ids']==expected[r['sample_id']]['output_ids'],('PD output mismatch',model,r['sample_id'])
        assert r['exit_depths']==expected[r['sample_id']]['depths']
    checks=[json.loads(s) for p in folder.glob('KV_checks-*.jsonl') for s in p.read_text().splitlines()]
    assert len(checks)>=count and all(r['passed'] for r in checks)
    def states(directory):
        result={}
        for p in directory.glob('prompt_states_*.jsonl'):
            for s in p.read_text().splitlines():
                r=json.loads(s);result[(r['seed'],r['prompt_tokens'])]=(r['hidden_sha256'],r['planes'])
        return result
    a=states(native);b=states(folder)
    assert all(a[(r['seed'],r['input_tokens'])]==b[(r['seed'],r['input_tokens'])] for r in rows)
    assert reference['full_depth_rows']==reference['full_depth_rows_observed']
    before=json.loads((folder/'before_measured.json').read_text())
    after=json.loads((folder/'after_measured.json').read_text())
    observed=sum(v['issued_loop_rows']-before[k]['issued_loop_rows'] for k,v in after.items())
    expected=sum((r['input_tokens']+r['output_tokens']-1)*MODEL_PINS[model][2] for r in rows)
    assert observed==expected,('PD full-depth work mismatch',observed,expected)
    result=dict(status=f'PASS_{count}_NATIVE_PD_QUALIFICATION',model=model,output_sequences=count,full_KV_hidden_checks=count,
        work_accounting=reference['full_depth_rows_observed'],PD_issued_loop_rows=observed,PD_expected_loop_rows=expected,
        scope='Independent finite 32-request transport and output qualification, not original-HF fidelity.',time=time.time())
    result['scope']=f'Finite {count}-request transport/output qualification at input {rows[0]["input_tokens"]}. Not original-HF fidelity.'
    write(ROOT/'state'/f'{model}_qualification{count}.json',result);return result

if __name__=='__main__':
    for name in sys.argv[1:]:
        if name.startswith('long:'):
            m=name[5:];result=qualification(m,2,f'qualification_{m}_long_pd_v1',f'qualification_{m}_long_reference_v1')
        elif name.startswith('qual:'):result=qualification(name[5:])
        elif name.startswith('pd:'):result=native_pd(name[3:])
        else:result=check(name)
        print(json.dumps(result),flush=True)
