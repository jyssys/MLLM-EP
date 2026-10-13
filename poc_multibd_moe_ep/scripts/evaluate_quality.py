"""Official full EvalPlus inputs/oracles, executed only inside OS isolation."""
import argparse
import ast
import collections
import json
import os
import pickle
import re
import resource
import subprocess
import tempfile
import time
from pathlib import Path
from poc_multibd_moe_ep.contract import *

DEPS=Path('/home/esjung/external/multibd-poc-deps')
BWRAP=Path('/home/esjung/external/multibd-sandbox/unpacked/usr/bin/bwrap')
ENV=Path('/home/esjung/anaconda3/envs/dinfer-ep-poc')


def extract_code(prefix,text):
    text=prefix+text
    text=text.split('<|role_end|>')[0].split('<|im_end|>')[0]
    match=re.search(r'```(?:python|py)?\s*\n(.*?)(?:```|\Z)',text,re.S)
    return (match.group(1) if match else text).strip()+'\n'


def problem_map(manifest,problems):
    mapping=[]
    for row in manifest['rows']:
        d=row['data'];domain=row['domain'];bank=problems[domain]
        if domain=='humaneval_plus':
            name=re.search(r'def\s+(\w+)\(',d['prefix']).group(1)
        else:
            # Identity comes from frozen official source tests, not generated code.
            calls={node.func.id for node in ast.walk(ast.parse(d['test_list'][0]))
                   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
            entry_names={x['entry_point'] for x in bank.values()}
            names=calls & entry_names
            assert len(names)==1,(row['selected_index'],names)
            name=names.pop()
        candidates=[x for x in bank.values() if x['entry_point']==name]
        if len(candidates)>1 and domain=='humaneval_plus':
            normalize=lambda s:re.sub(r'\s+','',s)
            original=normalize(extract_code(d['prefix'],''))
            candidates=[x for x in candidates if normalize(x['prompt'])==original]
        if len(candidates)>1 and domain=='mbpp_plus':
            normalize=lambda s:re.sub(r'\s+','',s)
            original=normalize(d['test_list'][0])
            candidates=[x for x in candidates if original in normalize(x['assertion'])]
        assert len(candidates)==1,(domain,name,[x['task_id'] for x in candidates])
        mapping.append(dict(domain=domain,selected_index=row['selected_index'],
            split=row['split'],entry_point=name,task_id=candidates[0]['task_id']))
    return mapping


def sandbox_worker(path):
    # Both official reference and candidate code stay inside the private
    # mount/PID/network namespace. No generated code runs in this host process.
    resource.setrlimit(resource.RLIMIT_CPU,(70,70))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    resource.setrlimit(resource.RLIMIT_NPROC,(512,512))
    from evalplus.gen.util import trusted_exec
    from evalplus.eval import untrusted_check
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    with open(path,'rb') as inp:payload=pickle.load(inp)
    problem=payload['problem'];code=payload['code'];dataset=payload['dataset']
    result={};canonical=problem['prompt']+problem['canonical_solution']
    output_not_none=dataset=='mbpp' and problem['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS
    for kind in ['base','plus']:
        inputs=problem[kind+'_input']
        expected,ref_times=trusted_exec(canonical,inputs,problem['entry_point'],
            record_time=True,output_not_none=output_not_none)
        status,details=untrusted_check(dataset,code,inputs,problem['entry_point'],
            expected,problem['atol'],ref_times,fast_check=True)
        result[kind]=dict(status=status,tests=len(inputs),executed=len(details))
    print(json.dumps(result))


def sandbox_check(problem,code,dataset):
    ast.parse(code)
    with tempfile.TemporaryDirectory(prefix='multibd-quality-') as tmp:
        target=Path(tmp)/'payload.pkl'
        target.write_bytes(pickle.dumps(dict(problem=problem,code=code,dataset=dataset)))
        command=[str(BWRAP),'--unshare-all','--die-with-parent','--new-session',
            '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib',
            '--ro-bind','/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/dev/shm',
            '--tmpfs','/tmp','--ro-bind',str(ENV),str(ENV),
            '--ro-bind',str(DEPS),'/deps','--ro-bind',str(PROJECT),'/poc',
            '--ro-bind',tmp,'/work','--chdir','/tmp','--clearenv',
            '--setenv','HOME','/tmp','--setenv','PYTHONPATH','/deps:/poc',
            '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
            '--setenv','EVALPLUS_MAX_MEMORY_BYTES',str(1024**3),
            '--',str(ENV/'bin/python'),'-m','poc_multibd_moe_ep.scripts.evaluate_quality',
            '--sandbox-worker','/work/payload.pkl']
        result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        if result.returncode:raise RuntimeError(result.stderr[-1500:])
        return json.loads(result.stdout.splitlines()[-1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--sandbox-worker');p.add_argument('--session');a=p.parse_args()
    if a.sandbox_worker:
        sandbox_worker(a.sandbox_worker);return
    assert a.session,'A hardware/session stratum must be explicitly selected'
    results=ROOT/'results'/a.session
    from evalplus.data import get_human_eval_plus,get_mbpp_plus
    problems={'humaneval_plus':get_human_eval_plus(),'mbpp_plus':get_mbpp_plus()}
    manifest=json.loads((ROOT/'artifacts/frozen_manifest.json').read_text())
    mapping=problem_map(manifest,problems)
    write(results/'quality_problem_identity.json',dict(rows=mapping,
        evaluator='evalplus==0.3.1',humaneval_version='v0.1.10',mbpp_version='v0.2.0',
        mapping_rule='unique official entry-point identity from frozen source, never output selection'))
    lookup={(r['domain'],r['selected_index']):r for r in mapping}
    previous=results/'quality_evalplus.json'
    rows=json.loads(previous.read_text()).get('rows',[]) if previous.exists() else []
    completed={r['artifact'] for r in rows}
    unique=set()
    for path in sorted((ROOT/'artifacts').glob('R*/request*.json')):
        result=json.loads(path.read_text())
        if result.get('measurement_session')!=a.session:continue
        if result['mode']!='clean' or result['is_warmup'] or result['bounded_smoke']:continue
        source=result['request'];key=(source['split'],source['domain'],source['selected_index'],result['R'],result['backend'])
        if key in unique:continue
        unique.add(key)
        artifact=str(path.relative_to(ROOT))
        if artifact in completed:continue
        identity=lookup[(source['domain'],source['selected_index'])]
        trajectory=result['trajectories'][0]
        row=dict(artifact=artifact,split=source['split'],domain=source['domain'],
            selected_index=source['selected_index'],R=result['R'],backend=result['backend'],
            measurement_session=a.session,
            task_id=identity['task_id'],completion_reason=trajectory['completion_reason'])
        # Upstream is_truncated ALSO includes normal EOS, because unused slots
        # are clipped. Only actual generation/NFE/context caps exclude accuracy.
        if any(trajectory[k] for k in ['max_new_tokens_reached','max_nfe_reached',
                                      'max_model_len_reached','max_repetition_run_reached']):
            row.update(evaluable=False,status='TRUNCATED_NOT_ACCURACY_EVIDENCE')
        else:
            code=extract_code(source['data']['prefix'],trajectory['text'])
            problem=problems[source['domain']][identity['task_id']]
            start=time.monotonic()
            try:
                scores=sandbox_check(problem,code,'humaneval' if source['domain']=='humaneval_plus' else 'mbpp')
                row.update(evaluable=True,status='SCORED',scores=scores,
                    passed=scores['base']['status']=='pass' and scores['plus']['status']=='pass')
            except SyntaxError as exc:
                row.update(evaluable=True,status='MALFORMED_CODE',passed=False,error=str(exc))
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                row.update(evaluable=False,status='SANDBOX_EVALUATION_UNRESOLVED',error=str(exc))
            row['sandbox_wall_s']=time.monotonic()-start
        rows.append(row)
        write(previous,dict(rows=rows,sandbox='bubblewrap private network/PID/mount + RLIMITs',
            tests='official EvalPlus full base AND plus inputs/oracles, not only source example tests'))
        print('QUALITY',row['split'],row['R'],row['task_id'],row['status'],row.get('passed'),flush=True)


if __name__=='__main__':main()
