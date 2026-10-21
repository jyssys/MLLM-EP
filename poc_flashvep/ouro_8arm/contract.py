import hashlib
import json
from pathlib import Path
from poc_flashvep.ouro_tp4.contract import PROJECT,MODEL,REVISION,PYTHON,VLLM_PYTHON,UUIDS

ROOT=PROJECT/'poc_flashvep/reports/ouro_8arm_static_dynamic_discovery'
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_8arm_static_dynamic_fidelity_serving_spec.md')
CDB=Path('/home/esjung/external/decode-first-cdb-upstream')
CDB_REV='abdfae0adbdd12be8c6123ccb34eee98a38afb39'
ARMS={a:dict(parallelism=p,mode=m,GPUs=[4] if p=='single' else [4,5,6,7])
      for a,p,m in [('S4','single','static'),('T4','tp4','static'),('L4','layer_pp4','static'),('O4','loop_pp4','static'),
                   ('SD','single','dynamic'),('TD','tp4','dynamic'),('LD','layer_pp4','dynamic'),('OD','loop_pp4','dynamic')]}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,default=str)+'\n')

def dataset(pilot=True):
    values=read(ROOT/'dataset_manifest.json')['rows']
    if (ROOT/'frozen_prompt_ids.json').exists():
        ids=read(ROOT/'frozen_prompt_ids.json')['rows']
        values=[dict(r,input_ids=ids[r['id']]) for r in values]
    return [r for r in values if r['pilot']] if pilot else values


def subset400(extension_only=False):
    manifest=read(ROOT/'evaluation_subset400.json');byid={r['id']:r for r in dataset(False)}
    selected=[]
    for cohort in manifest['cohorts']:
        if extension_only and cohort['source']!='extension':continue
        selected.extend(dict(byid[rid],cohort400=cohort['index']) for rid in cohort['ids'])
    return selected
