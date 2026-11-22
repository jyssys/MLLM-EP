"""Extend a reproduced graph-eligibility failure to pure-token and hybrid tails."""
import copy
import hashlib
import json
from .common import ROOT,write
from .experiment import create
from .next_jobs import plan


def dataset(length):
    pool=json.loads((ROOT/'data/longbench_balanced400_v1.json').read_text())
    source=next(r for r in pool if r['input_tokens']>=length);row=copy.deepcopy(source)
    row.update(prompt_id=source['prompt_id']+f'_SHAPE_PREFIX{length}',input_tokens=length,
        input_ids={'qwen8':source['input_ids']['qwen8'][:length]},diagnostic_only=True,
        context_truncated=True,diagnostic_scope='Truncated token-prefix shape test. Never Main400 or task-quality evidence.')
    row['input_ids_SHA256']=hashlib.sha256(json.dumps(row['input_ids']['qwen8'],separators=(',',':')).encode()).hexdigest()
    path=ROOT/f'data/diagnostic_query{length}.json'
    if path.exists():assert json.loads(path.read_text())==[row]
    else:write(path,[row])
    return path.name


def main():
    jobs=[]
    jobs.append(create('Edge8193_Full36_Teacher_v2','Native','diagnostic_query8193.json',
        pair=(6,7),output=16,fixed=True,correctness=True,reference_single=True,serial=True,
        axis_state_capture=False,state_capture=True,numerical_probe=True,numerical_probe_sync=True,kv_layout='HND'))
    for axis,length in [('Token',9217),('Hybrid',9361)]:
        path=dataset(length);extra=dict(hybrid_suffix_fraction=.125,partition_aligned_chunks=True) if axis=='Hybrid' else {}
        jobs.append(create(f'Edge{length}_{axis}4_Guarded_State_v1',axis,path,cut=4,pair=(6,7),
            output=16,fixed=True,correctness=True,controlled=True,serial=True,**extra))
        teacher=dict(teacher_suffix_fraction=.125) if axis=='Hybrid' else {}
        jobs.append(create(f'Edge{length}_{axis}Boundary_Teacher_v1','Native',path,pair=(6,7),
            output=16,fixed=True,correctness=True,reference_single=True,serial=True,
            axis_state_capture=False,state_capture=True,numerical_probe=True,numerical_probe_sync=True,
            token_teacher_cut=4,kv_layout='HND',**teacher))
    plan('query1_graph_edge_followup_v1',jobs)
    print('Frozen edge teachers and additional query1 boundary tests.',flush=True)


if __name__=='__main__':main()
