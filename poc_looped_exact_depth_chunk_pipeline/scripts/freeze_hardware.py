import gzip
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    target=ROOT/'results/frozen_hardware.json';assert not target.exists()
    source=PROJECT/'poc_looped_multigpu_kv_placement/results/frozen_tokens.json.gz'
    teachers=PROJECT/'poc_looped_recurrence_tp_4models/results/frozen_tokens.json.gz'
    with gzip.open(source,'rt') as stream:rows=[r for r in json.load(stream) if r['model']=='ouro']
    with gzip.open(teachers,'rt') as stream:t={r['request_id']:r['teacher_ids'] for r in json.load(stream) if r['model']=='ouro'}
    for row in rows:
        row['teacher_ids']=t[row['request_id']]
        row['prefix_token_ids']={key:value for key,value in row['prefix_token_ids'].items() if key in ['1024','8192','16384','32704']}
    write(target,dict(requests=rows,source_sha256=sha(source),teacher_sha256=sha(teachers),
        scope='Eight frozen continuity requests. Not fresh held-out. Metadata-selected before current measurements.',
        matrix=dict(TP4_R4=['FULL',256,512,1024,2048,4096],TP1_R1=[256,512,1024,2048,4096],TP2_R1=[256,512,1024,2048,4096]),
        conditional_exit='Exact graph is generic PP. This matrix is capped generic economics, not a loop-specific method trial.'))
    print('Frozen eight hardware requests and the capped cost matrix.')


if __name__=='__main__':main()
