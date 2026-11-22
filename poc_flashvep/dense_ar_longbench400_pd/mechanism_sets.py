"""Select unmodified request subsets with exact aggregate row feasibility."""
import hashlib
import json
from .common import ROOT,write,digest


def exact_subset(rows,n=16,modulus=288):
    ordered=sorted(rows,key=lambda row:hashlib.sha256(('mechanism-v1-'+row['prompt_id']).encode()).hexdigest())
    states={(0,0):()}
    for i,row in enumerate(ordered):
        for (count,residue),indices in sorted(list(states.items()),reverse=True):
            if count>=n:continue
            key=count+1,(residue+row['input_tokens'])%modulus
            if key not in states:states[key]=indices+(i,)
        if (n,0) in states:
            return [ordered[i] for i in states[n,0]],True
    return ordered[:n],False


def main():
    manifest=[]
    for split in ('discovery','heldout'):
        for bucket in range(4):
            source=ROOT/'data'/f'mechanism_input{bucket}_{split}.json'
            rows=json.loads(source.read_text());assert len(rows)>=16
            chosen,exact=exact_subset(rows)
            target=ROOT/'data'/f'input{bucket}_{split}_exact16.json'
            assert not target.exists(),'Preserve the frozen mechanism set'
            write(target,chosen)
            manifest.append(dict(split=split,bucket=bucket,n=len(chosen),source_count=len(rows),
                input_min=min(r['input_tokens'] for r in chosen),input_max=max(r['input_tokens'] for r in chosen),
                sum_input_tokens=sum(r['input_tokens'] for r in chosen),sum_mod288=sum(r['input_tokens'] for r in chosen)%288,
                exact_aggregate_16_token_alignment_feasible=exact,dataset=target.name,SHA256=digest(target),
                prompt_ids=[r['prompt_id'] for r in chosen]))
    result=dict(rows=manifest,evidence='STRUCTURAL_ACCOUNTING',
        scope='Requests remain unmodified. Selection uses input lengths and deterministic identity ordering only. '
            'A sum divisible by288 permits exact aggregate token-layer rows at all four primary cuts. '
            'No serving result selects these requests.')
    write(ROOT/'data/mechanism_exact16_sets_frozen_v1.json',result);print(result)


if __name__=='__main__':main()
