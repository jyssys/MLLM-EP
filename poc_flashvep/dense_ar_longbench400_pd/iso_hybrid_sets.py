"""Keep an exact iso-work Hybrid4/Layer8/Token8 frontier on original requests."""
import json
from .common import ROOT,write,digest
from .mechanism_sets import exact_subset
from .geometry import matched_suffixes


def main():
    rows=[]
    for split in ('discovery','heldout'):
        for bucket in range(4):
            source=ROOT/f'data/mechanism_input{bucket}_{split}.json'
            chosen,feasible=exact_subset(json.loads(source.read_text()),modulus=1152)
            assert feasible,'Do not call an approximate frontier exact'
            total=sum(r['input_tokens'] for r in chosen)
            token,token_audit=matched_suffixes([r['input_tokens'] for r in chosen],8)
            assert token_audit['exact_match']
            hybrid=[int(r['input_tokens']//128)*16 for r in chosen]
            required=total//8-sum(hybrid);assert required%16==0
            order=sorted(range(len(chosen)),key=lambda i:(-(chosen[i]['input_tokens']/8-hybrid[i]),i))
            for i in order[:required//16]:hybrid[i]+=16
            assert sum(hybrid)*32+total*4==total*8
            path=ROOT/f'data/iso8_input{bucket}_{split}_exact16.json'
            if path.exists():assert json.loads(path.read_text())==chosen
            else:write(path,chosen)
            rows.append(dict(split=split,input_bucket=bucket,n=16,dataset=path.name,SHA256=digest(path),
                input_sum=total,Layer_cut=8,Token_equivalent_cut=8,Hybrid_cut=4,
                Hybrid_suffix_by_request={'q'+str(i):s for i,s in enumerate(hybrid)},
                Token_suffix_by_request={'q'+str(i):s for i,s in enumerate(token)},
                D_rows_each=total*8,P_rows_each=total*28,exact_match=True))
    value=dict(rows=rows,evidence='STRUCTURAL_ACCOUNTING',
        scope='Unmodified source requests. Selection uses deterministic IDs and input length residues only. '
            'The common input sum is divisible by1152. All three arms have exactly equal aggregate GEMM rows. '
            'Their Attention pairs,hidden payloads and scheduling paths remain different. No performance result selects this frontier.')
    path=ROOT/'data/iso8_hybrid_sets_frozen_v1.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:write(path,value)
    print([(r['input_bucket'],r['split'],r['input_sum']) for r in rows],flush=True)


if __name__=='__main__':main()
