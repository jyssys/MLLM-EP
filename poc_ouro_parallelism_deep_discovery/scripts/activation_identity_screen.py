"""Inspect actual captured inputs for an exact whole-row reuse prerequisite."""
import argparse
import datetime as dt
import itertools
import re
from poc_ouro_parallelism_deep_discovery.runtime import configure_imports
from poc_ouro_parallelism_deep_discovery.contract import *
configure_imports()
import torch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    inputs={'1.4B':ROOT/'outputs/capture_base14_v2/1.4B/actual_mlp_inputs.pt',
            '2.6B':ROOT/'outputs/capture_base26_v1/2.6B/actual_mlp_inputs.pt'}
    records=[];provenance=[]
    for size,path in inputs.items():
        tensors=torch.load(path,map_location='cpu',weights_only=True)
        groups={}
        for key,value in tensors.items():
            match=re.fullmatch(r'(fixed|adaptive)_(\d+)_layer(\d+)_call(\d+)',key)
            assert match,key
            mode,request,layer,call=match.groups();call=int(call)
            if call<=4:phase='prefill';position_group=0;depth=call
            elif mode=='fixed':phase='decode';position_group=(call-5)//4;depth=(call-5)%4+1
            else:continue
            groups.setdefault((mode,int(request),int(layer),phase,position_group),{})[depth]=(key,value)
        for (mode,request,layer,phase,position_group),depths in groups.items():
            for a,b in itertools.combinations(sorted(depths),2):
                ka,x=depths[a];kb,y=depths[b]
                assert x.shape==y.shape and x.dtype==y.dtype==torch.bfloat16
                equal=x.contiguous().view(torch.int16)==y.contiguous().view(torch.int16)
                complete=equal.all(dim=-1)
                records.append(dict(size=size,mode=mode,request_index=request,layer=layer,
                    phase=phase,position_group=position_group,depth_a=a,depth_b=b,
                    source_a=ka,source_b=kb,rows=x.shape[0],features=x.shape[-1],
                    identical_rows=int(complete.sum()),row_identity_fraction=float(complete.float().mean()),
                    identical_scalars=int(equal.sum()),scalar_identity_fraction=float(equal.float().mean()),
                    limitation='Corresponding captured row comparison only. This does not certify equality of attention KV or the entire transition read set.'))
        provenance.append(dict(size=size,capture=str(path.relative_to(ROOT)),capture_sha256=sha(path),
            sampling='Two continuity smoke prompts. Three selected physical layers. Last32 prefill rows and bounded first calls.'))
    summaries=[]
    for size in inputs:
        for phase in ['prefill','decode']:
            selected=[r for r in records if r['size']==size and r['phase']==phase]
            denominator=sum(r['rows'] for r in selected)
            summaries.append(dict(size=size,phase=phase,compared_row_pairs=denominator,
                identical_row_pairs=sum(r['identical_rows'] for r in selected),
                scalar_identity_fraction=sum(r['identical_scalars'] for r in selected)/(denominator*2048),
                sample_unit='Overlapping descriptive row pairs from two bounded prompts. Not independent benchmark requests.'))
    write(args.output,dict(timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        evidence_class='ANALYTIC_BOUND',empirical_input='Actual pretrained BF16 activations from earlier GPU captures. This analysis uses CPU operations only.',
        provenance=provenance,records=records,summaries=summaries,
        candidate='R3-B exact transition coalescing prerequisite screen',
        inference='Whole-row identity is necessary for this unchanged-input MLP reuse design, not for all possible exact optimizations.',
        limitations=['No full historical-KV equality test','No population-level rate or CI','No GPU lookup/hash/codec timing','No new task accuracy or E2E result']))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
