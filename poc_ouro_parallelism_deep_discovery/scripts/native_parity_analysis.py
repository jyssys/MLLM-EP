"""Quantify matched native TP captures. No free-running parity declaration."""
import argparse
import datetime
import numpy as np
from poc_ouro_parallelism_deep_discovery.contract import *

def stats(x,y):
    delta=x.astype(np.float64)-y.astype(np.float64)
    return dict(shape=list(x.shape),max_abs=float(np.max(np.abs(delta))),
        relative_L2=float(np.linalg.norm(delta)/max(float(np.linalg.norm(y)),1e-12)),
        bitwise_float_capture_equal=bool(np.array_equal(x,y)))

def log_softmax(x):
    x=x.astype(np.float64);x=x-x.max(-1,keepdims=True)
    return x-np.log(np.exp(x).sum(-1,keepdims=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not args.output.exists()
    by_model={};records=[]
    for run in sorted((ROOT/'outputs').glob('native_*')):
        status=run/'guard_status.json'
        if not status.exists() or not read(status).get('success'):continue
        for capture in run.glob('*_TP*/parity_capture.json'):
            owner=read(capture.parent/'status.json');key=(owner['size'],owner['R'])
            evidence=read(capture)['evidence'];rank0=next(r for r in evidence if r['rank']==0)
            by_model.setdefault(key,{})[owner['TP']]=(run,rank0)
    for (model,R),arms in sorted(by_model.items()):
        if 1 not in arms:continue
        reference=np.load(arms[1][1]['path']);ref_names=arms[1][1]['KV_names']
        for P,(run,evidence) in sorted(arms.items()):
            current=np.load(evidence['path'])
            assert current['logits'].shape==reference['logits'].shape
            assert evidence['positions']==arms[1][1]['positions']
            # Rank0 selects the same first two global heads at every TP degree.
            order=[evidence['KV_names'].index(n) for n in ref_names]
            KV=current['KV'][order]
            lp,lq=log_softmax(reference['logits']),log_softmax(current['logits'])
            records.append(dict(model=model,R=R,P=P,run=run.name,
                reference=arms[1][0].name,logits=stats(current['logits'],reference['logits']),
                hidden=stats(current['hidden'],reference['hidden']),KV=stats(KV,reference['KV']),
                KL_reference_to_current_per_position=(np.exp(lp)*(lp-lq)).sum(-1).tolist(),
                top1_match_per_position=(current['logits'].argmax(-1)==reference['logits'].argmax(-1)).tolist(),
                scope='Four controlled teacher-forced positions. KV samples cover selected features of the first two rank0 global heads.',
                quality='NOT_RUN for this capture. Not free-running natural token, benchmark, or full-KV parity.',
                exactness='Quantified only. No post-hoc tolerance converts these differences to an equivalence claim.'))
    reference_records=[]
    for run in sorted((ROOT/'outputs').glob('reference_*')):
        state=run/'status.json'
        if not state.exists() or not read(state).get('success'):continue
        meta=read(state);key=(meta['size'],meta['R'])
        if key not in by_model:continue
        reference=np.load(meta['path'])
        for P,(native_run,evidence) in sorted(by_model[key].items()):
            current=np.load(evidence['path'])
            order=[evidence['KV_names'].index(n) for n in meta['KV_names']]
            assert meta['positions']==evidence['positions']
            lp,lq=log_softmax(reference['logits']),log_softmax(current['logits'])
            reference_records.append(dict(model=meta['size'],R=meta['R'],P=P,
                native_run=native_run.name,reference_run=run.name,
                logits=stats(current['logits'],reference['logits']),
                hidden=stats(current['hidden'],reference['hidden']),
                KV=stats(current['KV'][order],reference['KV']),
                KL_reference_to_current_per_position=(np.exp(lp)*(lp-lq)).sum(-1).tolist(),
                top1_match_per_position=(current['logits'].argmax(-1)==reference['logits'].argmax(-1)).tolist(),
                scope=meta['scope'],
                exactness='Quantified only. Independent staged reference uses the pinned pretrained checkpoint. No full-model equivalence claim.'))
    write(args.output,dict(records=records,independent_reference_records=reference_records,
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        evidence_class='QUALITY_ONLY',independent_reference='FOUR_POSITION_STAGED_CAPTURE' if reference_records else 'NOT_RUN',
        candidate='NONE. These are native comparator captures.'))
    print('NATIVE_CAPTURE_COMPARISONS',len(records),'REFERENCE_COMPARISONS',len(reference_records))

if __name__=='__main__':main()
