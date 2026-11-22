"""Tune every already-qualified static family under the same safe admission rule."""
import argparse
import json
from .common import ROOT,write
from .experiment import create
from .geometry import matched_suffixes
from .next_jobs import plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('lane',type=int,choices=(0,1));args=parser.parse_args()
    qps,first=((1.8,0),(2.8,4))[args.lane]
    # Preserve the source used by the original three-axis admission A/B.
    baseline=json.loads((ROOT/f'raw/GuardAB_Discovery200_q{qps}_Layer4Guard_v1/configuration.json').read_text())
    data=json.loads((ROOT/'data/balanced400_discovery.json').read_text());jobs=[];rows=[]
    candidates=[(axis,m,None) for axis in ('Layer','Token') for m in (2,8,12)]
    candidates += [('Hybrid',2,.125),('Hybrid',8,.0625),('Hybrid',12,.0625)]
    for axis,cut,fraction in candidates:
        extra={}
        if axis=='Token':
            suffix,work=matched_suffixes([r['input_tokens'] for r in data],cut)
            extra.update(suffix_by_request={'q'+str(i):v for i,v in enumerate(suffix)},aggregate_work_audit=work)
        if axis=='Hybrid':extra.update(hybrid_suffix_fraction=fraction,partition_aligned_chunks=True)
        label=axis+str(cut)+(('_f'+str(fraction)) if fraction is not None else '')
        name=create(f'GuardTune_Discovery200_q{qps}_{label}_v1',axis,'balanced400_discovery.json',
            cut=cut,pair=(first,first+1),qps=qps,seed=2026112291,
            d_quiescent_admission=True,continuous_source_credit_limit=4,
            qualification_gate='C3_continuous_actions_gate_v1.json',
            source_capsule=baseline['source_capsule'],source_capsule_SHA256=baseline['source_capsule_SHA256'],
            discovery_only=True,static_tuning=True,
            scope='Same frozen source,arrival and output contract as the cut4 admission discovery. '
                'All candidates passed their action-specific C0 and C3 gates. D-quiescent admission has a separate full400 proof. '
                'The selected final action still needs its own full400 stress and held-out confirmation. '
                'This is best-versus-best discovery,not exact iso-work or a continuous-D batching claim.',**extra)
        jobs.append(name);rows.append(dict(name=name,axis=axis,cut=cut,suffix_fraction=fraction))
    manifest=f'guard_static_tuning_lane{args.lane}_v1';plan(manifest,jobs)
    write(ROOT/'configs'/(manifest+'.json'),dict(rows=rows,QPS=qps,
        additional_cut4_results=[f'GuardAB_Discovery200_q{qps}_{a}4Guard_v1' for a in ('Layer','Token','Hybrid')],
        selector='Within each family,retain every outcome. Minimize TTFTp95 among correctness/quality-qualified candidates '
            'that pass both the frozen absolute decode SLO and Native-relative decode point checks. '
            'Break exact ties by SLO-goodput. Freeze choices before new held-out measurements.',
        evidence='FROZEN_STATIC_TUNING_DESIGN',
        limitations='The original cut4 results are already observed. This is discovery,not blind confirmation. '
            'Pair0/1 and4/5 are hardware-equivalent preflight pairs. Final Primary confirmation must keep each paired start on the same GPU pair.'))
    print('Frozen9 additional static candidates at QPS',qps,flush=True)


if __name__=='__main__':main()
