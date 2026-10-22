"""Prospective exploration admission, not an adaptive-method GO decision."""
import json
import numpy as np
from .common import ROOT,write,digest,append

def main():
    replay=ROOT/'raw/p1_MLP_same_input_geometry_v1/mlp_geometry_replay.json'
    result=json.loads(replay.read_text())
    assert result['original_geometries_reproduce_exactly']
    work=json.loads((ROOT/'surface_work_clock_verification.json').read_text())
    assert len(work)==48 and all(r['passed'] for r in work)
    qualifies=[json.loads(p.read_text())['summary'] for p in (ROOT/'raw').glob('p1_isolated*/completed.json')]
    assert len(qualifies)==8 and all(r['isolated_exact_pass'] for r in qualifies)
    jobs=json.loads((ROOT/'p1_quality_plan.json').read_text())['jobs']
    surface=json.loads((ROOT/'p2_surface_plan.json').read_text())['jobs']
    # The original source/condition matrix is immutable. This supplemental
    # dispatch plan changes affinity only, never inputs or policy parameters.
    jobs=jobs+[dict(j,allow_any_gpu=True) for j in surface]
    path=ROOT/'quality_and_surface_dispatch_v1.json'
    assert not path.exists()
    write(path,dict(gpus=list(range(8)),jobs=jobs,
        scope='Exploratory diagnostic costs and frozen quality pilot. No quality-preserving adaptive claim.',
        affinity_rule='Prefer registered GPU. An idle worker may claim an unstarted complete-policy block. All policies remain within-GPU paired.',
        constraints='Do not tune models or select serving policies from HELDOUT data. No live adaptive implementation before Phase4 review.'))
    # Numeric SLO registration precedes held-out cost and online candidate runs.
    cal=[]
    for p in (ROOT/'raw').glob('p0_*parallel1*gpu4/cal_P2048_N4_r*.json'):
        cal.append(json.loads(p.read_text())['median_ITL_ms'])
    assert len(cal)==6,cal
    baseline=float(np.median(cal))
    write(ROOT/'slo_registration.json',dict(reference='GPU4 warmed solo P2033/N4, median of six cohort medians from two process starts',
        reference_decode_ms=baseline,TBT_tiers_ms={name:baseline*factor for name,factor in [('Strict',5),('Moderate',10),('Relaxed',25)]},
        absolute_TBT_ms=[75,100,125,150,200,250,500],TTFT_s=[2,5,10],primary_TTFT_s=5,primary_TBT_tier='Moderate',
        registered_before='Main grid held-out data and any adaptive online outcome',
        qualification='Per-request every-gap joint attainment requires natural completion. Population p99 is separate.'))
    append(ROOT/'phase_exit_decisions.jsonl',dict(phase='1_to_2',status='CONDITIONAL',
        basis=['160 isolated numerical comparisons exact','48 preflight policy cohorts pass full-R4 accounting',
            'One historical case reproduced exactly by same-input BF16 MLP row geometry without KV/streams/graphs'],
        root_evidence_sha256=digest(replay),limits='Other mismatches and full-answer quality remain unqualified. Cost data are diagnostic. No adaptive advancement granted.'))
    print('ADMITTED',len(jobs),'quality/cost jobs; reference ms',baseline)

if __name__=='__main__':
    main()
