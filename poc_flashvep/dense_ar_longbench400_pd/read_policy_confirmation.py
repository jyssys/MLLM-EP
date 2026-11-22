"""Confirm the fixed core READ intervention before best-static test selection."""
import json
import time
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .metrics import summarize
from .confirmation_analysis import ratio


def main():
    capsule,sha=freeze('READ_policy_decode_protection_confirmation_v1_20261121')
    data=json.loads((ROOT/'data/balanced400_discovery.json').read_text())
    suffix,work=matched_suffixes([r['input_tokens'] for r in data],4)
    cells=[]
    for start,seed in enumerate((2026112491,2026112492,2026112493)):
        jobs={}
        for label,axis in (('NativeStrong','Native'),('NativeREAD','Native'),('Layer4','Layer'),
                            ('StockToken4','Token'),('Hybrid4','Hybrid')):
            extra={}
            if axis=='Token':
                extra.update(token_stock_model=True,suffix_by_request={'q'+str(i):s for i,s in enumerate(suffix)},
                    aggregate_work_audit=work,qualification_gate='C3_stock_token_gate_v1.json')
            elif axis!='Native':
                extra['qualification_gate']='C3_continuous_actions_gate_v1.json'
            if axis=='Hybrid':
                extra.update(hybrid_suffix_fraction=.125,partition_aligned_chunks=True)
            compiled=axis in ('Native','Token')
            jobs[label]=create(f'READNI_Discovery200_q2.8_s{start}_{label}_v1',axis,
                'balanced400_discovery.json',cut=0 if axis=='Native' else 4,
                pair=(6,7),qps=2.8,seed=seed,d_compute_throttle=label!='NativeStrong',
                defer_D_remote_setup=label!='NativeStrong',
                execution_family='INSTALLED_DEFAULT_COMPILE' if compiled else 'MATCHED_NO_COMPILE',
                continuous_source_credit_limit=0 if compiled else 4,
                source_capsule=capsule,source_capsule_SHA256=sha,
                scope='Fixed core cut4 and common READ policy selected before these new arrival seeds. '
                    'Three independent starts per arm on the same GPUs6/7. No running cap. '
                    'Discovery200 is not the held-out200 final test. Do not use these TTFT outcomes for static selection. '
                    'Hybrid4/f1/8 is genuine2D but not iso-work with Layer4/Token4. '
                    'Keep unchanged NativeStrong and common-policy NativeREAD. No timestamp smoothing.',**extra)
        cells.append(dict(start=start,seed=seed,jobs=jobs))
    write(ROOT/'configs/READ_policy_decode_protection_confirmation_v1.json',dict(
        cells=cells,source_capsule_SHA256=sha,QPS=2.8,GPU_pair=[6,7],
        evidence='FROZEN_INDEPENDENT_SAFETY_CONFIRMATION_DESIGN',
        limits=dict(native_relative_ITL_p99_CI_upper=1.05,native_relative_TPOT_p95_CI_upper=1.05),
        scope='Causal recovery confirmation on the existing discovery requests with new seeds. '
            'This does not replace best-static held-out latency confirmation or the primary GPUs0/1 test. '
            'No cut,workload,SLO or timestamp changes after observing the result.'))
    values={a:[] for a in cells[0]['jobs']}
    for cell in cells:
        # Reverse order in alternate starts without changing the trace contract.
        order=list(cell['jobs'])
        if cell['start']%2:
            order.reverse()
        for label in order:
            name=cell['jobs'][label]
            assert execute(name),name
            m=summarize(name)
            score=quality(name)
            assert score['passed'],(name,score)
            if label in ('Layer4','StockToken4','Hybrid4'):
                proof=audit(name)
                assert proof['passed'] and proof['decode_full36_observation_complete']
            values[label].append(m)
            print('READ_INDEPENDENT_CONFIRMATION',cell['start'],label,m['TTFT_p95_ms'],
                m['ITL_p99_ms'],m['TPOT_p95_ms'],flush=True)
    comparisons={}
    for label in ('NativeREAD','Layer4','StockToken4','Hybrid4'):
        comparisons[label]={}
        for native in ('NativeStrong','NativeREAD'):
            if label==native:
                continue
            tests={f:ratio([r[f] for r in values[label]],[r[f] for r in values[native]])
                for f in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')}
            comparisons[label][native]=dict(ratios=tests,decode_5percent_pass=all(tests[f]['CI95'][1]<=1.05
                for f in ('ITL_p99_ms','TPOT_p95_ms')),
                absolute_decode_pass=all(r['decode_SLO_attainment']>=.95 for r in values[label]))
    passed=all(v['decode_5percent_pass'] and v['absolute_decode_pass']
        for tests in comparisons.values() for v in tests.values())
    write(ROOT/'analysis/READ_policy_decode_protection_confirmation_v1.json',dict(time=time.time(),
        passed=passed,cells=cells,comparisons=comparisons,
        means={a:{f:sum(r[f] for r in records)/3 for f in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms',
            'SLO_qualified_output_tokens_per_second')} for a,records in values.items()},
        evidence='MEASURED_ONLINE',
        scope='Three independent paired server starts on GPUs6/7. New seeds,old discovery requests. '
            'Pointwise empirical paired bootstrap CI with27 resamples. No request-as-start resampling. '
            'A recovered core decode gate is not best-static serving superiority,an executable active-decode window,or dynamic gain.'))
    print('READ_DECODE_PROTECTION_CONFIRMATION',passed,flush=True)


if __name__=='__main__':
    main()
