"""Separate controlled C3 bridge evidence from C0 and natural400 serving."""
import collections
import json
import time
import numpy as np
from .common import ROOT,REPO,write,digest
from .metrics import summarize
from .confirmation_analysis import ratio
from .c0_queue_model import part
from .service_accounting import compare


FIELDS=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second',
    'generated_output_tokens_per_second','Joint_SLO_attainment')


def load(name,source,expected_pair=(4,5)):
    folder=ROOT/'raw'/name
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists():return None,None,'NOT_RUN'
    if not json.loads(receipt.read_text())['complete']:return None,None,'FAILED'
    assert json.loads(receipt.read_text())['source_SHA256']==source
    cfg=json.loads((folder/'configuration.json').read_text())
    m=summarize(name)
    assert m['output_tokens']==cfg['count']*cfg['output_cap']
    assert cfg['ignore_eos'] and not cfg['controlled_pipeline'] and cfg['gpus']==list(expected_pair)
    cap=cfg.get('D_running_admission_cap')
    if cap is not None:
        actual=json.loads((folder/'actual_runtime_decode.json').read_text())
        assert actual['runtime_max_num_running_requests']==cap
        assert m['actual_geometry']['decode']['maximum_requests']<=cap
    data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
    partitions=[part(cfg,i,r['input_tokens']) for i,r in enumerate(data)]
    logical=dict(P_rows=sum(p.P_rows for p in partitions),D_rows=sum(p.D_rows for p in partitions))
    padding={}
    for role in ('prefill','decode'):
        values=json.loads((folder/f'actual_queries_{role}.json').read_text())['rows']
        batches=collections.defaultdict(list)
        for row in values:batches[row['batch_id']].append(row)
        examples=[]
        for batch_id,rows in batches.items():
            if not any(r['API_prefilling'] for r in rows):continue
            extra=rows[0]['padded_queries']-rows[0]['actual_batch_queries']
            assert extra>=0
            if extra:examples.append(dict(batch_id=batch_id,extra_padded_queries=extra))
        padding[role]=dict(batches_with_prefill_extra_padding=len(examples),
            total_extra_padded_queries=sum(r['extra_padded_queries'] for r in examples),examples=examples,
            scope='Observed extra model-input query slots. Do not infer their physical GEMM rows or exclusive GPU time for a partial-layer adapter.')
    return dict(name=name,metrics=m,logical_work=logical,padding=padding),cfg,'MEASURED_ONLINE'


def main(plan_name='controlled_continuous_I3_bridge_frozen_v1.json',suffix=''):
    plan=json.loads((ROOT/'configs'/plan_name).read_text())
    groups=collections.defaultdict(list)
    for block in plan['cells']:groups[block['D_cap'],block['output']].append(block)
    output=[]
    loaded={}
    for (cap,tokens),blocks in sorted(groups.items()):
        blocks=sorted(blocks,key=lambda c:c['start'])
        arms=list(blocks[0]['jobs']);values={a:[] for a in arms};cfgs={a:[] for a in arms};starts=[]
        for block in blocks:
            statuses={}
            for arm,name in block['jobs'].items():
                m,cfg,status=load(name,plan['source_capsule_SHA256'],block['GPU_pair']);statuses[arm]=status
                if m is not None:
                    values[arm].append(m);cfgs[arm].append(cfg);loaded[name]=(m,cfg)
            starts.append(dict(seed=block['seed'],GPU_pair=block['GPU_pair'],status=statuses))
        row=dict(D_cap=cap,controlled_output=tokens,QPS=1.253873,required_starts=3,starts=starts,
            status='MEASURED_ONLINE_CONFIRMED' if all(len(v)==3 for v in values.values()) else 'PARTIAL_OR_NOT_RUN',
            arms={},comparisons={})
        for arm,data in values.items():
            row['arms'][arm]=dict(starts=len(data),means={f:float(np.mean([r['metrics'][f] for r in data])) for f in FIELDS} if data else {},
                geometry=[r['metrics']['actual_geometry'] for r in data],logical_work=[r['logical_work'] for r in data],
                padding=[r['padding'] for r in data],names=[r['name'] for r in data],
                compile=[c['execution_family'] for c in cfgs[arm]],
                source_credit=[c.get('continuous_source_credit_limit',0) for c in cfgs[arm]])
        for arm in ('Layer','StockToken','TokenMode0Control','Hybrid'):
            if len(values[arm])!=3:continue
            refs=['NativeStrong','NativeCommon']
            if arm=='Layer':refs+=['StockToken','TokenMode0Control']
            if arm=='Hybrid':refs+=['Layer','StockToken','TokenMode0Control']
            if arm=='TokenMode0Control':refs+=['StockToken']
            tests={}
            for ref in refs:
                if len(values[ref])!=3:continue
                for a,b in zip(cfgs[arm],cfgs[ref]):
                    for field in ('dataset_SHA256','arrival_offsets','seed','gpus','output_cap','ignore_eos'):
                        assert a[field]==b[field],(arm,ref,field)
                ratios={f:ratio([r['metrics'][f] for r in values[arm]],[r['metrics'][f] for r in values[ref]]) for f in FIELDS}
                tests[ref]=dict(ratios=ratios,decode_5percent_CI_pass=all(ratios[f]['CI95'][1]<=1.05 for f in ('ITL_p99_ms','TPOT_p95_ms')),
                    absolute_decode_pass=all(r['metrics']['decode_SLO_attainment']>=.95 for r in values[arm]),
                    TTFT_CI_win=ratios['TTFT_p95_ms']['CI95'][1]<1,
                    goodput_CI_win=ratios['SLO_qualified_output_tokens_per_second'] is not None and ratios['SLO_qualified_output_tokens_per_second']['CI95'][0]>1)
            row['comparisons'][arm]=tests
        if all(len(values[a])==3 for a in ('Layer','StockToken','TokenMode0Control','Hybrid')):
            rows={a:values[a][0]['logical_work'] for a in ('Layer','StockToken','TokenMode0Control','Hybrid')}
            assert len({tuple(r.items()) for r in rows.values()})==1,rows
            row['axis_logical_iso_work_verified']=True
        output.append(row)
    interventions=[]
    for arm in ('NativeCommon','Layer','StockToken','TokenMode0Control','Hybrid'):
        a=next(r for r in plan['cells'] if r['D_cap']==1 and r['output']==64 and r['start']==0)
        b=next(r for r in plan['cells'] if r['D_cap']==2 and r['output']==64 and r['start']==0)
        pairs=[]
        for index in range(3):
            c1=next(r for r in plan['cells'] if r['D_cap']==1 and r['output']==64 and r['start']==index)
            c2=next(r for r in plan['cells'] if r['D_cap']==2 and r['output']==64 and r['start']==index)
            n1,n2=c1['jobs'][arm],c2['jobs'][arm]
            if n1 not in loaded or n2 not in loaded:continue
            m1,g1=loaded[n1];m2,g2=loaded[n2]
            for field in ('source_capsule_SHA256','dataset_SHA256','arrival_offsets','seed','gpus','output_cap','ignore_eos',
                    'continuous_source_credit_limit','execution_family','axis_kind','axis_cut','suffix_by_request'):
                assert g1.get(field)==g2.get(field),(arm,field)
            pairs.append((m1,m2,n1,n2))
        if len(pairs)==3:
            interventions.append(dict(arm=arm,cap2_over_cap1={f:ratio([r[1]['metrics'][f] for r in pairs],
                [r[0]['metrics'][f] for r in pairs]) for f in FIELDS},
                host_boundary_cap2_minus_cap1=[compare(r[3],r[2]) for r in pairs]))
    value=dict(time=time.time(),cells=output,cap_interventions=interventions,
        complete=all(r['status']=='MEASURED_ONLINE_CONFIRMED' for r in output),
        source_capsule_SHA256=plan['source_capsule_SHA256'],evidence='MEASURED_ONLINE_CONTROLLED_OUTPUT',
        scope='Three fixed bridge conditions. All six arms and all declared starts remain visible. '
            'Pointwise27-resample paired empirical95percent CIs. Logical aggregate rows match among four offloading arms. '
            'Actual continuous batch and model-input padding are recorded,not forced equal. '
            'Source credit8 matches across axes. Native controls keep their stronger source0 policy. '
            'Mode0 Token is not the strong Token replacement. Its runtime-family contrast includes adapter and numerical paths. '
            'Cap intervention fixes source,pair,requests,output and arrival within each arm. It is not an exclusive kernel attribution. '
            'The C0 reference used a different pair,source and seeds. No paired C0→C3 CI. No natural400 or sustainable-capacity claim.')
    write(ROOT/'analysis'/('controlled_continuous_I3_bridge'+suffix+'_results_v1.json'),value)
    print([(r['D_cap'],r['controlled_output'],r['status']) for r in output],flush=True)


if __name__=='__main__':
    main()
