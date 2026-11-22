"""Use paired server starts for pointwise CIs and separate safety from speed."""
import itertools
import json
import time
import numpy as np
from .common import ROOT,write


def ratio(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);n=len(a)
    assert len(b)==n and n>=3 and np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    if np.any(b<=0):return None
    indices=np.asarray(list(itertools.product(range(n),repeat=n)))
    values=np.mean(a[indices],axis=1)/np.mean(b[indices],axis=1)
    return dict(estimate=float(a.mean()/b.mean()),CI95=np.percentile(values,[2.5,97.5]).tolist(),
        starts=n,resamples=len(indices),method='Exact enumeration of paired start-level empirical bootstrap samples',
        qualifier='Pointwise,unadjusted empirical CI. Small start counts limit coverage. No request-as-start resampling.')


def load(name,source_sha):
    folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
    if not receipt.exists():return None,None
    if not json.loads(receipt.read_text())['complete']:raise AssertionError(('Failed start is retained',name))
    path=ROOT/'analysis'/(name+'_metrics.json')
    if not path.exists():return None,None
    metric=json.loads(path.read_text());cfg=json.loads((folder/'configuration.json').read_text())
    assert metric['runtime_source_archive_SHA256']==source_sha
    assert metric['evidence']=='MEASURED_ONLINE' and metric['execution_scope']=='OPEN_LOOP_ONLINE'
    assert metric['n']==16 and metric['output_tokens']==16*cfg['output_cap']
    assert cfg['token_budget']==8192 and cfg['max_num_seqs']==1024 and cfg['source_credit_limit']==4
    assert cfg['controlled_pipeline'] and cfg['ignore_eos']
    assert all(g['maximum_requests']==1 for g in metric['actual_geometry'].values())
    if cfg.get('axis_kind'):
        proof=ROOT/'analysis'/(name+'_coverage.json')
        if not proof.exists():return None,None
        verified=json.loads(proof.read_text())
        assert verified['passed'] and verified.get('decode_full36_observation_complete',False)
        if cfg['axis_kind']=='Token':assert cfg['aggregate_work_audit']['exact_match']
    return metric,cfg


def main():
    protocol=json.loads((ROOT/'configs/heldout64_confirmation_protocol_r2.json').read_text())
    core_path=ROOT/'configs/core_five_start_confirmation_frozen_v1.json'
    core=json.loads(core_path.read_text()) if core_path.exists() else None
    core_keys={tuple(k) for k in core['chosen_cells']} if core else set()
    rows=[]
    for bucket,output,cut in itertools.product(range(4),(8,16,64,128),(2,4,8,12)):
        key=bucket,output,cut
        cells=[r for r in protocol['cells'] if (r['input_bucket'],r['output'],r['cut'])==key]
        if key in core_keys:cells += [r for r in core['additional_cells'] if (r['input_bucket'],r['output'],r['cut'])==key]
        row=dict(input_bucket=bucket,output=output,cut=cut,QPS=cells[0]['QPS'],
            required_independent_starts=len(cells),completed_paired_starts=0,status='NOT_RUN',results=[],
            source_capsule_SHA256=protocol['source_capsule_SHA256'])
        values={axis:[] for axis in ('Layer','Token','Native')};configs=[]
        try:
            for cell in cells:
                triple={};configuration={}
                for axis in values:triple[axis],configuration[axis]=load(cell[axis],protocol['source_capsule_SHA256'])
                if any(v is None for v in triple.values()):continue
                base=configuration['Native']
                for cfg in configuration.values():
                    for field in ('dataset_SHA256','arrival_offsets','seed','output_cap','gpus'):
                        assert cfg[field]==base[field],('Unmatched experiment',field,cell)
                for axis in values:values[axis].append(triple[axis])
                row['results'].append(dict(seed=cell['seed'],**{axis:cell[axis] for axis in values}))
            row['completed_paired_starts']=len(values['Layer'])
            if row['completed_paired_starts']==len(cells):
                def paired(a,b,metric):return ratio([v[metric] for v in values[a]],[v[metric] for v in values[b]])
                row.update(TTFT_p95_Layer_over_Token=paired('Layer','Token','TTFT_p95_ms'),
                    goodput_Layer_over_Token=paired('Layer','Token','SLO_qualified_output_tokens_per_second'))
                row['native_relative']={}
                for axis in ('Layer','Token'):
                    tests={metric:paired(axis,'Native',metric) for metric in
                        ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')}
                    tests['decode_5percent_pass']=all(tests[m]['CI95'][1]<=1.05 for m in ('ITL_p99_ms','TPOT_p95_ms'))
                    tests['absolute_decode_pass']=all(v['decode_SLO_attainment']>=.95 for v in values[axis])
                    tests['native_TTFT_gain']=tests['TTFT_p95_ms']['CI95'][1]<1
                    g=tests['SLO_qualified_output_tokens_per_second']
                    tests['native_goodput_gain']=g is not None and g['CI95'][0]>1
                    row['native_relative'][axis]=tests
                lo,hi=row['TTFT_p95_Layer_over_Token']['CI95']
                direction='Layer' if hi<1 else 'Token' if lo>1 else None
                row['latency_direction']=direction or 'TIE_OR_INCONCLUSIVE'
                if direction:
                    tests=row['native_relative'][direction]
                    row['status']=direction.upper()+'_CONFIRMED' if tests['decode_5percent_pass'] and tests['absolute_decode_pass'] else 'INVALID'
                    if row['status']=='INVALID':row['invalid_reason']='A latency direction exists,but the winner does not pass both decode-protection gates.'
                else:row['status']='TIE_OR_INCONCLUSIVE'
        except Exception as error:
            row.update(status='INVALID',invalid_reason=repr(error))
        rows.append(row)
    result=dict(time=time.time(),cells=rows,evidence='MEASURED_ONLINE',
        counts={label:sum(r['status']==label for r in rows) for label in
            ('LAYER_CONFIRMED','TOKEN_CONFIRMED','TIE_OR_INCONCLUSIVE','INVALID','NOT_RUN')},
        scope='C0 actual1/1/1 mechanism control. Common QPS across outputs within each input bucket. '
            'Each CI resamples independent paired server-start/arrival-seed blocks. '
            'The64 pointwise CIs do not provide familywise false-positive control. '
            'Core cells have five predeclared starts. Other cells have three. '
            'An axis win is not automatically a Native serving win,continuous gain,or sustainable capacity.')
    write(ROOT/'analysis/axis_matrix_heldout_confirmation_r2.json',result)
    print(result['counts'],flush=True)
    return result


if __name__=='__main__':main()
