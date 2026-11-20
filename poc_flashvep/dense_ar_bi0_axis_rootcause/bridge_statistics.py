"""Keep historical-family repeats separate from Natural LongBench main results."""
import itertools
import json
import numpy as np
from .common import ROOT,write
from .statistics import interval,identity


def results(name):
    path=ROOT/'analysis'/name
    return json.loads(path.read_text()).get('results',[]) if path.exists() else []


FIELDS=('TTFT_p95_ms','SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second','ITL_p99_ms','TPOT_p95_ms')


def compare(rows,seeds):
    cells={axis:{r['seed']:r for r in rows if r.get('axis')==axis and r.get('status')=='MEASURED_ONLINE'} for axis in ('Native','Layer','Token')}
    if not all(set(seeds)<=set(cells[a]) for a in cells):return None
    contract=[identity(cells['Layer'][s]['name'],cells['Token'][s]['name']) for s in seeds]
    ratios={f:interval([cells['Layer'][s]['metrics'][f] for s in seeds],[cells['Token'][s]['metrics'][f] for s in seeds]) for f in FIELDS}
    native={axis:{f:interval([cells[axis][s]['metrics'][f] for s in seeds],[cells['Native'][s]['metrics'][f] for s in seeds])
        for f in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')} for axis in ('Layer','Token')}
    return dict(seeds=list(seeds),Layer_over_Token=ratios,Native_relative=native,contract=contract,
        source_names={a:[cells[a][s]['name'] for s in seeds] for a in cells},
        Native_relative_decode_pass={a:all(native[a][f]['CI95'][1]<=1.05 for f in ('ITL_p99_ms','TPOT_p95_ms')) for a in native},
        scope='Fixed-output historical72,actual1/1/1. Not the Natural LongBench200 main dataset. Near-matched rows:Token suffix1776 gives -0.1% destination row difference.')


def factorial(rows):
    seeds=[2026111920,2026111921,2026111922]
    cells={(a,c):{r['seed']:r for r in rows if r.get('axis')==a and r.get('coalesced')==c and r.get('status')=='DIAGNOSTIC'}
        for a in ('Layer','Token') for c in (False,True)}
    if not all(set(seeds)<=set(v) for v in cells.values()):return None
    result={}
    for field in FIELDS:
        parts={c:interval([cells['Layer',c][s]['metrics'][field] for s in seeds],
                         [cells['Token',c][s]['metrics'][field] for s in seeds]) for c in (False,True)}
        def score(indices):
            means={(a,c):np.mean([cells[a,c][seeds[i]]['metrics'][field] for i in indices]) for a,c in cells}
            den=means['Token',True]*means['Layer',False]
            return float(means['Layer',True]*means['Token',False]/den) if den>0 else None
        draws=[score(i) for i in itertools.product(range(3),repeat=3)]
        result[field]=dict(no_coalescing=parts[False],coalescing=parts[True],
            relative_axis_ratio_change=score(range(3)),
            CI95=[float(np.percentile(draws,p)) for p in (2.5,97.5)] if all(v is not None for v in draws) else None)
    realization=[]
    for axis in ('Layer','Token'):
        for seed in seeds:
            record=cells[axis,True][seed]
            raw=json.loads((ROOT/'raw'/record['name']/'NIXL_host_setup_decode.json').read_text())['rows']
            requests=[r for r in raw if r['kind']=='make_transfer_host' and isinstance(r.get('request_id'),str) and r['request_id'].startswith('q')]
            realization.append(dict(name=record['name'],calls=len(requests),
                actual_descriptor_reduction_calls=sum(r['submitted_descriptors']<r['input_descriptors'] for r in requests),
                candidate_byte_records=sum(r.get('bytes') is not None for r in requests)))
    realized=any(r['actual_descriptor_reduction_calls'] for r in realization)
    return dict(seeds=seeds,metrics=result,evidence='DIAGNOSTIC',starts_per_factor_axis=3,new_heldout_seeds=2,
        physical_coalescing_intervention='PARTIAL_REALIZATION' if realized else 'NOT_REALIZED',realization=realization,
        Native_relative_decode_safety='NOT_MEASURED',
        scope='The numerical flag stays BI1. Only the coalescing host flag changes inside each seed/axis. '
        'Three paired starts include the original1-start diagnostic. Two additional seeds test repeatability. '
        'No physical descriptor reduction was observed in these runs. Do not attribute the timing difference to DMA coalescing. '
        'This is not three fully held-out serving confirmations or a BI0 layer-specific GO.')


def main():
    original=results('historical_runtime_bridge_v1.json')
    B128=[r for r in original if not r.get('BI')]
    block=results('historical_block16_confirmation_v1.json')+results('historical_block16_extra_holdout_v1.json')
    block=list({r['name']:r for r in block if r.get('name')}.values())
    standard=compare(B128,[2026111920,2026111921,2026111922])
    heldout=compare(block,[2026111921,2026111922,2026111923])
    transport=factorial(results('coalescing_paired_repetition_v1.json'))
    output=dict(current_BI0_block128_all3=standard,current_BI0_block16_heldout3=heldout,
        current_BI1_coalescing_factorial=transport,
        scope='Never pool BI1 transport diagnostics with BI0 or Natural200 main results. Missing independent repeats remain pending or NOT_RUN.')
    write(ROOT/'analysis/historical_bridge_independent_start_statistics.json',output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
