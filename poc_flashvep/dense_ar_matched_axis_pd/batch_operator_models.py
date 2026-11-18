"""Apply isolated projection medians to observed input shapes, not GPU time."""
import json
from .common import ROOT, write


def main():
    starts=json.loads((ROOT/'analysis/independent_start_statistics.json').read_text())['starts']
    lookup={}
    provenance=[]
    for name in ('isolated_BF16_axis_geometry_v1','isolated_realized_axis_geometry_v1',
                 'isolated_realized_all_pair_geometry_v1'):
        path=ROOT/'raw'/name/'operator_geometry.json'
        if not path.exists():continue
        data=json.loads(path.read_text());provenance.append(name)
        for row in data['results']:
            if row['kind']=='GEMM':
                lookup[(row['GPU'],row['projection'],row['M'])]=(row['p50_ms'],name)
    projections=sorted({key[1] for key in lookup})
    assert len(projections)==4,projections
    output=[]
    for start in starts:
        if start['method']=='Agg':continue
        name=start['name'];folder=ROOT/'raw'/name
        cfg=json.loads((folder/'configuration.json').read_text())
        components={}
        for gpu,role in enumerate(('prefill','decode')):
            counts=json.loads((folder/f'measured_geometry_{role}.json').read_text())['counts']
            values={};missing=[]
            for key,repeats in counts.items():
                shape=json.loads(key)
                prompt=role=='prefill' or shape['prefilling'] and start['method']!='Native'
                phase='Prompt' if prompt else 'Decode'
                layers=(36-start['layers'] if start['method']=='Layer' else 36) if role=='prefill' else (
                    start['layers'] if prompt and start['method']=='Layer' else 36)
                rows=shape['padded_query_tokens']
                found=[lookup.get((gpu,projection,rows)) for projection in projections]
                item=values.setdefault(phase,dict(model_input_count=0,measured_operator_covered_inputs=0,
                    modeled_sum_ms=0.,actual_padded_token_layer_rows=0,isolated_sources=[]))
                item['model_input_count']+=repeats
                item['actual_padded_token_layer_rows']+=rows*layers*repeats
                if any(value is None for value in found):
                    missing.append(dict(M=rows,phase=phase,count=repeats))
                    continue
                item['measured_operator_covered_inputs']+=repeats
                item['modeled_sum_ms']+=sum(value[0] for value in found)*layers*repeats
                item['isolated_sources']=sorted(set(item['isolated_sources'])|{value[1] for value in found})
            for item in values.values():
                item['complete_shape_coverage']=item['model_input_count']==item['measured_operator_covered_inputs']
                item['modeled_sum_per_completed_request_ms']=item['modeled_sum_ms']/cfg['count'] if item['complete_shape_coverage'] else None
            components[role]=dict(phases=values,missing_shapes=missing)
        output.append(dict(name=name,stage=start['stage'],method=start['method'],layers=start['layers'],
            output_cap=cfg['cap'],QPS=cfg['arrival_rate'],seed=cfg['arrival_seed'],count=cfg['count'],
            admission=cfg['D_admission_batch'],components=components))
    write(ROOT/'analysis/observed_batch_projection_models.json',dict(starts=output,isolated_starts=provenance,
        evidence='STRUCTURAL_ACCOUNTING',measurement_input='ISOLATED_CUDA_EVENT_PROJECTION_MEDIANS',
        scope='Use the four original layer0 projection medians at each observed padded M. Multiply by observed model-input counts and executed layer counts. These are modeled sums, not measured online kernel time. The sums exclude norm, activation, attention, transport, scheduling and contention. Missing operator shapes remain unmeasured. Native cached new-token queries are Decode work even when the input flag says prefilling. No serving speedup or tail confidence interval follows from this model.'))
    print('OBSERVED_BATCH_PROJECTION_MODELS',len(output),flush=True)


if __name__=='__main__':
    main()
