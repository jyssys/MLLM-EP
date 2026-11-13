"""Separate central prediction error from empirical upper-estimate coverage."""
import statistics
from .common import *
from .analyze import records,percentile,compare
from .forecast_analysis import analyze,errors


def online(name):
    result=analyze(name)
    if result is None:return None
    central=[];upper=[];by_shape={}
    for row in result['accepted_predictions']:
        if 'predicted_D_finish_mean' not in row:continue
        e=1000*(row['observed_first_token']-row['predicted_D_finish_mean'])
        u=1000*(row['observed_first_token']-row['predicted_D_finish'])
        central.append(e);upper.append(u)
        key=f"B{len(row['requests'])}_Last{row['layers']}"
        by_shape.setdefault(key,[]).append((e,u))
    result['central_completion']=errors(central)
    result['upper_estimate']=dict(samples=len(upper),coverage=sum(x<=0 for x in upper)/len(upper) if upper else None,
        exceedance_p95_ms=percentile([max(0.,x) for x in upper],.95),maximum_exceedance_ms=max([0.,*upper]),
        scope='Coverage of observed committed completions. Unexecuted rejected candidates have no measured counterfactual. Not a deadline guarantee.')
    result['central_and_upper_by_shape']={key:dict(central=errors([x[0] for x in values]),
        upper_coverage=sum(x[1]<=0 for x in values)/len(values),samples=len(values)) for key,values in by_shape.items()}
    return result


def main():
    rows=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):continue
        if not (folder/'online_decisions.jsonl').exists():continue
        result=online(folder.name)
        if result:rows.append(result)
    write(ROOT/'analysis/revised_prediction_validation.json',dict(results=rows,
        evidence='TRACE_ANALYSIS',scope='Analyze realized outcomes after execution. The online policy receives no future output or arrival data.'))
    from ..dense_ar_dynamic_layer_pd import calibrate_costs as previous_parser
    previous_parser.ROOT=ROOT
    primitive=[]
    for name in ('fresh_cost_B1_Last024812_N10_O2_v2','fresh_cost_B4_Last024812_N40_O2_v1'):
        p=ROOT/'raw'/name/'dispatcher_result.json'
        if p.exists() and json.loads(p.read_text()).get('complete'):
            gate=compare('fresh_calibration_reference_Native_O16_N40_v1',name,True)
            samples=previous_parser.parse(name)
            primitive.append(dict(name=name,output=gate,rows=samples,
                scope='Fresh stream-event and Host consistency samples. Two repetitions per shape are not a p99 or WCET bound.'))
    write(ROOT/'analysis/fresh_primitive_cost_validation.json',dict(results=primitive,evidence='MEASURED',
        scope='Reuse the frozen predecessor parser. Original code and raw evidence remain unchanged. Event enclosures are not exclusive kernel time.'))
    print('PREDICTION_RUNS',len(rows),'FRESH_PRIMITIVE_RUNS',len(primitive))


if __name__=='__main__':main()
