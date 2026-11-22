"""Test simple calibrated demand rules without claiming an online controller."""
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .c0_queue_model import part,features


def main():
    model_path=ROOT/'analysis/c0_host_service_model_v1.json'
    model=json.loads(model_path.read_text())['model']
    observed=json.loads((ROOT/'analysis/axis_matrix_heldout_confirmation_r2.json').read_text())
    rows=[]
    for cell in observed['cells']:
        if cell['completed_paired_starts']!=cell['required_independent_starts']:
            continue
        demands={}
        measured={}
        for axis in ('Layer','Token'):
            costs=[]
            actual=[]
            for start in cell['results']:
                name=start[axis]
                cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
                data=json.loads((ROOT/'data'/cfg['dataset']).read_text())[:cfg['count']]
                parts=[part(cfg,i,r['input_tokens']) for i,r in enumerate(data)]
                p=float(np.mean([np.dot(features(v,'P'),model['P']) for v in parts]))
                d=float(np.mean([np.dot(features(v,'D'),model['D'])+
                    (cfg['output_cap']-1)*np.dot([1,v.length/1e4],model['decode_step']) for v in parts]))
                costs.append((p,d))
                actual.append(json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())['TTFT_p95_ms'])
            p,d=np.mean(costs,axis=0)
            demands[axis]=dict(P_host_service_ms=float(p),D_first_plus_decode_host_service_ms=float(d),
                serial_sum_ms=float(p+d),bottleneck_demand_ms=float(max(p,d)),
                predicted_bottleneck_utilization=float(cfg['qps']*max(p,d)/1000))
            measured[axis]=float(np.mean(actual))
        winners={rule:min(demands,key=lambda a:demands[a][field]) for rule,field in
            (('serial_sum','serial_sum_ms'),('bottleneck','bottleneck_demand_ms'))}
        winners['load_blend_at_0_5']=min(demands,key=lambda a:demands[a][
            'bottleneck_demand_ms' if demands[a]['predicted_bottleneck_utilization']>=.5 else 'serial_sum_ms'])
        label={'LAYER_CONFIRMED':'Layer','TOKEN_CONFIRMED':'Token'}.get(cell['status'])
        rows.append(dict(input_bucket=cell['input_bucket'],controlled_output=cell['output'],cut=cell['cut'],
            measured_status=cell['status'],measured_label=label,demands=demands,
            measured_start_mean_TTFT_p95_ms=measured,predictions={rule:dict(axis=axis,
                correct=None if label is None else axis==label,
                regret_percent=100*(measured[axis]/min(measured.values())-1)) for rule,axis in winners.items()},
            scalar_bottleneck_delta_ms=demands['Layer']['bottleneck_demand_ms']-demands['Token']['bottleneck_demand_ms']))
    summary={}
    for rule in ('serial_sum','bottleneck','load_blend_at_0_5'):
        clear=[r['predictions'][rule] for r in rows if r['measured_label'] is not None]
        valid=[r['predictions'][rule]['regret_percent'] for r in rows if r['measured_status']!='INVALID']
        summary[rule]=dict(clear_winner_cells=len(clear),correct_clear_winners=sum(r['correct'] for r in clear),
            accuracy=sum(r['correct'] for r in clear)/len(clear),valid_regret_cells=len(valid),
            regret_mean_percent=float(np.mean(valid)),regret_p95_percent=float(np.percentile(valid,95)),
            regret_maximum_percent=float(max(valid)))
    value=dict(time=time.time(),model_SHA256=digest(model_path),rows=rows,summary=summary,
        evidence='OFFLINE_MODEL_EXPLORATORY_COMPARISON',
        formulas=dict(serial_sum='argmin_axis(mean_P_service + mean_D_first_service + decode_residency)',
            bottleneck='argmin_axis(max(mean_P_service,mean_D_first_service + decode_residency))',
            signed_scalar='max(P_L,D_L) - max(P_T,D_T). Negative predicts Layer.',
            load_blend_at_0_5='Use each action bottleneck demand when lambda*max(P,D)>=0.5. Else use serial sum.'),
        scope='The serial-calibrated model is not refit to winner labels. Calibration and matrix tokenized prompts/contexts are disjoint. '
            'The rule forms are assessed after earlier matrix results,so this is not a blind prospective rule confirmation. '
            'The0.5 blend threshold is not tuned. Preserve its failures. '
            'One score compresses calibrated host-service estimates and known controlled output. It is not one directly measured online state. '
            'Natural EOS duration is not known at admission. C3 continuous packing,KV state and active-decode safety are not modeled. '
            'Ties have no winner label. Invalid safety cells do not enter valid regret. Pointwise labels have no multiplicity adjustment. '
            'Bottleneck classification accuracy is not a serving speedup or a proof that serial cost causes queue delay.')
    write(ROOT/'analysis/minimal_calibrated_demand_rules_v1.json',value)
    print(summary,flush=True)


if __name__=='__main__':
    main()
