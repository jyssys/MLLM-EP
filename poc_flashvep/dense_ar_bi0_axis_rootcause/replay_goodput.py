"""Do not infer goodput accuracy from a TTFT quantile check."""
import json
import time
from .common import ROOT,write
from .causal_replay import extract,predict


def main():
    declaration=ROOT/'configs/frozen_replay_goodput_validation_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),training_seed=2026112082,test_seed=2026112085,
            relative_error_limit=.1,cut=4,QPS=2.35,N=200,
            SLO=dict(TTFT_ms=5000,every_ITL_ms=34,TPOT_ms=33),
            scope='Additional goodput validation before any oracle headroom claim. The earlier p95/MAE gate is unchanged. '
            'Use calibration decode-pass flags, not target latency, to form predictions. This does not validate a decode-latency predictor.'))
    rows=[]
    for axis in ('Layer','Token'):
        calibration_name=f'CausalLast4_{axis}_ONLINE_q2.35_c4_s2026112082_N200_v1'
        target_name=f'HeldoutReplay_{axis}4_q2.35_c4_s2026112085_N200_v1'
        if not (ROOT/'analysis'/(target_name+'_metrics.json')).exists():continue
        cost=extract(calibration_name);target=extract(target_name);result=predict(cost,target)
        source_metrics=json.loads((ROOT/'analysis'/(calibration_name+'_metrics.json')).read_text())
        target_metrics=json.loads((ROOT/'analysis'/(target_name+'_metrics.json')).read_text())
        flags={r['prompt_id']:r['decode_pass'] for r in source_metrics['per_request']}
        tokens=sum(row['output_tokens'] for row,state in zip(cost['rows'],result['rows'])
            if state['TTFT_ms']<=5000 and flags[row['prompt_id']])
        duration=result['rows'][-1]['finish']
        prediction=tokens/duration;actual=target_metrics['SLO_qualified_output_tokens_per_second']
        relative_error=prediction/actual-1 if actual>0 else None
        rows.append(dict(axis=axis,calibration=calibration_name,target=target_name,
            predicted_goodput_tokens_per_s=prediction,actual_goodput_tokens_per_s=actual,
            relative_error=relative_error,goodput_gate_passed=relative_error is not None and abs(relative_error)<=.1,
            scope='Calibration service and decode-pass fixtures are reused on a new arrival seed. No test service vector enters the prediction.'))
    write(ROOT/'analysis/heldout_replay_goodput_validation_v1.json',dict(results=rows,evidence='OFFLINE_REPLAY',
        scope='A p95/MAE pass does not imply goodput or safe-window qualification. This is not an online axis scheduler or a sustainable-capacity estimate.'))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
