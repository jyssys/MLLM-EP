"""Test the source/destination service model on an independent arrival seed."""
import json
import time
from .common import ROOT,write,digest
from .campaign import cell
from .causal_replay import extract,predict,error


def main():
    reference=ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json'
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    gold={r['prompt_id']:r for r in json.loads(reference.read_text())['rows']}
    lengths=[gold[r['prompt_id']]['output_tokens'] for r in data]
    declaration=ROOT/'configs/frozen_heldout_service_replay_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),calibration_seed=2026112082,test_seed=2026112085,QPS=2.35,
            cut=4,N=200,source_credit=4,actual_batches=[1,1,1],output='Fixed C1 lengths,4723tokens',
            reference_SHA256=digest(reference),metrics=['p95_relative_error<=.1','MAE<=max100ms,.1*meanTTFT'],
            scope='Arrival-seed holdout. Prompts are reused. Not unseen-request generalization. No future test service costs enter prediction.'))
    rows=[]
    for axis in ('Token','Layer'):
        name=f'HeldoutReplay_{axis}4_q2.35_c4_s2026112085_N200_v1'
        row=cell(name,axis,4,seed=2026112085,entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
            diagnostic_controlled_output_replay=True,replay_output_lengths=lengths,phase='HELDOUT_ARRIVAL_SERVICE_MODEL_VALIDATION')
        assert row['status']!='FAILED'
        calibration=extract(f'CausalLast4_{axis}_ONLINE_q2.35_c4_s2026112082_N200_v1')
        target=extract(name);prediction=predict(calibration,target)
        rows.append(dict(axis=axis,name=name,predicted_TTFT_p95_ms=prediction['TTFT_p95_ms'],
            measured_TTFT_p95_ms=target['measured_p95_TTFT_ms'],**error(prediction,target)))
        write(ROOT/'analysis/heldout_arrival_service_replay_v1.json',dict(results=rows,evidence='OFFLINE_REPLAY',
            scope='Predictions use another arrival-seed calibration. Targets are measured fixed-output diagnostics. Not a dynamic online scheduler.'))
    write(ROOT/'state/heldout_arrival_service_replay_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()
