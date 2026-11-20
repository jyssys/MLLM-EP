"""Test the serial/online reversal at cut8 without mixing cut4 measurements."""
import json
import time
from .common import ROOT,write,digest
from .campaign import cell


def main():
    data=json.loads((ROOT/'data/longbench_shortanswer200_v2.json').read_text())
    reference=ROOT/'raw/BI0_Standard_ShortAnswer200_v2_Natural_C1_cap2048_v1/natural_reference.json'
    gold={r['prompt_id']:r for r in json.loads(reference.read_text())['rows']}
    lengths=[gold[r['prompt_id']]['output_tokens'] for r in data]
    path=ROOT/'configs/frozen_Last8_serial_online_control_v1.json'
    if not path.exists():
        write(path,dict(time=time.time(),cut=8,N=200,QPS=2.35,seed=2026112082,
            source_credit=4,hidden_source_slots=4,actual_batches=[1,1,1],KV_block=128,
            reference_SHA256=digest(reference),output='Fixed C1 request lengths, total4723tokens',
            scope='Direct matched cut8 test. Do not add the previous cut8 serial advantage to a cut4 online difference. One independent start per mode/axis.'))
    rows=[]
    for axis,mode in [('Layer','SERIAL_ADMISSION'),('Token','ONLINE'),('Token','SERIAL_ADMISSION'),('Layer','ONLINE')]:
        name=f'CausalLast8_{axis}_{mode}_q2.35_c4_s2026112082_N200_v1'
        row=cell(name,axis,8,seed=2026112082,
            entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
            strict_serial_pipeline=mode=='SERIAL_ADMISSION',diagnostic_controlled_output_replay=True,
            replay_output_lengths=lengths,phase='LAST8_MATCHED_SERIAL_ONLINE_CONTROL')
        row['causal_mode']=mode;rows.append(row)
        write(ROOT/'analysis/Last8_serial_online_causal_control_v1.json',dict(results=rows,
            evidence='MEASURED_ONLINE',diagnostic=True,scope='Same cut8, requests, arrival trace and output counts in both modes. Not Natural EOS performance.'))
        assert row['status']!='FAILED'
    write(ROOT/'state/Last8_serial_online_causal_control_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/Last8_serial_online_causal_control_hold.json',dict(time=time.time(),error=repr(error)));raise
