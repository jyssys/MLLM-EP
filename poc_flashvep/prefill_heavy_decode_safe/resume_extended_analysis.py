"""Complete the finished Native pilot analysis after a metadata lookup error."""
import json,time
from .common import *
from .host_screen import screen


def main():
    initialize();name='extended_ouro_L24576_O16_P2_mu130_pilot64_a2026111272'
    for label in ('extended_24k_native_qualification_controller','extended_24k_native_pilot_controller'):
        receipt=json.loads((ROOT/'state'/(label+'_receipt.json')).read_text());assert receipt['returncode']==0
    result=screen(name)
    write(ROOT/'state/measurement_alternatives_controller_complete.json',dict(time=time.time(),complete=True,
        resumed_analysis=name,reran_GPU_jobs=False,
        reason='The inherited metadata table has no ouro_L24576 entry. An analysis-only view supplies the already-declared campaign limits. Thresholds and raw evidence remain unchanged.',
        failed_branch='cost_ouro_L16384_loop3_full_state8',
        phase_1_qualified=result['Phase_1_qualified']))
    print(json.dumps({k:result.get(k) for k in ('classification','P_service_fraction','queue','D_credit_refusal_snapshot_fraction','NVML')}))


if __name__=='__main__':main()
