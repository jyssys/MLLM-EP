"""Close the GPU phase only after controllers and measured gates are checked."""
import json
import subprocess
import time
from pathlib import Path
from .common import ROOT,REPO,PYTHON,preserve,write,digest
from .monitor import alive,snapshot


def main():
    preserve();state=snapshot()
    assert not any(v['alive'] for v in state['controllers'].values()),state['controllers']
    for name in ('monitor_process.json','freeze_service_model_process.json','calibration_analysis_process.json','freeze_control_gap_model_process.json','freeze_control_replay_inputs_process.json'):
        path=ROOT/'state'/name
        if path.exists():
            record=json.loads(path.read_text())
            assert not alive(record['pid']),(name,record)
    assert state['GPU_lease']['status']=='RELEASED',state['GPU_lease']
    frozen=json.loads((ROOT/'state/frozen_predecessor_hashes.json').read_text())
    assert all(digest(REPO/p)==sha for p,sha in frozen.items())
    installed=json.loads((ROOT/'state/installed_runtime_sources.json').read_text())['hashes']
    assert all(digest(Path(p))==sha for p,sha in installed.items())
    from .snapshot_integrity import main as inherited_sources
    inherited_sources()
    frozen_model=json.loads((ROOT/'state/service_model_v2_frozen.json').read_text())
    assert digest(ROOT/'configs/frozen_action_service_costs_v2.json')==frozen_model['SHA256']
    if (ROOT/'state/service_model_v3_frozen.json').exists():
        v3=json.loads((ROOT/'state/service_model_v3_frozen.json').read_text())
        assert digest(ROOT/'configs/frozen_action_service_costs_v3.json')==v3['SHA256']
    if (ROOT/'state/service_model_v4_frozen.json').exists():
        v4=json.loads((ROOT/'state/service_model_v4_frozen.json').read_text())
        assert digest(ROOT/'configs/frozen_action_service_costs_v4.json')==v4['SHA256']
        tests=json.loads((ROOT/'state/batch4_knob_confirmation_design.json').read_text())['names']
        for name in tests:
            begin=ROOT/'raw'/name/'measurement_begin.json'
            if begin.exists():assert json.loads(begin.read_text())['time']>v4['monotonic']
    if (ROOT/'state/control_feedback_replay_inputs_frozen.json').exists():
        control=json.loads((ROOT/'state/control_feedback_replay_inputs_frozen.json').read_text())
        assert digest(ROOT/'configs/frozen_control_feedback_replay_inputs_v1.json')==control['SHA256']
        tests=json.loads((ROOT/'state/batch4_knob_confirmation_design.json').read_text())['names']
        for name in tests:
            begin=ROOT/'raw'/name/'measurement_begin.json'
            if begin.exists():assert json.loads(begin.read_text())['time']>control['monotonic']
    fixed=json.loads((ROOT/'configs/frozen_best_fixed_axes_v1.json').read_text())
    for row in fixed['selection']['discovery']:
        assert digest(ROOT/'analysis'/(row['name']+'_metrics.json'))==row['metric_SHA256']
    tested=('configuration','historical_audit','replay','axis_oracle','cohort_audit','observable_features','measurement_export','resampling','cohort_axis_oracle','control_replay')
    command=[str(PYTHON),'-m','unittest',*['poc_flashvep.dense_ar_axis_sweetspot.test_'+p for p in tested]]
    result=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
    write(ROOT/'analysis/final_CPU_test_receipt.json',dict(time=time.time(),command=command,
        returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,result.stderr
    from .statistics import main as stats
    stats()
    inventory=json.loads((ROOT/'analysis/experiment_inventory.json').read_text())
    gates=[]
    for row in inventory['runs']:
        if row['status'] not in ('MEASURED_ONLINE','CORRECTNESS_ONLY'):
            continue
        suffix='_state_gate.json' if row['status']=='CORRECTNESS_ONLY' else '_metrics.json'
        target=ROOT/'analysis'/(row['name']+suffix)
        value=json.loads(target.read_text())
        if row['status']=='CORRECTNESS_ONLY':
            assert value['passed']
        else:
            assert value['output_gate']['passed'] and value['unified_worker_gate']
            assert value['common_reservation_gate']['passed']
            assert value['actual_query_coverage_gate']['passed']
            if row['method']!='Native':assert value['exact_GEMM_row_gate']
        gates.append(dict(name=row['name'],status=row['status'],gate_SHA256=digest(target)))
    write(ROOT/'analysis/final_preservation_and_gate_receipt.json',dict(time=time.time(),
        passed=True,predecessor_files=len(frozen),installed_runtime_files=len(installed),
        measured_and_state_gates=gates,user_checkout_and_index_preserved=True,
        scope='Fresh completed state/performance gates. Observer diagnostics and the preserved premeasurement v1 failure remain outside these eligible results.'))
    target=ROOT/'state/CAMPAIGN_CLOSED.json'
    assert not target.exists(),'Preserve the first GPU closeout boundary'
    write(target,dict(time=time.time(),status='GPU_PHASE_CLOSED',inventory_counts=inventory['counts'],
        scope='All GPU controllers returned. This is an execution boundary, not a scientific GO or completed publication. Unexecuted cells remain NOT_RUN. Run final analysis, report review, burner closeout and private-index publication next.'))
    preserve();print('GPU_PHASE_CLOSED_AND_PRESERVED',flush=True)


if __name__=='__main__':
    main()
