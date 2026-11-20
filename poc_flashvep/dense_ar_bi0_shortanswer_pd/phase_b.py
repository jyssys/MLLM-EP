"""Qualify official BI0 paths, then make controlled NIXL interventions."""
import json
import time
from .common import ROOT,write,save
from .quality import reference_summary
from .qualification import configuration
from .run import execute


def main():
    while not (ROOT/'state/phase_A_reference_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists()
        time.sleep(3)
    reference=reference_summary()
    assert reference['qasper_mean_below100'], 'Preserve outputs and evaluate preregistered alternatives before changing the workload'
    contract=ROOT/'configs/frozen_BI0_correctness_quality_contract_v1.json'
    if not contract.exists():
        write(contract,dict(time=time.time(),same_geometry='Strict output and imported KV equality',
            across_geometry='Report strict tokens, sampled numerical errors and official task scores separately',
            tensor_atol=.02,tensor_rtol=.01,task_score_margin=2.,margin_status='Research default pending optional user response',
            point_and_CI_gates_separate=True,full200_required_for_final_task_quality=True,
            performance_start_bootstrap='Paired independent server starts, never request bootstrap'))
    for label,method,extra in (
        ('Standard','Native',dict(reference_single=True,gpus=[1])),
        ('Native','Native',{}),
        ('Agg','Agg',dict(serial=False,arrival_offsets=[float(i) for i in range(8)]))):
        name=f'qual_{label}_ShortAnswer_N8_BI0_state_v1'
        cfg=configuration(name,method,'qualification8.json',**extra)
        assert execute(save(cfg)),name
    write(ROOT/'state/BI0_N8_state_jobs_complete.json',dict(time=time.time(),complete=True,
        scope='Execution receipts only. Analyze full state and quality before qualification.'))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/phase_B_hold.json',dict(time=time.time(),error=repr(error)))
        raise
