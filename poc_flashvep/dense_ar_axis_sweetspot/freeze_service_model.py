"""Freeze action costs from calibration/discovery, without reading test metrics."""
import collections
import fcntl
import json
import statistics
import time
from .common import ROOT,QUIET_LOCK,DEADLINE,write,digest,preserve
from .model_validation import extract


def names():
    initial=json.loads((ROOT/'state/initial_design_v2_before_performance.json').read_text())
    selected=[n for n in initial['calibration'] if '_Native_' not in n]
    design=ROOT/'state/mechanism_discovery_design.json'
    if design.exists():
        selected.extend(n for n in json.loads(design.read_text())['names'] if '_Native_' not in n)
    for m in (4,8,12):
        for a in ('Layer','Token'):
            selected.append(f'source_window_control_{a}_Last{m}_N72_O16_q6_s2026111901_B2_W32_v1')
    return selected


def main():
    while not (ROOT/'state/mechanism_discovery_complete.json').exists():
        assert not (ROOT/'state/research_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(10)
    selected=names()
    while not all((ROOT/'analysis'/(n+'_service_queue_audit.json')).exists() for n in selected):
        assert not (ROOT/'state/counterfactual_extensions_stopped.json').exists()
        assert time.time()<DEADLINE-3600
        time.sleep(10)
    preserve();grouped=collections.defaultdict(list);provenance=[]
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        for name in selected:
            cfg,rows,cost=extract(ROOT,name)
            assert cfg['matrix_stage'] in ('calibration','mechanism_discovery','source_window_control')
            assert cfg['cap']==16 and cfg['source_outstanding_limit']==32
            grouped[cfg['evaluation_arm'],cfg['static_layers'],cfg['D_admission_batch']].append(cost)
            provenance.append(dict(name=name,request_log_SHA256=digest(ROOT/'raw'/name/'requests.jsonl'),
                configuration_SHA256=digest(ROOT/'configs'/(name+'.json')),cost_seconds=cost))
        result=[]
        for (a,m,b),values in sorted(grouped.items()):
            cost={k:statistics.median(v[k] for v in values) for k in values[0]}
            result.append(dict(method=a,layers=m,batch=b,output=16,cost_seconds=cost,
                training_starts=len(values),status='CALIBRATED_NEW_DISCOVERY'))
        for row in list(result):
            if row['batch']==2 and row['layers']==8:
                result.append(dict(row,batch=4,status='BATCH4_EXTRAPOLATION_NOT_FITTED_TO_TEST',
                    assumption='Keep prep1 P/suffix service and whole-cohort Full36 Decode cost. Divide shared Decode occupancy by the new actual batch in scoring. This approximation needs validation.'))
        base=list(result)
        for o in (32,64,128,200):
            for row in base:
                cost=dict(row['cost_seconds']);cost['decode']*= (o-1)/15
                result.append(dict(row,output=o,cost_seconds=cost,
                    status='OUTPUT_LENGTH_EXTRAPOLATION_NOT_FITTED_TO_TEST',
                    assumption='Input and partition stay fixed. Scale shared Full36 Decode residency by controlled token intervals. Context/clock/host effects are not assumed absent and must pass held-out validation.'))
        document=dict(models=result,training=provenance,
            scope='Fit only new calibration, q2/q6 mechanism discovery and source-window32 controls. Do not read heldout_confirmation, Decode2/4 counterfactual, output-residency counterfactual or newQPS3 metrics. Costs are host service enclosures, not exclusive kernels. Keep long host pauses in measured TTFT and validation error.',
            protocol_SHA256=digest(ROOT/'state/minimal_knob_protocol_v2_before_new_batch_output.json'),
            frozen_SLO_SHA256=digest(ROOT/'configs/common_output_sweep_SLO.json'),
            evidence='EMPIRICAL_COST_CALIBRATION')
        target=ROOT/'configs/frozen_action_service_costs_v2.json'
        if target.exists():
            assert json.loads(target.read_text())['model']==document
        else:
            write(target,dict(frozen_epoch=time.time(),model=document))
        write(ROOT/'state/service_model_v2_frozen.json',dict(time=time.time(),path=str(target),SHA256=digest(target)))
    preserve();print('SERVICE_MODEL_V2_FROZEN',len(selected),len(result),flush=True)


if __name__=='__main__':
    main()
