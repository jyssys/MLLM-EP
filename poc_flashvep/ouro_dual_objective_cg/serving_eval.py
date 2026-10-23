"""Natural serving on native RLT. No model kernels or inference semantics change."""
import gc
import gzip
import json
import time
from collections import defaultdict
import torch
from .common import ROOT,PRIOR,guard,write,digest
from .adaptive_selector import DualControl
from .work_ledger import WorkLedger
from ..ouro_budget_joint_cg.benchmark import engine
from ..ouro_budget_joint_cg.fairness import attach
from ..ouro_budget_joint_cg.cost_predictor import ServiceLookup

@torch.inference_mode()
def serving(model,job,target):
    from ..ouro_chunk_loop_slo import worker as frontend
    frontend.ROOT=ROOT
    payload=json.loads(gzip.open(ROOT/'workload_payloads.json.gz','rt').read())
    byid={c['id']:c for c in payload['cases']}
    lock=json.loads((ROOT/'workload_lock.json').read_text())
    assert digest(ROOT/'workload_lock.json')==job['workload_lock_sha256']
    assert digest(ROOT/'workload_payloads.json.gz')==lock['payload_sha256']
    scenario_file=ROOT/job.get('scenario_manifest','serving_scenarios.json')
    if job.get('scenario_sha256'):assert digest(scenario_file)==job['scenario_sha256']
    scenarios={r['id']:r for r in json.loads(scenario_file.read_text())['scenarios']}
    norm=json.loads((ROOT/'objective_normalization_v1.json').read_text())
    assert digest(ROOT/'objective_normalization_v1.json')==job['normalization_sha256']
    lookup=ServiceLookup(json.loads((PRIOR/'cost_model_artifacts/service_lookup_v1.json').read_text()))
    summaries=[]
    for cell in job['cells']:
        e=engine(model,cell.get('runtime','split'),cell.get('C',2048),job['width'],job)
        e.frontend_stop_token_ids=lock['frontend_stop_ids']
        fairness=attach(e,cell.get('fairness','completion')=='completion')
        control=DualControl(e,lookup,norm,cell) if cell.get('surface') else None
        # Input-only warm-up from CALIBRATION; no FINAL request trains the model.
        warm=[c for c in payload['cases'] if c['split']=='CALIBRATION' and c['domain']=='chat'][:job['width']]
        frontend.serve(e,warm,16,[0.]*len(warm),.001,observe=False)
        for scenario_id in job['scenarios']:
            scenario=scenarios[scenario_id]
            cases=[byid[c] for c in scenario['case_ids']]
            if control:control.reset()
            fairness_start=len(fairness.events)
            ledger=WorkLedger(e)
            observer=None
            if cell.get('observe',job.get('observe',False)):
                from ..ouro_budget_joint_cg.service_observer import ServiceObserver
                observer=ServiceObserver(e,dict(run_id=job['id'],label=cell['label'],physical_GPU=guard(),
                    evidence_lane='OBSERVER_AUDIT_NOT_CLEAN_SERVING'),prefill_only=True)
            torch.cuda.reset_peak_memory_stats()
            try:
                data=frontend.serve(e,cases,job.get('cap',1024),scenario['arrivals_s'],scenario['duration_s'],
                    drain=job.get('drain_s',300),observe=False)
                e.model_runner.synchronize()
                service_rows=[] if observer is None else observer.finish()
                work=defaultdict(int)
                for row in ledger.rows:
                    for span in row['spans']:work[span['id']]+=span['tokens']*span['G']
                integrity=all(request['status']=='COMPLETE' and
                    work[request['id']]==4*request['input_tokens'] and
                    all(d==4 for d in request['depths']) for request in data['requests'])
                data.update(run_id=job['id'],physical_GPU=guard(),label=cell['label'],cell=cell,
                    scenario=scenario,workload=scenario['workload'],split=scenario['split'],
                    configured_N=job['width'],budget_ms=cell.get('budget_ms'),
                    P=[len(c['input_ids']) for c in cases],families=[c['family'] for c in cases],
                    full_R4=integrity,prompt_token_loop_work=dict(work),dispatch_descriptors=ledger.rows,
                    service_observations=service_rows,heavy_observer=observer is not None,
                    optimizer_decisions=[] if control is None else list(control.selector.trace),
                    chunk_decisions=[] if control is None else list(control.actions.decisions),
                    admission_snapshots=[] if control is None else list(control.actions.snapshot_states.values()),
                    snapshot_calls=0 if control is None else control.actions.snapshot_calls,
                    admission_cpu=None if control is None else dict(checks=control.admission.checks,
                        total_ns=control.admission.total_ns,max_ns=control.admission.max_ns),
                    overbudget_admission=[] if control is None else list(control.admission.trace),
                    output_control_events=[] if control is None else list(control.output_events),
                    fairness_events=list(fairness.events[fairness_start:]),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    lane=job.get('lane','LIVE_MEASURED_CHARACTERIZATION_SHARED_HOST'),
                    scope='Finite natural generation. Capacity remains unqualified below the registered denominator.',
                    risk_model='Experimental planned exposure. No complete-token deadline guarantee.')
                write(target/f"{cell['label']}_{scenario_id}.json",data)
                assert integrity,'Quarantined serving result. Preserve the saved raw requests and work ledger.'
                summaries.append(dict(label=cell['label'],scenario=scenario_id,requests=len(cases),full_R4=True))
            finally:
                if observer:observer.close()
                ledger.close()
        if control:control.close()
        e.close();del e,control,fairness,data
        gc.collect();torch.cuda.empty_cache()
    return dict(serving=summaries)

def run_job(model,job,target):
    if job.get('diagnostic_barrier'):
        from .alternative_diagnostic import run_job as alternative
        return alternative(model,job,target)
    if job['kind']=='dual_serving':return serving(model,job,target)
    if job['kind']=='dual_quality':
        from .quality import run_job as quality
        return quality(model,job,target)
    raise ValueError(job['kind'])
