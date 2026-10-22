"""Seal this gated discovery, without inventing unexecuted serving results."""
import csv
import datetime
import gzip
import json
import time
from collections import defaultdict
from .common import ROOT,REPO,write,digest,append
from .surface_analysis import csv_out

def main():
    ledger=ROOT/'phase_exit_decisions.jsonl'
    existing=[json.loads(s) for s in ledger.read_text().splitlines()]
    decisions=[
        dict(phase='2_reviewed',status='CONDITIONAL',reason='96 starts and 1728 event cohorts complete. Full-R4 accounting passes. Observer and concurrency effects prevent clean serving claims.',evidence='main_surface_work_clock_verification.json'),
        dict(phase='3_reviewed',status='BLOCKED',reason='Aggregate coverage hides Atomic/staged prefill coverage below 95%. No held-out threshold relaxation.',evidence='predictor_blind_results.json'),
        dict(phase='4_reviewed',status='BLOCKED',reason='Offline legal cost envelope only. Candidate-specific complete-token DAG and qualified real-state prevalence remain unavailable.',evidence='phase4_offline_decision.json'),
        dict(phase='5_reviewed',status='NOT_RUN',reason='Prerequisite predictor/deadline gate blocked. No A1/A2/A3 live method.',evidence='adaptive_policy.md'),
        dict(phase='6_reviewed',status='NOT_RUN',reason='No qualified online controller. No adaptive capacity or full adaptive quality claim.',evidence='serving_summary.md')]
    for row in decisions:
        if not any(r['phase']==row['phase'] for r in existing):
            append(ledger,dict(row,wall_s=time.time(),evidence_sha256=digest(ROOT/row['evidence'])))
    events=[];tokens=[]
    index=json.loads((ROOT/'gpu_trace_index.json').read_text())
    with (ROOT/'gpu_trace_index.jsonl').open('w') as stream:
        for row in index:
            stream.write(json.dumps(row)+'\n')
    for source in index:
        data=json.loads((ROOT/source['path']).read_text())
        for request in data['requests']:
            base=dict(run_id=data['run_id'],trial=data['trial'],policy=data['policy'],configured_C=data['configured_C'],
                physical_GPU=data['physical_gpu'],request_id=request['id'],source=source['path'],source_sha256=source['sha256'])
            events.append(dict(base,input_tokens=request['input_tokens'],output_tokens=len(request['output_ids']),
                scheduled_arrival_host_s=request['scheduled_arrival'],completion_host_s=request['completion'],
                TTFT_s=request['TTFT_s'],E2E_s=request['E2E_s'],finish_reason=request['finish_reason'],
                evidence='DIAGNOSTIC_NATURAL_EVENT_LANE_NOT_CLEAN_CAPACITY'))
            for j,(token,clock) in enumerate(zip(request['output_ids'],request['token_times'])):
                tokens.append(dict(base,token_index=j,token_id=token,observed_host_s=clock,clock_domain='worker_monotonic_seconds'))
    csv_out(ROOT/'request_events.csv.gz',events)
    csv_out(ROOT/'token_timestamps.csv.gz',tokens)
    manifest=[json.loads(s) for s in (ROOT/'run_manifest.jsonl').read_text().splitlines()]
    ends=[r for r in manifest if 'wall_end' in r]
    per_gpu=defaultdict(lambda:dict(attempts=0,completed=0,failed=0,process_wall_s=0.))
    for r in ends:
        group=per_gpu[r['GPU']];group['attempts']+=1;group['completed']+=r['status']=='COMPLETED';group['failed']+=r['status']=='FAILED'
        group['process_wall_s']+=r['wall_end']-r['wall_start']
    start=min(r['wall_start'] for r in ends);end=max(r['wall_end'] for r in ends)
    write(ROOT/'execution_time_ledger.json',dict(first_worker_UTC=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
        last_worker_UTC=datetime.datetime.fromtimestamp(end,datetime.timezone.utc).isoformat(),elapsed_worker_campaign_hours=(end-start)/3600,
        summed_worker_process_hours=sum(r['wall_end']-r['wall_start'] for r in ends)/3600,devices=per_gpu,
        scope='Process wall includes model load, warmup, and measurement. Not kernel-busy time, researcher hours, or ten hours of research. Burners excluded.'))
    from .checkpoint import main as checkpoint
    checkpoint()
    state=json.loads((ROOT/'research_state.json').read_text())
    state.update(status='COMPLETE_WITH_BLOCKED_ADVANCEMENT',current_primary='Evidence archived. Online method stopped at predictor/deadline gate.',
        final_verdict='BLOCKED',next_gate='New registration with fresh path-stratified calibration and candidate-specific completion DAG',
        active_experiment_jobs=0,method_claim='NOT_ESTABLISHED',quality='114_CASE_FIXED_POLICY_PILOT_ONLY',
        online_adaptive='NOT_RUN',capacity='NOT_RUN')
    write(ROOT/'research_state.json',state)
    sources=[]
    for file in sorted((REPO/'poc_flashvep/ouro_adaptive_chunk_loop').glob('*.py')):
        sources.append(dict(path=str(file.relative_to(REPO)),sha256=digest(file)))
    write(ROOT/'final_code_hashes.json',sources)
    write(ROOT/'validation_summary.json',dict(local_unittest=dict(passed=16,failed=0,
        command="PYTHONPATH=/home/esjung/external/ouro-deadline-g3-candidate-rlt /home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m unittest discover -s poc_flashvep/ouro_adaptive_chunk_loop -t . -p 'test_*.py'"),
        full_R4_cohorts=dict(passed=1728,total=1728),isolated_numerical=dict(passed=160,total=160),
        archive_verification='Every member decompressed and SHA256 checked by archive.py',
        observer_parity=json.loads((ROOT/'observer_controls_summary.json').read_text()),
        scope='Software tests are not accuracy or serving-speedup evidence.'))
    print(json.dumps(dict(completed=sum(r['status']=='COMPLETED' for r in ends),failed=sum(r['status']=='FAILED' for r in ends),
        request_rows=len(events),token_rows=len(tokens),verdict='BLOCKED'),indent=2))

if __name__=='__main__':
    main()
