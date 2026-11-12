"""Join actual transfer generations to existing Mixed execution descriptors."""
import csv,json,subprocess
from collections import Counter
from .common import *


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def selected_measured(batch,metrics,window):
    return bool(batch['spans']) and window['start']<=batch['host_begin']<=window['end'] and all(
        s['request_id'] in metrics for s in batch['spans'])


def compatibility(spans):
    decode={s['depth'] for s in spans if s['phase']=='decode'}
    prefill=[s for s in spans if s['phase']=='prefill']
    depths={s['depth'] for s in prefill}
    return dict(has_decode=bool(decode),has_prefill=bool(prefill),
        any_aligned_pair=bool(decode&depths),strict_single_depth=len(decode|depths)==1,
        cross_depth_only=bool(decode and prefill and not decode&depths),
        unmatched_prefill_rows=sum(s['rows'] for s in prefill if s['depth'] not in decode))


def remote_evidence_present(tid,metrics,splits,checks):
    metric=metrics.get(tid);cut=splits.get(tid);check=checks.get(tid)
    return bool(metric and cut and check and cut.get('ownership')=='P_to_D' and
        check.get('passed') and check['expected']==check['actual'] and
        'd_received' in metric['timings'])


def parse_run(folder):
    config=read(folder/'rcc_contract.json');window=read(folder/'window.json');closed=read(folder/'closed_stats.json')
    assert read(folder/'status.json')['status']=='COMPLETE'
    measured={r['transfer_id']:r for r in closed['metrics'] if r['request_id'].startswith('r0_q')}
    assert len(measured)==config['count']==8
    splits={r['tid']:r for p in folder.glob('splits-*.jsonl') for r in lines(p)}
    checks={r['tid']:r for p in folder.glob('KV_checks-*.jsonl') for r in lines(p)}
    inputs=[folder/n for n in ('rcc_contract.json','window.json','closed_stats.json','requests.jsonl','configuration.json','status.json')]
    inputs+=list(folder.glob('splits-*.jsonl'))+list(folder.glob('KV_checks-*.jsonl'))
    frames=[];requests=[];missing=[]
    for tid,r in measured.items():
        t=r['timings'];cut=splits.get(tid);valid=checks.get(tid)
        if not remote_evidence_present(tid,measured,splits,checks):
            missing.append(tid);continue
        requests.append(dict(run=folder.name,model=config['model'],request_id=r['request_id'],transfer_id=tid,
            seed=config['seed']+int(r['request_id'].split('_q')[1]),prompt_tokens=cut['prompt_tokens'],cut=cut['cut'],
            source_cut_snapshot_Host=cut['time'],
            P_compute_complete_Host_observation=r['created_monotonic']+t['prefill_compute_completed'],
            D_receive_Host_observation=r['created_monotonic']+t['d_received'],
            D_activation_Host_observation=r['created_monotonic']+t['d_activated'],
            first_output_Host_observation=r['created_monotonic']+t['first_token'],
            exact_import_bytes=valid['actual']['KV_valid_bytes']+valid['actual']['hidden_bytes'],
            transfer_byte_identity_pass=valid['actual']==valid['expected'],
            exact_D_compute_ready='NOT_MEASURED',Source_ACK_time='NOT_MEASURED',
            per_step_KV_credit='NOT_MEASURED',next_Decode_deadline='NOT_MEASURED',
            timestamp_scope='Host notifications, not CUDA boundaries. Snapshot log does not time event completion.'))
    assert not missing,missing
    for path in sorted(folder.glob('fusion_probe_*.json')):
        inputs.append(path);probe=read(path);calls={r['batch_id']:r for r in probe['calls']}
        for batch in probe['batches']:
            if not selected_measured(batch,measured,window) or batch['operation']!='recurrent':continue
            c=compatibility(batch['spans'])
            # A Source batch can contain the same transfer IDs. It has no Decode spans.
            if not c['has_decode']:continue
            assert batch['loops']==1,'Do not equate multi-loop calls with individual core opportunities'
            for span in batch['spans']:
                assert span['rows']>0
                if span['phase']=='prefill':
                    assert span['request_id'] in splits and checks[span['request_id']]['passed']
                    assert splits[span['request_id']]['cut']<=span['depth']<splits[span['request_id']]['depth']
            call=calls[batch['id']]
            row=dict(run=folder.name,model=config['model'],probe_file=path.name,batch_id=batch['id'],
                host_begin=batch['host_begin'],host_end=batch['host_end'],rows=batch['rows'],
                Decode_request_count=sum(s['phase']=='decode' for s in batch['spans']),
                remote_prefill_rows=sum(s['rows'] for s in batch['spans'] if s['phase']=='prefill'),
                spans=batch['spans'],compatibility=c,observed_core_enclosure_ms=call['cuda_event_envelope_ms'],
                physical_operand_shapes=call['mm_shapes'],graph_requested=call['graph_requested'],
                scope='Historical completed model dispatch. All imported spans refer to measured transfer generations. Enclosure is not exclusive kernel time.')
            frames.append(row)
    assert frames,'No instrumented Decode frames exist in this diagnostic cell'
    mixed=[r for r in frames if r['remote_prefill_rows']]
    selected_ids={s['request_id'] for r in mixed for s in r['spans'] if s['phase']=='prefill'}
    for r in requests:r['observed_cross_phase_batches']=sum(any(s['request_id']==r['transfer_id'] and s['phase']=='prefill' for s in b['spans']) for b in mixed)
    pending_observations=[]
    for b in mixed:
        for s in b['spans']:
            if s['phase']!='prefill':continue
            m=measured[s['request_id']];t=m['timings'];base=m['created_monotonic']
            pending_observations.append(dict(batch_id=b['batch_id'],tid=s['request_id'],
                receive_notice_before_dispatch=b['host_begin']>=base+t['d_received'],
                activation_notice_before_dispatch=b['host_begin']>=base+t['d_activated']))
    all_released=all(x['requests']==0 and x['used_blocks']==0 for x in closed['worker_metrics'].values())
    summary=dict(run=folder.name,model=config['model'],input_tokens=config['chunk'],output_cap=config['cap'],
        GPU_mapping=config['gpus'],measured_requests=len(measured),Decode_core_executions=len(frames),
        observed_remote_Prefill_Decode_batches=len(mixed),
        observed_any_aligned_pair_batches=sum(r['compatibility']['any_aligned_pair'] for r in mixed),
        observed_strict_single_depth_batches=sum(r['compatibility']['strict_single_depth'] for r in mixed),
        observed_cross_depth_only_batches=sum(r['compatibility']['cross_depth_only'] for r in mixed),
        observed_unmatched_prefill_rows=sum(r['compatibility']['unmatched_prefill_rows'] for r in mixed),
        selected_remote_prefill_rows=sum(r['remote_prefill_rows'] for r in mixed),
        remote_requests_observed_co_batched=len(selected_ids),Native_Mixed_used_observed_opportunities=len(mixed),
        unused_legal_opportunities='NOT_MEASURED',KV_or_deadline_rejections='NOT_MEASURED',
        all_measured_import_checks_pass=len(requests)==8 and all(r['transfer_byte_identity_pass'] for r in requests),
        closure_requests_and_KV_empty=all_released,host_notice_consistency=pending_observations,
        evidence='TRACE_ANALYSIS',
        scope='Selected-batch opportunity lower bounds from historical N8 diagnostics. No full ready queue, primary long-prompt Serving, clean latency, or sustainable capacity claim.')
    return summary,frames,requests,inputs


def cpu_replay(frames):
    import sys
    sys.path.insert(0,str(RLT))
    from vllm_rlt import SamplingParams,SchedulerConfig
    from vllm_rlt.request import Request,RequestPhase,Operation
    from vllm_rlt.core.scheduler import ScheduledRequest,SchedulerOutput
    from vllm_rlt.core.scheduling_policy import MixedPolicy
    from vllm_rlt.worker.executor import build_execution_batch
    checks=[]
    for row in frames:
        if not row['remote_prefill_rows']:continue
        depth=4 if row['model']=='ouro' else 32;scheduled=[]
        for s in row['spans']:
            decode=s['phase']=='decode';phase=RequestPhase.RECURRENT if decode else RequestPhase.PREFILLING_RECURRENT
            length=s['position'] if decode else s['position']+s['rows']
            request=Request(s['request_id'],[1]*length,SamplingParams(min_loops=depth,max_loops=depth),
                generated_token_ids=[2] if decode else [],loops_done=s['depth'],phase=phase)
            scheduled.append(ScheduledRequest(request,token_start=s['position'],token_count=s['rows'],
                depth=s['depth'],chunk_start=None if decode else s['position'],phase=phase))
        expected=SchedulerOutput(None,scheduled,1)
        class ObservedReadySet:
            def has_ready(self,phase):return phase in (RequestPhase.RECURRENT,RequestPhase.PREFILLING_RECURRENT)
            def take_phases(self,phases):
                assert phases==(RequestPhase.RECURRENT,RequestPhase.PREFILLING_RECURRENT)
                return expected
        selected=MixedPolicy(SchedulerConfig(mode='mixed')).schedule(ObservedReadySet())
        batch=build_execution_batch(selected,depth)
        exact=batch.operation==Operation.RECURRENT and batch.num_rows==row['rows'] and [s.depth for s in batch.spans]==[s['depth'] for s in row['spans']]
        assert exact
        checks.append(dict(run=row['run'],batch_id=row['batch_id'],passed=exact,rows=batch.num_rows))
    return dict(passed=bool(checks) and all(r['passed'] for r in checks),checks=checks,
        evidence='CORRECTNESS_ONLY',scope='Actual existing policy and execution binder on already-observed selected descriptors. This CPU check does not simulate unseen ready work or GPU timing.')


def main():
    initialize();preserve();summaries=[];frames=[];request_rows=[];inputs=[]
    old=read(OLD/'configs/cpu_audit.json');source_rows=[]
    for name,expected in old['source_hashes'].items():
        actual=digest(name);source_rows.append(dict(path=name,old_SHA256=expected,current_SHA256=actual,unchanged=actual==expected))
    critical=('split.py','boundary.py','core/scheduling_policy.py','core/scheduler.py','worker/executor.py','worker/model_runner.py','pd/connector.py')
    source_pass=all(r['unchanged'] for r in source_rows if any(r['path'].endswith(s) for s in critical))
    write(ROOT/'analysis/source_equivalence_audit.json',dict(rows=source_rows,critical_source_match=source_pass,
        source_release='3b7a722898dab5f6dd797b474c1ff461659d8fa1',evidence='TRACE_ANALYSIS',
        distinction='Same execution capability does not prove every legal online opportunity is consumed.'))
    for model in ('ouro','huginn'):
        for tag in ('diag128_v1','diag256_v2'):
            folder=OLD/'raw'/f'{tag}_{model}_A_native_mixed_0'
            summary,batches,requests,paths=parse_run(folder)
            summaries.append(summary);frames+=batches;request_rows+=requests;inputs+=paths
    replay=cpu_replay(frames);write(ROOT/'analysis/existing_policy_descriptor_replay.json',replay)
    write(ROOT/'analysis/observed_opportunities.json',dict(runs=summaries,evidence='TRACE_ANALYSIS',
        ready_queue_complete=False,opportunity_scope='Actual selected opportunities, not every contemporaneous legal candidate.',
        new_GPU_measurements='NOT_RUN'))
    write(ROOT/'raw/selected_D_core_frames.json',frames)
    write(ROOT/'raw/remote_request_chronology.json',request_rows)
    for filename,rows in [('observed_opportunities.csv',summaries),('remote_request_chronology.csv',request_rows)]:
        fields=[k for k in rows[0] if k not in ('host_notice_consistency',)]
        with (ROOT/'analysis'/filename).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write(ROOT/'state/input_evidence_manifest.json',dict(files=[dict(path=str(p.relative_to(REPO)),SHA256=digest(p),bytes=p.stat().st_size) for p in sorted(set(inputs))],
        original_files_modified=False,scope='Existing immutable evidence. New files contain derived rows only.'))
    write(ROOT/'state/phase_gate.json',dict(time=time.time(),critical_source_match=source_pass,
        CPU_descriptor_replay_pass=replay['passed'],historical_remote_mixed_observed=all(r['observed_remote_Prefill_Decode_batches']>0 for r in summaries),
        candidate_mechanism='Existing A import plus Native Mixed',
        equivalence_verified=source_pass and replay['passed'],
        Phase_B='NOT_RUN',Phase_C='NOT_RUN',
        reason='Requested immediate stop: the proposed Remote-ready cross-depth path already exists and was used in the historical GPU traces.',
        overall_scope='No new mechanism beyond Existing Mixed. Long-prompt Decode-safe online headroom remains unmeasured.'))
    preserve()
    print(json.dumps(dict(source_match=source_pass,CPU_replays=len(replay['checks']),
        runs=[{k:r[k] for k in ('run','Decode_core_executions','observed_remote_Prefill_Decode_batches',
            'observed_any_aligned_pair_batches','observed_cross_depth_only_batches',
            'observed_unmatched_prefill_rows')} for r in summaries]),indent=2))


if __name__=='__main__':main()
