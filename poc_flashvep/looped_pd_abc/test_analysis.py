"""CPU checks for report accounting. No GPU inference occurs here."""
import pytest
from .critical_path_analysis import PHASES,partition_statistics,fixed_timeline_oracle

def record(elapsed,values):
 return dict(phase_order_valid=True,harness_E2E_ms=elapsed,
  **{name+'_ms':value for name,value in zip(PHASES,values)})

def test_host_partition_has_one_explicit_denominator():
 rows=[record(100,[10,10,10,10,10,40,10]),record(200,[0,0,0,0,0,200,0])]
 result=partition_statistics(rows)
 assert sum(r['share_of_summed_request_E2E_percent'] for r in result)==pytest.approx(100)
 assert result[0]['share_of_summed_request_E2E_percent']==pytest.approx(100/30)
 assert result[0]['mean_request_share_percent']==pytest.approx(5)

def test_host_partition_rejects_a_missing_interval():
 with pytest.raises(ValueError,match='does not equal'):
  partition_statistics([record(100,[1,1,1,1,1,1,1])])

def test_fixed_timeline_oracle_does_not_remove_prefill_or_decode():
 rows=[record(100,[10,10,10,20,10,30,10])]
 result=fixed_timeline_oracle(rows)
 assert result['frozen_timeline_p95_E2E_ms']==80
 assert result['frozen_timeline_p95_ratio']==1.25
 assert result['evidence']=='ORACLE_FROZEN_TIMELINE_NOT_MEASURED_SPEEDUP'

def test_invalid_order_is_not_silently_clamped():
 row=record(100,[0,0,0,-10,0,110,0]);row['phase_order_valid']=False
 assert partition_statistics([row])==[]
 assert fixed_timeline_oracle([row]) is None

def test_q1_bf16_posthoc_selection_is_not_guaranteed_to_choose_R4():
 import torch
 from .ouro_gate_audit import q1_posthoc_selection
 # These are synthetic inputs, not a measured checkpoint gate distribution.
 inputs=[torch.full((1,1,1),2.2) for _ in range(4)]
 fp32,cdf32=q1_posthoc_selection(inputs)
 bf16,cdf16=q1_posthoc_selection([x.to(torch.bfloat16) for x in inputs])
 assert fp32.item()==3 and bf16.item()==2
 assert cdf32[0,0,2]<1 and cdf16[0,0,2]==1

def test_shadow_routing_preserves_the_actual_state_dictionary(monkeypatch,tmp_path):
 import json,time
 from types import SimpleNamespace as NS
 from . import placement,shadow_worker,workload_worker
 monkeypatch.setattr(placement,'install',placement.install)
 monkeypatch.setattr(shadow_worker,'ROOT',tmp_path)
 monkeypatch.setenv('ABC_MANIFEST_RUN','shadow_test')
 request=NS(request_id='old',prompt_token_ids=list(range(128)),generated_token_ids=[],sampling_params=NS(max_tokens=32))
 active=NS(request=request,p='P',d='D',p_released=False,d_released=False,created=time.monotonic(),
  timings={'prefill_started':0.},loop_frontiers=[[64,128],[0,64],[0,0],[0,0]])
 engine=NS(model_config=NS(model_type='ouro',total_ut_steps=4),transfers={'old':active})
 peer=NS(name='P',role='prefill',client=NS(info={'max_active_requests':8,'num_blocks':4096}),reserved_blocks=10,slots=1)
 original=engine.transfers
 def fake_workload():
  placement.install(engine,'loop',tmp_path/'raw/shadow_test/placements.jsonl')
  score=engine.placement_key(peer,NS(request=NS(request_id='new')))
  assert score[0]>=0 and engine.transfers is original and engine.transfers['old'] is active
 monkeypatch.setattr(workload_worker,'main',fake_workload)
 shadow_worker.main()
 row=json.loads((tmp_path/'raw/shadow_test/shadow_states.jsonl').read_text())
 assert set(row['scores'])=={'queue','token','service','loop'}
 assert row['states'][0]['completed_prefill_frontiers']==active.loop_frontiers

def test_generic_transport_wrapper_returns_the_actual_PDConfig(monkeypatch,tmp_path):
 import vllm_rlt.pd as pd
 from . import transport_worker,worker
 original=pd.PDConfig
 monkeypatch.setattr(transport_worker,'ROOT',tmp_path)
 monkeypatch.setenv('ABC_TRANSPORT_RUN','transport_test')
 monkeypatch.setenv('ABC_TRANSPORT_CHUNK_MIB','16')
 monkeypatch.setenv('ABC_TRANSPORT_INFLIGHT_MIB','32')
 def fake_worker():
  config=pd.PDConfig(prefill_devices=(0,),decode_devices=(1,))
  assert type(config) is original and config.transfer_chunk_bytes==16*1024**2
  assert config.max_inflight_bytes==32*1024**2
 monkeypatch.setattr(worker,'main',fake_worker)
 transport_worker.main()
 assert pd.PDConfig is original

def test_request_SLO_keeps_a_gap_before_a_coalesced_emission():
 from .slo_compare import qualifies
 row=dict(finish_reason='length',output_tokens=3,ttft_ms=20,e2e_ms=500,
  token_events=[dict(time=.3,previous_time=.1,new_tokens=2,first=False)])
 assert not qualifies(row,150,1000,1000)
 assert qualifies(row,250,1000,1000)

def test_shadow_scoring_rounds_do_not_merge_repeated_worker_evaluations():
 from .shadow_analysis import groups
 rows=[dict(request_id='Q',role=role,worker=worker) for role,worker in
  [('prefill','P0'),('prefill','P1'),('prefill','P0'),('prefill','P1'),('decode','D0')]]
 assert [len(batch) for batch in groups(rows)]==[2,2,1]

def test_an_empty_decision_file_cannot_approve_the_reports(monkeypatch,tmp_path):
 from . import render
 monkeypatch.setattr(render,'ROOT',tmp_path)
 assert render.reviewed_decisions()=={}
 (tmp_path/'configs').mkdir();(tmp_path/'configs/reviewed_decisions.json').write_text('{}')
 with pytest.raises(ValueError,match='explicit completed scientific review'):render.reviewed_decisions()

def test_cost_calibration_requires_a_completed_registered_async_case(monkeypatch,tmp_path):
 import json
 from . import analyze,evidence_protocol
 monkeypatch.setattr(analyze,'ROOT',tmp_path)
 monkeypatch.setattr(evidence_protocol,'check',lambda folder:dict(valid=(folder/'accepted').exists()))
 old=tmp_path/'raw/ouro_calibration2_P128_C1_s0';old.mkdir(parents=True)
 (old/'closed_stats.json').write_text('{}')
 clean=tmp_path/'raw/ouro_calibration_async_P128_C1_s0';clean.mkdir()
 (clean/'status.json').write_text(json.dumps(dict(status='RUNNING')))
 assert analyze.calibration_folder('ouro',128,1)==old
 (clean/'status.json').write_text(json.dumps(dict(status='COMPLETE')))
 assert analyze.calibration_folder('ouro',128,1)==old
 (clean/'accepted').touch()
 assert analyze.calibration_folder('ouro',128,1)==clean

def test_background_IO_is_not_reported_as_request_critical_path(monkeypatch,tmp_path):
 import csv,json
 from . import profile_analysis,evidence_protocol
 monkeypatch.setattr(profile_analysis,'ROOT',tmp_path)
 monkeypatch.setattr(evidence_protocol,'check',lambda folder:dict(valid=True))
 folder=tmp_path/'raw/sample';folder.mkdir(parents=True)
 (folder/'configuration.json').write_text(json.dumps(dict(arguments=dict(attempt_id='current'))))
 stats=dict(pid=10,attempt_id='current',enqueued_records=2,background_written_records=2,
  foreground_enqueue_seconds=.001,foreground_max_enqueue_seconds=.0006,
  background_write_seconds=.08,background_fsync_batches=1,peak_queued_records=2,scope='Process lifetime')
 (folder/'async_evidence_summary_10.json').write_text(json.dumps(stats))
 (folder/'async_evidence_summary_11.json').write_text(json.dumps(dict(stats,pid=11,attempt_id='old')))
 profile_analysis.nonblocking_evidence()
 rows=list(csv.DictReader((tmp_path/'nonblocking_evidence_cost.csv').open()))
 assert len(rows)==1 and float(rows[0]['foreground_enqueue_ms'])==1
 assert float(rows[0]['background_write_ms'])==80
 assert rows[0]['evidence']=='MEASURED_INSTRUMENTATION_COUNTERS_NOT_ADDITIVE_E2E_COMPONENTS'

def test_canonical_replay_keeps_failed_evidence_and_uses_the_new_qualified_cut(monkeypatch,tmp_path):
 import json
 from . import frozen_runs,campaign,dispatch
 monkeypatch.setattr(frozen_runs,'ROOT',tmp_path)
 monkeypatch.setattr(frozen_runs,'matches',lambda folder:True)
 monkeypatch.setattr(campaign,'selected_cut',lambda model,length:3)
 (tmp_path/'configs').mkdir()
 (tmp_path/'configs/clean_chosen_cuts.json').write_text('{}')
 jobs=[]
 for label,status in [('A','COMPLETE'),('DP','FAILED')]:
  name=f'ouro_{label}_P512_C4_s0';folder=tmp_path/'raw'/name;folder.mkdir(parents=True)
  (folder/'status.json').write_text(json.dumps(dict(status=status)))
  (folder/'configuration.json').write_text(json.dumps(dict(arguments=dict(split_cut=1))))
  args=['--name',name,'--mode','pd','--model','ouro','--length','512']
  if label=='A':args+=['--enable-recurrence-split','--split-cut','1']
  jobs.append(dict(name=name,args=args,gpus=[0,1],env={}))
 (tmp_path/'configs/serving_campaign.json').write_text(json.dumps(jobs))
 calls=[]
 monkeypatch.setattr(dispatch,'execute',lambda path:calls.extend(json.loads(path.read_text())))
 frozen_runs.confirm()
 assert len(calls)==2
 assert all(task['env']['ABC_ASYNC_EVIDENCE']=='1' for task in calls)
 assert calls[0]['args'][calls[0]['args'].index('--split-cut')+1]=='3'
 assert calls[0]['name'].endswith('_s3')
 assert json.loads((tmp_path/'raw/ouro_DP_P512_C4_s0/status.json').read_text())['status']=='FAILED'

def test_protocol_rejects_a_reused_PID_with_a_stale_summary(monkeypatch,tmp_path):
 import json
 from . import evidence_protocol
 monkeypatch.setattr(evidence_protocol,'ROOT',tmp_path)
 (tmp_path/'configs').mkdir();folder=tmp_path/'raw/case';folder.mkdir(parents=True)
 (tmp_path/'configs/evidence_protocol_epoch.json').write_text(json.dumps(dict(telemetry_sha256='locked')))
 record=dict(mode='async_nonblocking',telemetry_sha256='locked',attempt_id='current',pid=12)
 (folder/'evidence_protocol.json').write_text(json.dumps(record))
 (folder/'configuration.json').write_text(json.dumps(dict(arguments=dict(attempt_id='current'))))
 (folder/'async_evidence_writer_12.json').write_text(json.dumps(record))
 (folder/'async_evidence_summary_12.json').write_text(json.dumps(dict(record,attempt_id='old',
  enqueued_records=2,background_written_records=2,error=None)))
 assert not evidence_protocol.check(folder)['valid']
 (folder/'async_evidence_summary_12.json').write_text(json.dumps(dict(record,
  enqueued_records=2,background_written_records=2,error=None)))
 assert evidence_protocol.check(folder)['valid']
