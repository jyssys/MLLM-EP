"""CPU checks for a user scope change. No GPU engine starts in these tests."""
import json

def configure(monkeypatch,tmp_path,models=('ouro','huginn')):
 from . import scope
 monkeypatch.setattr(scope,'ROOT',tmp_path)
 (tmp_path/'configs').mkdir(exist_ok=True)
 (tmp_path/'configs/model_scope.json').write_text(json.dumps(dict(active_models=list(models))))

def test_scope_defaults_to_the_original_models_without_an_override(monkeypatch,tmp_path):
 from . import scope
 monkeypatch.setattr(scope,'ROOT',tmp_path)
 assert scope.active_models()==('ouro','huginn','dory')

def test_explicit_scope_keeps_model_independent_controls(monkeypatch,tmp_path):
 from .scope import enabled,active_models
 configure(monkeypatch,tmp_path)
 assert active_models()==('ouro','huginn')
 assert enabled(dict(name='ouro_case',args=['--model','ouro']))
 assert enabled(dict(name='huginn_case',args=['--model','huginn']))
 assert enabled(dict(name='nixl_pair01',args=[]))
 assert not enabled(dict(name='dory_case',args=['--model','dory']))
 assert not enabled(dict(name='dory_official_B1_100',args=[]))
 assert not enabled(dict(name='wrapper',args=[],env=dict(ABC_MANIFEST_MODEL='dory')))

def test_dispatch_defers_in_memory_jobs_and_preserves_old_results(monkeypatch,tmp_path):
 from . import dispatch
 configure(monkeypatch,tmp_path);monkeypatch.setattr(dispatch,'ROOT',tmp_path)
 folder=tmp_path/'raw/dory_done';folder.mkdir(parents=True)
 (folder/'status.json').write_text('{"status":"COMPLETE"}')
 tasks=[dict(name='dory_done',args=['--model','dory']),dict(name='ouro_next',args=['--model','ouro'])]
 assert [task['name'] for task in dispatch.eligible_jobs(tasks,'frozen.json')]==['ouro_next']
 assert json.loads((folder/'status.json').read_text())['status']=='COMPLETE'
 data=json.loads((tmp_path/'state/model_scope_deferred_jobs.json').read_text())
 assert data['dory_done']['status']=='DEFERRED_BY_USER'

def test_empty_resume_never_changes_GPU_or_burner_state(monkeypatch,tmp_path):
 from . import dispatch
 configure(monkeypatch,tmp_path);monkeypatch.setattr(dispatch,'ROOT',tmp_path)
 monkeypatch.setattr(dispatch,'setup',lambda:None)
 folder=tmp_path/'raw/ouro_done';folder.mkdir(parents=True)
 (folder/'status.json').write_text('{"status":"COMPLETE"}')
 jobs=[dict(name='ouro_done',args=['--model','ouro']),dict(name='dory_deferred',args=['--model','dory'])]
 path=tmp_path/'jobs.json';path.write_text(json.dumps(jobs))
 def prohibited(*args):raise AssertionError('No GPU lease change is permitted for an empty resume')
 monkeypatch.setattr(dispatch.resources.base,'stop',prohibited)
 monkeypatch.setattr(dispatch.resources.base,'restore',prohibited)
 dispatch._execute(path)

def test_scale_manifest_contains_only_active_models_and_keeps_all_GPU_budgets(monkeypatch,tmp_path):
 import sys
 from . import campaign
 from .scope import job_model
 configure(monkeypatch,tmp_path);monkeypatch.setattr(campaign,'ROOT',tmp_path)
 monkeypatch.setattr(sys,'argv',['campaign','scale'])
 campaign.main()
 jobs=json.loads((tmp_path/'configs/scale_campaign.json').read_text())
 assert len(jobs)==174
 assert {job_model(task) for task in jobs}=={'ouro','huginn'}
 assert {len(task['gpus']) for task in jobs}=={2,4,8}

def test_decision_schema_requires_only_the_current_models(monkeypatch,tmp_path):
 from . import render
 configure(monkeypatch,tmp_path);monkeypatch.setattr(render,'ROOT',tmp_path)
 item=dict(status='NOT_MEASURED',reason='Test only',scale_status={str(n):'NOT_MEASURED' for n in (2,4,8)})
 value=dict(review_complete=True,**{track:dict(overall=dict(status='NOT_MEASURED',reason='Test only'),ouro=item,huginn=item) for track in ('A','B','C')})
 (tmp_path/'configs/reviewed_decisions.json').write_text(json.dumps(value))
 assert render.reviewed_decisions()==value
