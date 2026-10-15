"""Validate completed evidence and preserve original worktree state."""
import collections
import datetime
import gzip
import subprocess
from poc_looped_multigpu_kv_placement.contract import *
from poc_looped_multigpu_kv_placement.scripts.gpu_session import contexts
from poc_looped_moe_ep.scripts.final_audit import burner_worker_ancestor


def read(path):return json.loads(path.read_text())


def main():
    raw=ROOT/'results/raw';results=ROOT/'results';checks={}
    env=read(results/'env_and_revisions.json');frozen=read(results/'frozen_workload.json')
    checks['spec_snapshot_matches']=sha(ROOT/'authoritative_spec.md')==env['spec_sha256']
    checks['frozen_token_artifact_matches']=sha(results/'frozen_tokens.json.gz')==frozen['token_artifact_sha256']
    expected={'ouro_tp1_native_v1':162,'ouro_tp2_native_v1':162,'ouro_tp4_native_v1':225}
    completed=[]
    for name,count in expected.items():
        directory=raw/name
        assert read(directory/'status.json')['success'] and read(directory/'guard_status.json')['success']
        paths=[p for p in directory.glob('P*.repeat*.json') if not p.name.endswith('.memory.json')]
        assert len(paths)==count,(name,len(paths),count)
        owned=read(directory/'loaded_ownership.json');tp=owned[0]['world']
        assert len(owned)==tp and {r['rank'] for r in owned}==set(range(tp))
        assert {r['GPU_uuid'].removeprefix('GPU-') for r in owned}==set(UUIDS[:tp])
        for rank in owned:
            assert rank['cache_slots']==192 and rank['physical_layers']==48
            assert len({t['storage_ptr'] for t in rank['cache']})==192
            assert all(g['Q_heads']==16//tp and g['KV_heads']==16//tp for g in rank['geometry'])
        for path in paths:
            value=read(path)
            assert len(value['requests'])==value['concurrency']
            assert all(len(r['output_ids'])==64 and r['num_cached_tokens']==0 for r in value['requests'])
        completed.append(dict(name=name,measurement_files=len(paths),tp=tp,ownership='ACTUAL_TENSORS_AND_UUIDS'))
    for name in ['ouro_replicas_tp1_dp4_v1','ouro_replicas_tp2_dp2_v1']:
        directory=raw/name;assert read(directory/'status.json')['success'] and read(directory/'guard_status.json')['success']
        records=[read(path) for path in directory.glob('P*.repeat*.json')]
        assert len(records)==18
        assert all(sum(len(a['requests']) for a in r['actors'])==r['concurrency'] for r in records)
        ranks=[rank for path in directory.glob('actor*.ownership.json') for rank in read(path)]
        assert len(ranks)==4 and {r['GPU_uuid'].removeprefix('GPU-') for r in ranks}==set(UUIDS)
        completed.append(dict(name=name,measurement_files=18,ownership='FOUR_PHYSICAL_NATIVE_REPLICA_RANKS'))
    hg=raw/'huginn_reference_native_v1';assert read(hg/'status.json')['success']
    assert len(list(hg.glob('P*.json')))==48
    checks['native_campaigns_complete']=True
    for name in ['ouro_tp4_kernel_profile_v1','ouro_tp4_eager_source_v2','ouro_tp4_outer_events_v1',
                 'ouro_tp4_natural_quality_v1','huginn_natural_quality_v1','huginn_default32_cache_probe_v1']:
        assert read(raw/name/'status.json')['success'] and read(raw/name/'guard_status.json')['success'],name
    observer=read(results/'observer_validity.json');correctness=read(results/'native_correctness_and_pairs.json')
    assert correctness['same_config_repeat_parity'] and observer['outer_events']['parity']
    assert observer['outer_events']['observer_percent']['mean']<5
    assert observer['ouro_tp4_eager_source_v2']['quantitative_validity']=='UNVALIDATED'
    h32=read(raw/'huginn_default32_cache_probe_v1/smoke_clean.json')
    assert h32['cache']['slot_count']==132
    reference=read(raw/'ouro_reference_v1/reference.json')['output_ids']
    assert reference==read(raw/'ouro_tp1_smoke_v3/smoke.json')['tokens']==read(raw/'ouro_tp4_smoke_v1/smoke.json')['tokens']
    checks['bounded_reference_and_observer_parity']=True
    checks['official_benchmark_accuracy']='NOT_RUN'
    checks['candidate_runtime']='NOT_RUN'
    # Inspect the dirty original without staging or changing any path there.
    original=Path('/home/esjung/MLLM-EP-github')
    current={label:hashlib.sha256(subprocess.check_output(command,cwd=original)).hexdigest() for label,command in
        [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]}
    checks['original_staged_diff_unchanged']=current['staged']==env['original_diff_sha256']['staged']
    checks['original_unstaged_diff_unchanged']=current['unstaged']==env['original_diff_sha256']['unstaged']
    assert checks['original_staged_diff_unchanged'] and checks['original_unstaged_diff_unchanged']
    burner=read(results/'final_burner_restore.json')
    assert burner['verification']=='RUNNING_ON_ALL_FOUR_ASSIGNED_GPUs'
    live_contexts=contexts(GPUS);assert len(live_contexts)==4
    for pid in live_contexts:burner_worker_ancestor(pid)
    checks['scoped_burners_restored']=True
    tests=subprocess.run(['python','-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],
        cwd=PROJECT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert tests.returncode==0
    write(results/'test_validation.json',dict(exit_code=tests.returncode,output=tests.stdout))
    prose=[str(ROOT/name) for name in ['README.md','report.md','reproduce.md','architecture_contract.md',
        'cache_contract.md','prior_overlap.md','timing_contract.md']]
    lint=subprocess.run(['python','/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py',*prose,'--json'],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert lint.returncode==0,lint.stdout
    write(results/'writing_validation.json',json.loads(lint.stdout))
    gates=dict(verdict='NO_GO_NATIVE_TP_SUFFICIENT',decision_scope='Tested Ouro fixed-R4 full-cache regimes. Not global native optimality.',
        flags=['MODEL_B_TP_BLOCKED','PARTIAL_MODEL_COVERAGE'],
        gate_0='PASS: Ouro physical TP1/2/4 and real pretrained Huginn single-GPU.',
        M='NOT_DEMONSTRATED: no candidate capacity benefit. Seven largest full-KV requests exceed aggregate HBM.',
        L='NOT_DEMONSTRATED: no candidate clean E2E gain. Native TP4 is fastest tested B1.',
        T='NOT_DEMONSTRATED: native four-GPU TP/DP controls measured. No candidate throughput gain.',
        X='BLOCKED: one-family physical TP coverage. Huginn 55-head TP unsupported.',
        Q='BOUNDED_NATIVE_CHECKS_PASS. Different TP output variation retained. Official accuracy and candidate quality NOT_RUN.',
        stage_B='OWNERSHIP_AND_TRANSFER_SCREEN_ONLY. Candidate latency UNKNOWN.',stage_C='NOT_RUN',
        heldout='NOT_SELECTED_OR_EXECUTED: no strong candidate discovery signal.',
        next_action='Do not build a new placement engine. Require a supported second physical recurrent family before broader claims.')
    write(results/'verdict_gates.json',gates)
    write(results/'supported_layouts.json',dict(ouro=dict(TP1='PHYSICAL_PASS',TP2='PHYSICAL_PASS',TP4='PHYSICAL_PASS',
        TP1_DP4='PHYSICAL_PASS_THROUGHPUT',TP2_DP2='PHYSICAL_PASS_THROUGHPUT',max_total_positions=32768),
        huginn=dict(HF_TP1='REAL_PRETRAINED_REFERENCE',TP2='MODEL_B_TP_BLOCKED',TP4='MODEL_B_TP_BLOCKED',
        max_total_positions=4096,primary_depth=16,H32='TINY_CACHE_PROBE_ONLY')))
    write(results/'correctness.json',dict(reference_smoke_tokens=reference,same_config_64_token_repeat_parity=True,
        outer_event_parity=True,native_kernel_profile_parity=True,eager_source_parity=True,
        cross_TP_continuation=correctness['pairs'],huginn_repeat_parity=read(results/'huginn_summary.json')['repeat_output_parity'],
        compress_s16_exact_alternative=False,official_quality='NOT_RUN',candidate_accuracy='NOT_RUN',
        natural_sanity='Ouro math correct. Two Ouro cases capped at512. Huginn H16 math wrong and coding incomplete. Document unscored.'))
    write(results/'final_audit.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,
        campaigns=completed,original_current_diff_sha256=current,failed_logs_preserved=True,
        source_provenance='Per-run source hashes remain in run_contract.json. Final artifact hashes are separate.'))
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in ROOT.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'}
    assert all(v['bytes']<95*1024**2 for v in manifest.values())
    write(results/'artifact_manifest.json',manifest)
    print('Completed campaign audit, native correctness checks, report gates, dirty-work preservation, and burner restoration.')


if __name__=='__main__':main()
