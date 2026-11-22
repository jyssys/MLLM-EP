"""Test a role-local compiler intervention after the fixed-output safety failure."""
import argparse
import hashlib
import json
import time
import zipfile
from .common import ROOT, REPO, SOURCE, write, digest, preserve
from .final_confirmation_jobs import clone
from .registration_deferral_campaign import registration_signature
from .run import save, execute


PLAN = 'decode_only_compile_correctness_frozen_v3.json'


def role_signature(cfg):
    return dict(registration_signature(cfg),
                D_execution_family_override=cfg.get('D_execution_family_override'),
                preserve_compiled_decode_only=bool(cfg.get('preserve_compiled_decode_only')),
                D_compiled_custom_ops_all=bool(cfg.get('D_compiled_custom_ops_all')))


def capsule(parent):
    base = ROOT / 'references/source_capsules' / parent['source_capsule']
    assert digest(base) == parent['source_capsule_SHA256']
    path = base.with_name('decode_only_compile_overlay_v1.zip')
    if path.exists():
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['base_capsule_SHA256'] == digest(base)
        assert meta['SHA256'] == digest(path)
        assert meta['files']['poc_flashvep/dense_ar_longbench400_pd/role_compile_options.py'] == digest(SOURCE/'role_compile_options.py')
        return path.name, digest(path)
    with zipfile.ZipFile(base) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    old = dict(files)
    worker = 'poc_flashvep/dense_ar_longbench400_pd/standard_worker.py'
    adapter = 'poc_flashvep/dense_ar_longbench400_pd/compiled_full_adapter.py'
    helper = 'poc_flashvep/dense_ar_longbench400_pd/role_compile_options.py'
    text = files[worker].decode()
    pairs = [
        ('from ..dense_ar_bi0_shortanswer_pd.engine_options import options',
         'from .role_compile_options import options'),
        ("elif cfg.get('preserve_compiled_full_model'):",
         "elif cfg.get('preserve_compiled_full_model') or (role=='decode' and cfg.get('preserve_compiled_decode_only')):")]
    for before, after in pairs:
        assert text.count(before) == 1
        text = text.replace(before, after)
    files[worker] = text.encode()
    text = files[adapter].decode()
    before = "assert self.cfg.get('preserve_compiled_full_model')"
    assert text.count(before) == 1
    text = text.replace(before, before + " or (self.role=='decode' and self.cfg.get('preserve_compiled_decode_only'))")
    files[adapter] = text.encode()
    files[helper] = (SOURCE / 'role_compile_options.py').read_bytes()
    changed = [n for n in old if files[n] != old[n]]
    assert set(changed) == {worker, adapter}
    assert set(files) - set(old) == {helper}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, files[name])
    write(path.with_suffix('.json'), dict(SHA256=digest(path),
        base_capsule=base.name, base_capsule_SHA256=digest(base),
        changed_existing_files=changed, added_files=[helper],
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Only the declared D compiler and full-model adapter dispatch change. '
              'P retains its Mode0 options and original partial-prefill adapter. '
              'Disabled flags retain prior behavior. Live actor files remain unchanged. '
              'No output, timestamp, model, installed runtime or SLO modification.'))
    return path.name, digest(path)


def build():
    preserve()
    path = ROOT / 'configs' / PLAN
    assert not path.exists()
    parent = json.loads((ROOT / 'configs/request_registration_controlled_bridge_frozen_v1.json').read_text())
    source, sha = capsule(parent)
    original = ROOT/'references/source_capsules'/source
    path_capsule=original.with_name('decode_only_compile_custom_ops_overlay_v1.zip')
    assert not path_capsule.exists()
    with zipfile.ZipFile(original) as archive:
        files={n:archive.read(n) for n in archive.namelist()}
    worker='poc_flashvep/dense_ar_longbench400_pd/standard_worker.py'
    helper='poc_flashvep/dense_ar_longbench400_pd/decode_compile_custom_ops.py'
    before='from .role_compile_options import options'
    assert files[worker].decode().count(before)==1
    files[worker]=files[worker].decode().replace(before,'from .decode_compile_custom_ops import options').encode()
    files[helper]=(SOURCE/'decode_compile_custom_ops.py').read_bytes()
    with zipfile.ZipFile(path_capsule,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,files[name])
    write(path_capsule.with_suffix('.json'),dict(SHA256=digest(path_capsule),
        base_capsule=source,base_capsule_SHA256=sha,changed_existing_files=[worker],added_files=[helper],
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},
        scope='Explicit official custom_ops all only on D. Keep compiler mode and other options unchanged. '
              'This follows a failed D-only strict test. Do not discard or rewrite that failure.'))
    source,sha=path_capsule.name,digest(path_capsule)
    block = next(c for c in parent['cells'] if c['output'] == 16 and c['D_cap'] == 1 and c['start'] == 0)
    actions = {}
    teacher_audits = []
    for lane, axis in enumerate(('Layer', 'Hybrid')):
        base = json.loads((ROOT / 'configs' / (block['jobs'][axis] + '.json')).read_text())
        teacher = 'Q4_Full36_Teacher_N8' if axis=='Layer' else 'Q4_HybridBoundaryTeacher_f0125_N8'
        tc = json.loads((ROOT / 'raw' / teacher / 'configuration.json').read_text())
        receipt = json.loads((ROOT / 'raw' / teacher / 'dispatcher_result.json').read_text())
        provenance = json.loads((ROOT / 'raw' / teacher / 'input_provenance.json').read_text())
        assert receipt['complete'] and tc['count'] == 8 and tc['reference_single']
        assert tc['execution_family']=='MATCHED_NO_COMPILE'
        assert tc['output_cap'] == 16 and tc['ignore_eos'] and not tc['batch_invariant']
        assert tc['dataset_SHA256'] == digest(ROOT / 'data/qualification8.json')
        assert provenance['model_revision'] == 'b968826d9c46dd6066d109eabc6255188de91218'
        teacher_audits.append(dict(name=teacher, configuration_SHA256=digest(ROOT/'raw'/teacher/'configuration.json'),
            source_SHA256=receipt['source_SHA256'], dataset_SHA256=tc['dataset_SHA256'],
            scope='Independent Mode0 full36 teacher with the legal query boundary. '
                  'Teacher source is older. Only its state/output is reused,not performance. '
                  'The preceding D-only path differs at the first D-owned layer. '
                  'The explicit all-custom-op intervention tests that dispatch difference.'))
        base.update(D_execution_family_override='INSTALLED_DEFAULT_COMPILE',
                    preserve_compiled_decode_only=True,D_compiled_custom_ops_all=True)
        jobs = []
        for count in (1, 8):
            name = f'DecodeCompileQ_{axis}_StrictN{count}_v3'
            cfg = clone(base,name,'qualification8.json',20.,2026112201,(lane*2,lane*2+1),source,sha,correctness=True)
            cfg.update(count=count,output_cap=16,ignore_eos=True,serial=True,
                       controlled_pipeline=True,axis_capture_hidden_last_only=False,
                       continuous_source_credit_limit=0,
                       gpu_state_digest=False,axis_import_hash_after_model=False)
            save(cfg)
            jobs.append(name)
        for count, dataset in ((32,'continuous_qualification32_natural.json'),(400,'longbench_balanced400_v1.json')):
            name = f'DecodeCompileQ_{axis}_Natural{count}_v3'
            cfg = clone(base,name,dataset,2.8,2026112201,(lane*2,lane*2+1),source,sha,correctness=True)
            save(cfg)
            jobs.append(name)
        actions[axis] = dict(lane=lane,teacher=teacher,jobs=jobs,signature=role_signature(base),
            gate=f'Decode_compile_{axis}_gate_v3.json')
    write(path,dict(time=time.time(),source_capsule=source,source_capsule_SHA256=sha,
        parent_plan_SHA256=digest(ROOT/'configs/request_registration_controlled_bridge_frozen_v1.json'),
        teacher_audits=teacher_audits,qualification=actions,
        symptom=dict(output=16,cap=1,starts=3,
            Layer_NativeStrong_ITL_CI_upper=1.0636997704981743,
            Hybrid_NativeStrong_ITL_CI_upper=1.0776688800981993),
        hypotheses=[
            'H0: Default compilation changes CustomOp dispatch in eager partial layers. '
            'Explicit all custom ops should restore the Mode0 full-prefill KV/hidden state.',
            'H1: Mode0 D full-model dispatch contributes the remaining decode tail. '
            'Restore only D compiled full36 calls while retaining the P path.',
            'H2: Adapter ownership/graph safety dominates. Keeping these hooks will retain the tail.',
            'H3: Client delivery dominates. D compile changes will not consistently reduce client tails.'],
        strict_harness_correction='V1 inherited a continuous source window into the serial C0 harness. '
            'Both jobs failed the existing serving assertion before LLM initialization. '
            'Keep those failures. V2 disables only that incompatible extra C3 window for strict serial tests. '
            'Natural32/400 retain the original continuous source window8.',
        scope='Adaptive role-local compiler intervention. Do not alter or interrupt the frozen54 bridge starts. '
              'StrictN1/N8,then Natural32/400 state/lifetime/quality. No timing claim from these observer-heavy proofs. '
              'A failed strict gate blocks subsequent stages and performance. No tensor correction or assertion removal.'))
    print('D_ONLY_COMPILE_CORRECTNESS_FROZEN',source,sha,flush=True)


def run(lane):
    import torch
    from .qualification import audit
    from .qualification_gate import compare
    from .quiescent_admission_qualification import strict_first
    from .quality_gate import compare as quality
    from .compute_throttle_qualification import no_mixed
    torch.set_num_threads(1)
    plan = json.loads((ROOT/'configs'/PLAN).read_text())
    axis, item = next((a,r) for a,r in plan['qualification'].items() if r['lane']==lane)
    path = ROOT/'state'/item['gate']
    assert not path.exists(), 'Do not repeat a terminal correctness outcome'
    results = []
    failed = None
    for index, name in enumerate(item['jobs']):
        try:
            assert execute(name), name
            state = audit(name)
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(v['status']=='PASS' for v in state['imported_prefix_KV'])
            if index == 0:
                numerical = strict_first(item['teacher'],name)
                passed = all(numerical.values())
            elif index == 1:
                numerical = compare(item['teacher'],name)
                passed = numerical['passed']
            else:
                numerical = quality(name)
                passed = numerical['passed']
            results.append(dict(name=name,state=state,numerical=numerical,passed=passed,
                                compute_rule=no_mixed(name)))
            assert passed, (name,numerical)
        except Exception as error:
            failed = dict(name=name,error=repr(error),status='FAILED')
            break
        write(ROOT/'state'/f'decode_compile_lane{lane}_progress_v3.json',dict(results=results))
        print('D_COMPILE_QUALIFICATION',name,'PASS',flush=True)
    passed = failed is None and len(results)==len(item['jobs'])
    stopped = [dict(name=n,status='NOT_RUN',reason='BLOCKED_CORRECTNESS') for n in item['jobs'][len(results):]
               if failed is None or n != failed['name']]
    write(path,dict(time=time.time(),passed=passed,full400_passed=passed,results=results,
        failure=failed,stopped=stopped,signature=registration_signature(
            json.loads((ROOT/'configs'/(item['jobs'][-1]+'.json')).read_text())),
        role_signature=item['signature'],source_capsule_SHA256=plan['source_capsule_SHA256'],
        evidence='CORRECTNESS_ONLY' if passed else 'FAILED',
        scope='Role-local D compiler proof. This does not establish decode safety,latency or an online dynamic opportunity.'))
    print('D_ONLY_COMPILE_TERMINAL',axis,passed,failed,flush=True)


def analyze():
    import bisect
    import numpy as np
    parent=json.loads((ROOT/'configs/request_registration_controlled_bridge_frozen_v1.json').read_text())
    blocks=[c for c in parent['cells'] if c['output']==16 and c['D_cap']==1]
    rows=[]
    for block in blocks:
        for arm,name in block['jobs'].items():
            folder=ROOT/'raw'/name
            requests=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
            client=[];worker=[];delivery=[]
            for request in requests:
                events=request['token_events']
                delivery.extend(1000*(r['time']-r['worker_time']) for r in events)
                for left,right in zip(events,events[1:]):
                    client.append(1000*(right['time']-left['time']))
                    worker.append(1000*(right['worker_time']-left['worker_time']))
            steps=json.loads((folder/'engine_steps_decode.json').read_text())['rows']
            queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows']
            starts=[r['start'] for r in steps]
            groups={}
            for query in queries:
                index=bisect.bisect_right(starts,query['time'])-1
                if index>=0 and query['time']<=steps[index]['end']:
                    groups.setdefault(index,[]).append(query)
            pure=[i for i,q in groups.items() if all(r['logical_phase']=='DECODE' for r in q)]
            walls=[steps[i]['wall_ms'] for i in pure]
            gaps=[1000*(steps[i]['start']-steps[i-1]['end']) for i in pure if i-1 in pure]
            runtime=json.loads((folder/'actual_runtime_decode.json').read_text())
            pct=lambda a,q:float(np.percentile(a,q)) if a else None
            rows.append(dict(arm=arm,name=name,start=block['start'],
                D_compile_mode=runtime['compile_mode'],D_graph_mode=runtime['cudagraph_mode'],
                client_ITL_p99_ms=pct(client,99),worker_emission_p99_ms=pct(worker,99),
                worker_to_client_delivery_p99_ms=pct(delivery,99),
                pure_decode_host_step_p50_ms=pct(walls,50),pure_decode_host_step_p99_ms=pct(walls,99),
                consecutive_pure_decode_host_gap_p99_ms=pct(gaps,99),pure_host_steps=len(pure)))
    proofs={}
    for version in (1,2,3):
        for axis in ('Layer','Hybrid'):
            path=ROOT/'state'/f'Decode_compile_{axis}_gate_v{version}.json'
            if path.exists():proofs[f'{axis}_v{version}']=json.loads(path.read_text())
    complete=all(f'{axis}_v3' in proofs for axis in ('Layer','Hybrid'))
    controls=[]
    control_plan=ROOT/'configs/decode_compile_disabled_control_frozen_v1.json'
    if control_plan.exists():
        import torch
        from .qualification import audit
        from .quiescent_admission_qualification import strict_first
        from .metrics import summarize
        torch.set_num_threads(1)
        def state(name):
            base=ROOT/'raw'/name
            row=json.loads((base/'axis_events_decode.json').read_text())['states']['q0']
            values=torch.load(row['tensor_file'],map_location='cpu',weights_only=False)
            request=json.loads((base/'requests.jsonl').read_text().splitlines()[0])
            return row,values['hidden'][-1].float(),torch.stack(values['KV_samples']).float(),request['output_ids']
        def error(left,right):
            delta=left-right
            return dict(max_absolute=float(delta.abs().max()),RMS=float(delta.square().mean().sqrt()),
                relative_L2=float(delta.norm()/right.norm().clamp_min(1e-30)),
                exact_elements=int((left==right).sum()),elements=left.numel())
        for axis in ('Layer','Hybrid'):
            name=f'DecodeCompileControl_{axis}_StrictN1_v1'
            receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if not receipt.exists():continue
            assert json.loads(receipt.read_text())['complete']
            coverage=audit(name)
            teacher='Q4_Full36_Teacher_N8' if axis=='Layer' else 'Q4_HybridBoundaryTeacher_f0125_N8'
            original=strict_first(teacher,name)
            base,h,k,tokens=state(name)
            comparisons=[]
            for version in (2,3):
                action=f'DecodeCompileQ_{axis}_StrictN1_v{version}'
                actual,ah,ak,atokens=state(action)
                comparisons.append(dict(actual=action,full_KV_exact=base['KV']==actual['KV'],
                    differing_KV_layers=[i for i,(a,b) in enumerate(zip(base['KV'],actual['KV'])) if a!=b],
                    hidden_error=error(ah,h),KV_sample_error=error(ak,k),greedy_exact=atokens==tokens,
                    scope='Compare the unchanged current Mode0 control and D-only compiler on the same legal partition. '
                          'Samples do not replace full KV hashes. No new numerical tolerance is declared after observation.'))
                summarize(action)
            summarize(name)
            controls.append(dict(axis=axis,name=name,original_teacher_comparison=original,
                coverage_passed=coverage['passed'],new_path_comparisons=comparisons))
        write(ROOT/'state/decode_compile_disabled_control_terminal_v1.json',dict(time=time.time(),
            complete=len(controls)==2,results=controls,evidence='DIAGNOSTIC_CORRECTNESS_NEGATIVE_CONTROL',
            scope='A failed original-teacher comparison can predate the D-only intervention. '
                  'Do not call BI0 non-equivalence a transport or arithmetic error without stronger evidence. '
                  'The strict new-source gates remain failed. No performance or safety eligibility is inherited.'))
    write(ROOT/'analysis/decode_compile_fixed_output_residual_v1.json',dict(time=time.time(),
        rows=rows,correctness=proofs,correctness_stage_terminal=complete,disabled_controls=controls,
        evidence='TRACE_ANALYSIS_AND_CORRECTNESS_ONLY',
        scope='Same original I3 requests,output16,cap1,arrival and pair4/5. '
              'Worker emission and engine wall/gaps are host timestamps,not GPU-exclusive time. '
              'Different arms have different adapter/compiler paths. These comparisons alone do not isolate compile causality. '
              'A role-local D-only intervention must pass state correctness before performance.'))
    lines=['# LongBench400: Fixed Output에서 남은 Decode Tail','',
        'Natural400의 안전성 결과를 모든 고정 Output에 확대하지 않는다.',
        'Output16/Cap1은 같은 I3 Prompt16개와 독립 Start3회를 사용했다.',
        '등록 수정 후 Layer와 Hybrid는 Common-policy Native를 통과했다.',
        '하지만 두 Arm은 더 빠른 원래 NativeStrong의 ITL 기준을 통과하지 못했다.',
        'Layer/NativeStrong ITL p99 CI 상한은1.063700이다.',
        'Hybrid/NativeStrong ITL p99 CI 상한은1.077669이다.',
        '동일 기준의 Stock Token 상한은1.044486으로 통과했다.',
        '상대 기준은1.05다. 이 실패를 반올림으로 제거하지 않는다.','',
        '## 같은 실행 기록의 Host 경계','',
        '아래 값은 각 독립 Start의 통계량을 산술 평균한 값이다.',
        'Worker와 Client의 Timestamp를 사용한다. GPU-exclusive Kernel 비용은 아니다.','',
        '| Arm | D Compile Mode | Client ITL p99(ms) | Worker Emission p99(ms) | Delivery p99(ms) | Pure Decode Step p50(ms) | Pure Decode Step p99(ms) | Host Gap p99(ms) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    fields=('client_ITL_p99_ms','worker_emission_p99_ms','worker_to_client_delivery_p99_ms',
            'pure_decode_host_step_p50_ms','pure_decode_host_step_p99_ms',
            'consecutive_pure_decode_host_gap_p99_ms')
    for arm in blocks[0]['jobs']:
        selected=[r for r in rows if r['arm']==arm]
        values=[float(np.mean([r[f] for r in selected])) for f in fields]
        lines.append(f'| {arm} | {selected[0]["D_compile_mode"]} | '+
                     ' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '## D-only Compile 검사','',
        'P의 Mode0와 Partial-prefill 계산은 유지했다.',
        'D의 Full36 경로만 원래 Compiled Model로 복원하는 별도 Source를 만들었다.',
        '원래 실행 중인54개 Bridge는 변경하지 않았다.',
        '새 경로는 Strict N1/N8과 Natural32/400 State·Lifetime·Quality를 순서대로 검사한다.',
        'N1의 V1 Harness는 연속 Source Window를 C0에 잘못 상속했다.',
        '기존 Assertion이 LLM 초기화 전에 이를 차단했다. 실패 로그를 보존했다.',
        'V2의 Serial State 검사만 해당 Window를0으로 정정했다.',
        'V2의 Natural Continuous 검사는 원래 Window8을 유지한다.','',
        '| 검사 | Terminal | Correctness PASS | 실패 |',
        '|---|---|---|---|']
    for version in (1,2,3):
        for axis in ('Layer','Hybrid'):
            value=proofs.get(f'{axis}_v{version}')
            lines.append(f'| {axis}/V{version} | {value is not None} | '+
                (str(value['passed'])+' | '+str(value.get('failure')) if value else 'PENDING | N/A')+' |')
    lines += ['', 'V3는 D에서 공식 Custom-op 설정을 명시적으로 `all`로 유지한다.',
        '이 경로도 엄격한 KV·Hidden 기준을 통과하지 못했다. Layer와 Hybrid의 Greedy16은 일치했다.',
        'Greedy 일치만으로 KV·Hidden Gate를 통과한 것으로 처리하지 않는다.',
        '새 경로의 N8/Natural32/Natural400과 성능은 BLOCKED_CORRECTNESS다.',
        '이는 계산 오류가 확정됐다는 뜻은 아니다. Numerical Path 차이와 구현 오류를 아직 모두 분리하지 못했다.']
    lines += ['', 'V2는 원래 Eager-prefix/Compiled-suffix Stage Teacher를 사용했다.',
        'V3는 Mode0 Full36 Boundary Teacher를 사용했다. 두 Teacher의 계약은 동결한 Plan에 남긴다.',
        '새 Teacher 선택으로 KV·Hidden 불일치를 통과 처리하지 않았다.',
        '기존 Source071에서 실패한 성능 Gate도 유지한다.',
        'D-only Source SHA256은 `49e494730ff48f3682d476d993122b4ba5b610b44e84621f484b4e207787bf87`이다.',
        'Custom-op Source SHA256은 `17021999bdb7436ac256e758f59ecf5e862dfb4c679ea75e15313bb4fd970dd0`이다.']
    if controls:
        lines += ['', '### 현재 Mode0의 Negative Control','',
            '같은 새 Source와 같은 Serial Harness에서 D Compile Flag만 모두 껐다.',
            '이 검사는 앞선 엄격한 불일치가 기존 경로에도 있었는지 확인한다.',
            '관측 후 새로운 Tolerance를 정해 실패한 Gate를 PASS로 바꾸지 않는다.','',
            '| Axis | 현재 Control/원래 Teacher KV 일치 | Hidden 일치 | Greedy 일치 |',
            '|---|---|---|---|']
        for row in controls:
            r=row['original_teacher_comparison']
            lines.append(f'| {row["axis"]} | {r["full_KV_exact"]} | {r["last_hidden_exact"]} | {r["greedy_exact"]} |')
        lines += ['', '| Axis / 새 경로 | Hidden Max Abs | Hidden RMS | Hidden Relative L2 | KV Sample RMS | Control 대비 Greedy16 일치 |',
            '|---|---:|---:|---:|---:|---|']
        for row in controls:
            for item in row['new_path_comparisons']:
                h=item['hidden_error'];k=item['KV_sample_error']
                lines.append(f'| {row["axis"]}/{item["actual"].rsplit("_",1)[-1]} | '+
                    f'{h["max_absolute"]:.6f} | {h["RMS"]:.9f} | {h["relative_L2"]:.9f} | '+
                    f'{k["RMS"]:.9f} | {item["greedy_exact"]} |')
        lines += ['', '새 D 경로와 현재 Control의 전체 KV Hash, Hidden 오차와 KV Sample 오차는 JSON에 저장한다.',
            'KV Sample은 모든 KV Tensor의 수치 오차 분포가 아니다.',
            'Layer는 P가 계산한 Lower28 KV가 일치했다. 불일치는 D가 계산한 Layer28에서 시작했다.',
            'Hybrid는 D가 Suffix의 모든 Layer를 계산하므로 전체 Prompt KV의 Layer0부터 차이가 있었다.',
            '이 차이는 D 경로 개입으로 발생했다. 개별 Operator의 정확한 수치 원인은 아직 미확정이다.',
            'Custom-op `all`만으로 원래 State를 복원한다는 가설은 이 검사에서 반증됐다.',
            '관측 후 Tolerance를 정하거나 State를 수정하지 않았다.',
            '이 진단 결과를 적격 성능 개선으로 사용하지 않는다.']
    lines += ['', '## 판정 범위','',
        'Compiler와 Adapter의 차이는 원인 후보다. 위의 Arm 간 상관관계만으로 인과성을 확정하지 않는다.',
        '정합성을 통과하지 못한 경로는 성능 또는 안전성 증거로 사용하지 않는다.',
        '구조적 수치 차이와 실제 KV Ownership 오류도 구분해야 한다.',
        'D-only 경로의 최종 성능 판정은 별도 적격 A/B가 필요하다.',
        'C3의 GPU-exclusive Operator 비중과 SM/HBM은 이 검사에서 NOT_MEASURED다.','',
        '- [원래 수정 후 Bridge](dense_ar_longbench400_registration_controlled_bridge.md)',
        '- [Host 경계와 새 정합성 결과](dense_ar_longbench400_2d_20261121/analysis/decode_compile_fixed_output_residual_v1.json)','']
    (REPO/'poc_flashvep/reports/dense_ar_longbench400_decode_compile_residual.md').write_text('\n'.join(lines),encoding='utf-8')
    print('D_ONLY_COMPILE_RESIDUAL_REPORT',len(rows),'start rows',complete,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('build','run','analyze'))
    parser.add_argument('--lane',type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    if args.action=='build':build()
    elif args.action=='run':run(args.lane)
    else:analyze()
