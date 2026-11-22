"""Freeze an exact-condition paired test of the qualified scoped compiler."""
import argparse
import hashlib
import json
import time
import numpy as np
from .common import ROOT,REPO,write,digest,preserve
from .final_confirmation_jobs import clone
from .scoped_decode_compile_followup import signature
from .registration_deferral_campaign import registration_signature
from .run import save,execute


PLAN='scoped_compile_confirmation_frozen_v1.json'


def build():
    preserve();path=ROOT/'configs'/PLAN;assert not path.exists()
    source_plan=json.loads((ROOT/'configs/scoped_decode_compile_correctness_frozen_v1.json').read_text())
    old=json.loads((ROOT/'configs/request_registration_controlled_bridge_frozen_v1.json').read_text())
    blocks=[c for c in old['cells'] if c['output']==16 and c['D_cap']==1]
    prototypes={a:json.loads((ROOT/'configs'/(n+'.json')).read_text()) for a,n in blocks[0]['jobs'].items()
                if a!='TokenMode0Control'}
    for axis in ('Layer','Hybrid'):
        prototypes[axis+'Scoped']=dict(prototypes[axis],preserve_compiled_decode_only=True,
            D_execution_family_override='INSTALLED_DEFAULT_COMPILE',D_compiled_custom_ops_all=True,
            D_mode0_IR_norm_priority=False,D_partial_IR_norm_priority=True)
    proofs={};lanes=[[],[]]
    source,sha=source_plan['source_capsule'],source_plan['source_capsule_SHA256']
    for arm,base in prototypes.items():
        if arm.endswith('Scoped'):
            axis=base['axis_kind'];proofs[arm]=dict(gate=source_plan['qualification'][axis]['gate'],
                signature=signature(base),existing_scoped_proof=True)
            continue
        name=f'ScopedSourceQ_{arm}_Natural400_v1'
        first=4 if arm in ('Layer','Hybrid') else 6
        cfg=clone(base,name,'longbench_balanced400_v1.json',2.8,2026112201,(first,first+1),source,sha,correctness=True)
        save(cfg)
        gate=f'Scoped_source_{arm}_gate_v1.json'
        proofs[arm]=dict(job=name,gate=gate,signature=signature(base),existing_scoped_proof=False)
        lanes[0 if first==4 else 1].append(arm)
    cells=[];jobs=[]
    for block in blocks:
        arms={}
        for arm,base in prototypes.items():
            name=f'ScopedCompile_I3_O16_Cap1_s{block["start"]}_{arm}_v1'
            cfg=clone(base,name,block['dataset'],block['QPS'],block['seed'],(0,1),source,sha)
            cfg.update(output_cap=16,ignore_eos=True,scope='Adaptive exact-condition compiler/IR-path ablation. '
                'Same I3 prompts,arrival,cut,work,cap and source credit as the original fixed O16 bridge. '
                'All new paired arms use0/1. Do not pair their metrics with old4/5 starts. '
                'Within-axis Mode0 versus Scoped changes only the declared D numerical/compiler path. '
                'All strict failures and the original54 remain. No changed SLO or corrected token/timestamp.')
            for field in ('suffix_by_request','aggregate_work_audit'):
                if field in base:cfg[field]=base[field]
            proof=proofs[arm]
            if cfg.get('defer_D_request_registration'):
                cfg['registration_deferral_gate']=proof['gate']
            if arm.endswith('Scoped'):
                cfg['qualification_gate']=proof['gate'];cfg['D_compile_role_gate']=proof['gate']
            save(cfg);arms[arm]=name
        ordered=list(arms.values());np.random.default_rng(block['seed']+73).shuffle(ordered)
        jobs.extend(ordered)
        cells.append(dict(output=16,D_cap=1,start=block['start'],seed=block['seed'],QPS=block['QPS'],
            GPU_pair=[0,1],dataset=block['dataset'],jobs=arms))
    write(path,dict(time=time.time(),source_capsule=source,source_capsule_SHA256=sha,
        qualification=proofs,proof_lanes=lanes,cells=cells,performance_jobs=jobs,
        parent_plan_SHA256=digest(ROOT/'configs/request_registration_controlled_bridge_frozen_v1.json'),
        hypotheses=['H1: Keeping native compiled full decode while matching eager partial-prefill normalization reduces residual decode tails.',
                    'H2: Host/adapter/delivery cost remains dominant. The scoped compiler will not pass both Native gates.'],
        conditional_stop='If both new scoped actions fail correctness,all21 serving starts are NOT_RUN(BLOCKED_CORRECTNESS). '
                         'If one passes,retain all disabled controls and block only the failed action. '
                         'No seed,cut,output or SLO changes after observing performance.',
        scope='Twenty-one paired serving starts. Five fresh-source full400 proofs for unchanged control families. '
              'Those inference families already passed N1/N8/N32/400 before this metadata/disabled-dispatch source change. '
              'New scoped actions require their separate strictN1/N8,mixed32,guarded32 and400 sequence. '
              'Control lane4/5 waits for the old54 to finish. Lane6/7 waits for independent teachers. '
              'Serving0/1 waits for all proof outcomes and the original54. Natural400 safety is not inferred from this fixed16 test.'))
    print('SCOPED_COMPILER_N3_PLAN_FROZEN',len(jobs),'serving starts',flush=True)


def proof(lane):
    from .qualification import audit
    from .decode_cap_discovery import full_native_state
    from .quality_gate import compare as quality
    from .compute_throttle_qualification import no_mixed
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    markers=plan.get('control_wait_markers',[
        ['request_registration_controlled_bridge_complete_v1.json'],['scoped_compile_teachers_complete_v1.json']])[lane]
    while not all((ROOT/'state'/marker).exists() for marker in markers):time.sleep(20)
    for arm in plan['proof_lanes'][lane]:
        item=plan['qualification'][arm];target=ROOT/'state'/item['gate'];assert not target.exists()
        name=item['job'];results=[]
        try:
            assert execute(name),name
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            state=audit(name) if cfg.get('axis_kind') else full_native_state(name,False)
            assert state['passed']
            if cfg.get('axis_kind'):
                assert state['decode_full36_observation_complete']
                assert all(r['status']=='PASS' for r in state['imported_prefix_KV'])
            score=quality(name);assert score['passed']
            if cfg.get('defer_D_request_registration'):
                assert json.loads((ROOT/'raw'/name/'D_request_registration_deferral.json').read_text())['passed']
            rule=no_mixed(name) if cfg.get('d_compute_throttle') else None
            results=[dict(name=name,state=state,quality=score,compute_rule=rule)]
            value=dict(passed=True,full400_passed=True,results=results)
        except Exception as error:
            value=dict(passed=False,full400_passed=False,results=results,error=repr(error))
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        write(target,dict(value,time=time.time(),signature=registration_signature(cfg),
            role_signature=item['signature'],source_capsule_SHA256=plan['source_capsule_SHA256'],
            evidence='CORRECTNESS_ONLY' if value['passed'] else 'FAILED'))
        print('SCOPED_SOURCE_CONTROL_PROOF',arm,value['passed'],flush=True)
    prefix=plan.get('state_prefix','scoped_compile_confirmation')
    marker=f'scoped_compile_control_lane{lane}_complete_v1.json' if prefix=='scoped_compile_confirmation' else f'{prefix}_control_lane{lane}_complete_v1.json'
    write(ROOT/'state'/marker,dict(complete=True))


def run():
    from .qualification import audit
    from .metrics import summarize
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    prefix=plan.get('state_prefix','scoped_compile_confirmation')
    markers=[r['gate'] for r in plan['qualification'].values()]
    markers+=['request_registration_controlled_bridge_complete_v1.json']
    while not all((ROOT/'state'/m).exists() for m in markers):time.sleep(20)
    proofs={a:json.loads((ROOT/'state'/r['gate']).read_text()) for a,r in plan['qualification'].items()}
    any_candidate=any(proofs[a]['passed'] for a in ('LayerScoped','HybridScoped'))
    results=[]
    for cell in plan['cells']:
        for name in (n for n in plan['performance_jobs'] if n in cell['jobs'].values()):
            arm=next(a for a,n in cell['jobs'].items() if n==name)
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());gate=proofs[arm]
            eligible=any_candidate and all(proofs[a]['passed'] for a in ('NativeStrong','NativeCommon','StockToken')) and gate['passed']
            if not eligible:
                result=dict(name=name,status='NOT_RUN',reason='BLOCKED_CORRECTNESS')
            else:
                assert gate['role_signature']==signature(cfg) and gate['source_capsule_SHA256']==plan['source_capsule_SHA256']
                if execute(name):
                    if cfg.get('axis_kind'):
                        state=audit(name);assert state['passed'] and state['decode_full36_observation_complete']
                    metric=summarize(name);assert metric['output_tokens']==256
                    result=dict(name=name,status='MEASURED_ONLINE',
                        **{f:metric[f] for f in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')})
                else:result=dict(name=name,status='FAILED')
            results.append(result)
            write(ROOT/'state'/(prefix+'_progress_v1.json'),dict(results=results))
            print('SCOPED_COMPILER_SERVING',result,flush=True)
    write(ROOT/'state'/(prefix+'_complete_v1.json'),dict(time=time.time(),complete=True,results=results,
        scope='Operational completion. A paired safety/scientific result requires CPU analysis and root review.'))


def analyze():
    from .confirmation_analysis import ratio
    from .final_confirmation_analysis import complete_metric
    from .final_report_tables import interval
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    prefix=plan.get('state_prefix','scoped_compile_confirmation')
    arms={};fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms',
        'SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second')
    terminal=ROOT/'state'/(prefix+'_complete_v1.json')
    stops={r['name']:r for r in json.loads(terminal.read_text())['results']} if terminal.exists() else {}
    for arm in plan['cells'][0]['jobs']:
        records=[]
        for cell in plan['cells']:
            name=cell['jobs'][arm];receipt=ROOT/'raw'/name/'dispatcher_result.json'
            if receipt.exists() and json.loads(receipt.read_text())['complete']:
                cache=ROOT/'analysis'/(name+'_metrics.json')
                records.append(complete_metric(json.loads(cache.read_text()) if cache.exists() else {},name))
        arms[arm]=dict(starts=len(records),records=records,
            means={f:float(np.mean([r[f] for r in records])) for f in fields} if records else {},
            maximum_client_ITL_ms=max((r['max_ITL_ms'] for r in records),default=None))
    comparisons={}
    for arm in ('Layer','Hybrid','LayerScoped','HybridScoped','StockToken'):
        tests={}
        for ref in ('NativeStrong','NativeCommon','StockToken')+((arm.removesuffix('Scoped'),) if arm.endswith('Scoped') else ()):
            if arm==ref or len(arms[arm]['records'])!=3 or len(arms[ref]['records'])!=3:continue
            a,b=arms[arm]['records'],arms[ref]['records']
            values={f:ratio([r[f] for r in a],[r[f] for r in b]) for f in fields}
            absolute=all(r['decode_SLO_attainment']>=.95 for r in a)
            tests[ref]=dict(ratios=values,absolute_decode_pass=absolute,
                decode_5percent_CI_pass=all(values[f]['CI95'][1]<=1.05 for f in ('ITL_p99_ms','TPOT_p95_ms')),
                TTFT_CI_win=values['TTFT_p95_ms']['CI95'][1]<1,
                goodput_CI_win=values['SLO_qualified_output_tokens_per_second']['CI95'][0]>1)
        comparisons[arm]=tests
    complete=terminal.exists()
    value=dict(time=time.time(),complete=complete,arms=arms,comparisons=comparisons,
        terminal_outcomes=stops,source_capsule_SHA256=plan['source_capsule_SHA256'],
        evidence='MEASURED_ONLINE_CONTROLLED_KERNEL_POLICY_ABLATION' if any(r['records'] for r in arms.values()) else 'NOT_RUN',
        scope='Same frozen I3/Output16/Cap1 conditions. All fresh paired starts use0/1 and the same source. '
              'Do not pair with old4/5 measurements. Correctness-only Natural400 does not establish Natural400 serving performance. '
              'Client metrics are not GPU-exclusive times. Keep every strict and safety failure.')
    write(ROOT/'analysis'/(prefix+'_results_v1.json'),value)
    lines=['# LongBench400: Prefill Norm과 Compiled Decode를 분리한 검증','',
        '상태: '+('독립 Start의 Terminal 결과를 확보했다.' if complete else '실행 중이다. 미완료 Arm을 성능 PASS로 표시하지 않는다.'),'',
        '## 변경과 대조군','',
        '실제 설치 소스의 CUDA Platform은 Compile 여부에 따라 IR Norm 우선순위를 바꾼다.',
        'Mode0는 `vllm_c,native`를 사용한다. 기본 Inductor Compile은 `native`를 사용한다.',
        '두 설정은 같은 BF16 수학을 구현하지만 모든 Tensor Byte가 같지는 않았다.',
        '`custom_ops=all`만으로 IR Norm 우선순위가 바뀌지는 않았다.',
        '기본 D Compile의 엄격한 불일치와 Global CUDA 우선순위의 N1/N8 통과를 모두 보존한다.',
        '이 개입은 RMSNorm과 Fused-add RMSNorm의 Provider Bundle을 함께 바꾼다.',
        '두 Operator 각각의 수치 기여율을 별도로 분해한 것은 아니다.','',
        ('모든 D Prefill은 원래 CUDA Norm을 사용해 Eager로 실행한다.' if plan.get('eager_all_prefill') else
         'Scoped 경로는 Eager Partial-prefill 안에서만 `vllm_c,native`를 선택한다.'),
        ('Decode-only Full36은 원래 Native Compiled Model을 유지한다.' if plan.get('eager_all_prefill') else
         'Full36 Suffix와 Decode의 원래 Compiled Model은 `native`를 유지한다.'),
        'D의 Custom-op 설정은 `all`이다. P의 기존 Mode0 경로는 바꾸지 않는다.',
        'Tensor, Sampling Token이나 Client Timestamp를 덮어쓰지 않는다.',
        '원래 Full36 Single-GPU Teacher도 같은 Stage별 Provider 계약으로 독립 실행했다.',
        'Teacher에는 Partition Adapter, NIXL과 수입 KV가 없다.',
        'Teacher의 Warm State를 평가 Request로 세었던 CPU Assertion은 수정했다.',
        '평가 ID0–7과36개 KV Plane을 검사한다. Warm State와 실패 Controller Log는 삭제하지 않았다.','',
        '## 동결한 성능 계약','',
        'I3의 원문 Prompt16개, Mean Input28296 Tokens, QPS1.253873을 유지한다.',
        'Arrival Seed는2026112551/2552/2553이다. 각 Arm은 독립 Server Start3회다.',
        '모든 새 Paired 성능은 GPU0/1을 사용한다. 원래 GPU4/5 결과와 CI를 합치지 않는다.',
        '모델은 Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`이다.',
        'BI0, Original BF16, Exact Causal Attention과 Full36 Decode를 유지한다.',
        'B8192/S1024, KV42GiB/Block128, Prefix Cache Disabled를 유지한다.',
        'Output16은 `ignore_eos=True`인 Mechanism Control이다. Natural EOS 결과가 아니다.',
        'Layer8, Rows-matched Stock Token8, Hybrid4+정확한 L/8을 사용한다.',
        '각 Offload의 Logical P/D Rows는12676608/3621888이다.',
        'D Cap1과 Source Credit8을 모든 Offload Arm에 유지한다.',
        'NativeStrong은 원래 Scheduler를 유지한다. Common-policy Native의 Cap은1이다.',
        '실제 Batch와 Padding은 JSON에 별도 기록한다. B/S를 실제 Batch로 해석하지 않는다.',
        'Source SHA256은 `'+plan['source_capsule_SHA256']+'`다.',
        'Dataset SHA256은 `6429dde7a7dd44156636e7f9c8f041d7b7d0b6f91c80bc2cce1697dc51e8c780`이다.',
        'TTFT5000ms, 모든 ITL34ms, TPOT33ms의 Frozen SLO를 바꾸지 않았다.','',
        '## 실측 결과','',
        '| Arm | 독립 Starts | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(Output Tokens/s) | Raw Output Tokens/s | 관측 Max ITL(ms) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,item in arms.items():
        vals=' | '.join(f'{item["means"][f]:.6f}' for f in fields) if item['starts'] else 'N/A | N/A | N/A | N/A | N/A'
        maximum=f'{item["maximum_client_ITL_ms"]:.6f}' if item['starts'] else 'N/A'
        lines.append(f'| {arm} | {item["starts"]} | {vals} | {maximum} |')
    lines += ['','## 실제 Batch와 실행 경로','',
        '아래 Max는 완료한 Start에서 관측한 최댓값이다. 고정 Batch 또는 평균 Batch가 아니다.',
        'D Cap1은 Admission 상한이다. P Batch와 Query Padding을 같은 값으로 고정하지 않는다.','',
        'Graph 열은 Engine 설정이다. 모든 Partial Prefill이 Graph로 실행됐다는 뜻은 아니다.','',
        '| Arm | 관측 P/D 최대 Request Batch | 관측 P/D 최대 Padded Query | P/D Compile Mode | D Graph 설정 |',
        '|---|---|---|---|---|']
    for arm,item in arms.items():
        records=item['records']
        if not records:continue
        batch='/'.join(str(max(r['actual_geometry'][role]['maximum_requests'] for r in records)) for role in ('prefill','decode'))
        padding='/'.join(str(max(r['actual_geometry'][role]['maximum_padded_queries'] for r in records)) for role in ('prefill','decode'))
        runtimes={role:json.loads((ROOT/'raw'/records[0]['name']/f'actual_runtime_{role}.json').read_text()) for role in ('prefill','decode')}
        mode='/'.join(str(runtimes[role]['compile_mode']) for role in ('prefill','decode'))
        lines.append(f'| {arm} | {batch} | {padding} | {mode} | {runtimes["decode"]["cudagraph_mode"]} |')
    lines+=['','## Paired 판정','',
        '| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] | Decode5% | 매 Start Absolute Decode≥95% |',
        '|---|---|---|---|---|---|---|']
    for arm,tests in comparisons.items():
        for ref,result in tests.items():
            lines.append(f'| {arm}/{ref} | '+
                ' | '.join(interval(result['ratios'][f]) for f in fields[:4])+
                f' | {result["decode_5percent_CI_pass"]} | {result["absolute_decode_pass"]} |')
    lines+=['','## 판정의 제한','',
        'NI는 NativeStrong과 정확한 Common-policy Native에 각각 적용한다.',
        '두 Native에 대한 ITL p99와 TPOT p95 CI 상한이 모두1.05 이하여야 한다.',
        'Absolute Decode Attainment≥95%도 모든 Start에서 별도로 요구한다.',
        '실패한 Gate를 반올림하거나 큰 Client Interval을 제거하지 않는다.',
        'CI는 Paired Start-level Empirical 95% CI다. N3의27개 Bootstrap 조합을 열거한다.',
        '다중 비교 보정은 하지 않았다. 48개 Request를48개 독립 Start로 사용하지 않는다.',
        'Within-axis 개입에는 D의 Compiler Dispatch와 Stage별 Kernel 정책이 함께 포함된다.',
        '후속 Admission, Batch와 Queue도 바뀔 수 있다. 이를 순수 GPU 시간의 인과 효과로 표현하지 않는다.',
        'GPU-exclusive Operator 비중과 SM/HBM은 이 Source에서 NOT_MEASURED다.',
        '이 결과를 Natural400 또는 다른 Cut·QPS의 성능으로 확대하지 않는다.',
        '원래54개와 등록 수정54개의 실패를 새 Source로 덮어쓰지 않는다.',
        'Online Dynamic Window나 Axis Selection은 이 검사에 없다.',
        '유한16개 Throughput을 Sustainable Capacity로 표현하지 않는다.','',
        '- [동결한 설계](dense_ar_longbench400_2d_20261121/configs/'+PLAN+')',
        '- [전체 결과와 CI](dense_ar_longbench400_2d_20261121/analysis/'+prefix+'_results_v1.json)',
        '- [기존 수치 불일치와 Negative Control](dense_ar_longbench400_decode_compile_residual.md)','']
    if complete and not any(r['records'] for r in arms.values()):
        lines[2:2]=['성능 실행은0회다. 표의 N/A는 실제 지연 측정값이 아니다.',
            '정합성 실패 이후 첫 성능 실행 전에 계획 전체를 명시적으로 대체했다.',
            '원래21개 조건과 NOT_RUN 사유는 Terminal Outcome에 보존한다.','']
    (REPO/'poc_flashvep/reports'/plan.get('report_name','dense_ar_longbench400_scoped_decode_compile_results.md')).write_text('\n'.join(lines),encoding='utf-8')
    print('SCOPED_COMPILER_ANALYSIS',complete,{a:r['starts'] for a,r in arms.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('build','proof','run','analyze'))
    p.add_argument('--lane',type=int,choices=(0,1),default=0);args=p.parse_args()
    if args.action=='build':build()
    elif args.action=='proof':proof(args.lane)
    elif args.action=='run':run()
    else:analyze()
