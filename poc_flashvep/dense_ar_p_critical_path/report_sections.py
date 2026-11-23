"""Answer the requested questions from measured family results."""
import json
from .common import ROOT


def load(relative,default=None):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else default


def sections(table,estimate,number):
    verdict=load('analysis/scientific_verdict.json',{})
    main=load('analysis/c3_credit_confirmation_analysis.json',dict(comparisons=[],starts=[]))
    geometry=load('analysis/p_geometry_confirmation_analysis.json',dict(comparisons=[]))
    causal=load('analysis/p_critical_causal_summary.json',dict(comparisons=[]))
    arrivals=load('analysis/arrival_trace_audit.json',dict(traces=[]))
    serial=load('analysis/serial_serving_operator_analysis.json',dict(comparisons=[]))
    historical=load('analysis/historical_stage_recalculation.json',dict(comparisons=[],new_R0_bridges=[]))
    stress=load('analysis/p_output_stress_analysis.json',dict(cells=[]))
    cap4=load('analysis/p_geometry_D_cap4_discovery_analysis.json')
    cache=load('analysis/compiler_cache_regime_causal_audit.json')
    poc=load('attention_poc/analysis/attention_head_poc_analysis.json')
    warm=load('analysis/coordinator_hygiene_warm_regime_confirmation_analysis.json')
    output_cap=load('analysis/output192_D_cap_intervention_analysis.json')
    realized_admission=load('analysis/output192_D_admission_realization.json')
    gpu_budget=load('analysis/gpu_budget_accounting.json')
    outcomes=verdict.get('outcomes',[])
    def comparison(data,left,right,mode='EqualReplay'):
        return next((row for row in data['comparisons'] if row['left']==left and row['right']==right and row['mode']==mode),None)
    original=comparison(main,'OriginalLayer','StockToken')
    retirement=comparison(main,'OptimizedLayer','OriginalLayer')
    balance=comparison(geometry,'OptimizedLayer','OriginalLayer')
    host=next((row for row in verdict.get('additive_mean_host_comparisons',[]) if
        row['family']=='Credit Retirement' and row['mode']=='EqualReplay'),None)
    measured_N3=[row for row in outcomes if row['paired_starts']>=3]
    summary=['## 현재 과학적 판정','',
        f'확인된 Serving GO Cell은 {sum(row["serving_GO"] for row in measured_N3)}/{len(measured_N3)}개입니다.',
        'Serving GO는 Strong Stock 대비 TTFT·SLO-goodput 우위와 품질·Decode Gate를 함께 요구합니다.',
        'P GPU·Queue의 원인은 PARTIAL_ROOT_CAUSE_IDENTIFICATION입니다. 과거1.76초의 완전한 인과 분해는 아닙니다.',
        'Dynamic Admission과 미래 Safe Window는 미검증입니다. 결과가 없는 방법을 성공으로 포함하지 않습니다.','',
        '## 질문별 실측 답변','',
        '### Q1. P 측 추가 지연 중 GPU와 Queue·Host의 비용은 얼마인가?','',
        '과거 약1.76초의 Host 차이는 완전한 GPU·Host 인과 분해가 아닙니다.',
        '새 Equal-output N3에서 다음 가법 평균 차이를 측정했습니다.']
    old=next((row for row in historical['comparisons'] if row['left']=='Layer'),None)
    if old and 'P_path_delta_ms' in old:
        summary+=['',table(['Historical Raw 재계산 Layer−Stock','Mean 차이 ms [Start-level95% CI]'],[
            ['Arrival→P-ready',estimate(old['P_path_delta_ms'])],
            ['P-ready→Client First Token',estimate(old['post_P_path_delta_ms'])],
            ['Mean TTFT',estimate(old['mean_host_delta_ms']['TTFT_ms'])]]),
            '위 Historical N3는 원래9개 Raw 실행의 재계산입니다. 새 GPU N3 실행으로 표시하지 않습니다.','']
    if host:
        summary+=['',table(['Layer−Strong Stock','Mean 차이 ms [Start-level95% CI]'],[
            ['Arrival→P-ready',estimate(host['P_path_delta_ms'])],
            ['P-ready→Client First Token',estimate(host['post_P_path_delta_ms'])],
            ['Mean TTFT',estimate(host['mean_TTFT_delta_ms'])]])]
    summary+=['P-ready 이전의 손해가 이후 경로의 절약보다 컸습니다.',
        '이 표는 TTFT p95나 GPU-exclusive Time의 분해가 아닙니다.','']
    kernel_rows=[]
    for row in causal['comparisons']:
        if row['right']!='StockToken':continue
        tests=row.get('tests',{})
        kernel_rows.append([row['left'],row['qualified_paired_starts'],
            estimate(tests.get('P_kernel_resource_demand_ms_per_request',{}).get('delta_ms')),
            estimate(tests.get('Arrival_to_P_ready_mean_ms',{}).get('delta_ms'))])
    operator_rows=[]
    for row in causal['comparisons']:
        if row['left']!='Layer' or row['right']!='StockToken':continue
        for category in ('GEMM','ATTENTION','RMSNORM','FUSED_RMSNORM_RESIDUAL','ROPE','UNKNOWN'):
            operator_rows.append([category,estimate(row['operator_resource_demand_delta_ms_per_request'].get(category))])
    summary+=[table(['Axis−Stock','CPC-qualified Pair 수','Kernel Union/N Δms/request','Arrival→P-ready Δms'],kernel_rows),
        table(['P Kernel Category','Layer−Stock Δms/request [Start-level95% CI]'],operator_rows),
        'Kernel Union/N은 Resource Demand입니다. 대기 중 앞선 Request의 Kernel과 겹친 시간과 다릅니다.',
        '이 비교는 Best Layer2와 Stock WorkCut8입니다. 같은 P/D GEMM Rows 비교가 아닙니다.',
        '따라서 이 표의 GEMM 차이를 같은 Rows의 Shape 효율 차이로 단정하지 않습니다.',
        '현재 Best-vs-best에서 Attention만이 유일한 추가 비용은 아닙니다.',
        '관측된 GPU 계산 증가와 Queue 누적은 Compute-only 또는 Host-only 설명을 구분하는 근거입니다.',
        '그러나 두 원인의 독점적인 인과 비중은 ROOT_CAUSE_NOT_FULLY_IDENTIFIED입니다.','',
        '### Q2. 실제로 제거 가능한 비용은 무엇인가?','',
        '필요한 ACK와 GPU Fence 이후에도 유지된 Logical Source Credit은 조기에 반환할 수 있었습니다.',
        '별도 진단에서 Evidence 파일 쓰기와 Coordinator Cyclic GC가 Token 전달을 지연했습니다.',
        '이 진단은 원래1.85초 Client Tail 전체의 원인을 확정하지 않습니다.',
        '같은 GEMM Rows의 P Attention Pair 차이는 Host Scheduling만으로 제거할 수 없습니다.','',
        '### Q3. Intervention이 P 경로를 얼마나 줄였는가?','']
    if retirement:
        summary+=['Safe-retirement A/B는 Credit 상한과 D 정책을 유지했습니다.',
            table(['Metric','Optimized/Original 또는 차이 [95% CI]'],[
                ['Layer P 경로 Ratio',estimate(retirement['P_critical_path_ratio'])],
                ['Layer P 경로 Δms',estimate(retirement['P_critical_path_delta_ms'])],
                ['Layer TTFT p95 Ratio',estimate(retirement['ratios']['TTFT_p95_ms'])]])]
    if balance:
        summary+=['Last2→Last8은 실제 P 작업을 D로 옮긴 별도 Geometry Intervention입니다.',
            table(['Metric','Last8/Last2 [95% CI]'],[
                ['P 경로 Ratio',estimate(balance['P_critical_path_ratio'])],
                ['TTFT p95 Ratio',estimate(balance['ratios']['TTFT_p95_ms'])]])]
    iso_rows=[]
    for row in serial.get('comparisons',[]):
        if row['family']!='SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL':continue
        fields=row['fields']
        iso_rows.append([row['left']+'/'+row['right'],row['qualified_paired_starts'],
            estimate(fields['P_arrival_to_ready_mean_GPU_kernel_union_ms']['delta_ms']),
            estimate(fields['D_submit_to_first_worker_mean_GPU_kernel_union_ms']['delta_ms']),
            estimate(fields['observed_kernel_P_plus_D_first_ms']['delta_ms']),
            [value['D_rows_relative_error'] for value in row['work']]])
    summary+=['P 경로가 줄었다는 사실만으로 전체 Serving 개선을 판정하지 않습니다.','',
        '### Q4. D 측 Layer 이점은 유지되는가?','',
        'Serial 실제 Serving API에서 P와 D-first Kernel을 각각 계측했습니다.',
        'D-first에는 Continuation과 첫 출력 처리가 포함됩니다. Suffix-only 비용은 아닙니다.',
        'Best Layer2와 Stock WorkCut8은 D 작업량이 다릅니다.',
        '별도의 Last8/Stock8/Hybrid Iso-work Control은 실제 Rows와 CPC Gate를 따로 제공합니다.',
        table(['Serial 비교','Qualified Starts','P Kernel Δms','D-first Kernel Δms','P+D-first Kernel Δms','D Rows 상대 오차'],iso_rows),
        '위 표의 N1은 통계적으로 확정한 우위가 아닙니다.32개 Request를32개 독립 Start로 취급하지 않습니다.',
        'Natural C3의 D Operator 비중은 Serial 결과로 대체하지 않습니다.','',
        '### Q5. Strong Stock보다 C3 TTFT와 Goodput이 좋아졌는가?','']
    outcome_rows=[[row['family'],row['mode'],row['label'],estimate(row['TTFT_ratio']),estimate(row['goodput_ratio']),
        row['decode_passed'],row['decision'],row['identical_per_request_output_work']] for row in outcomes]
    long_output_rows=[];long_output_safe=[]
    for row in stress['cells']:
        if row['mode']!='Fixed192':continue
        gate=row['decode_gate']
        long_output_rows.append([row['label'],row['independent_starts'],estimate(row['fields']['ITL_p99_ms']),
            estimate(row['fields']['TPOT_p95_ms']),gate.get('relative_5percent_pass'),
            gate.get('absolute_decode_attainment_95percent_pass'),gate.get('per_start_max_ITL_ms')])
        if row['independent_starts']>=3 and row['quality_passed_all_starts'] and gate.get('qualified'):
            long_output_safe.append(row['label'])
    summary+=[table(['Family','Workload','Arm/Stock','TTFT p95 Ratio [CI]','SLO-goodput Ratio [CI]',
        'Decode Gate','Cell 판정','동일 Output Work'],outcome_rows),
        'Natural EOS에서 TTFT 우위와 Goodput 열세가 함께 나타날 수 있습니다.',
        '해당 TTFT 우위를 삭제하지 않습니다. 서로 다른 Output Work의 효과를 Compute Speedup으로 표현하지 않습니다.',
        'Equal-output Replay와 Natural EOS를 별도로 판정합니다.','',
        '### Q6. 긴 Output에서 어떤 Cut이 Decode를 보호하는가?','',
        'Fixed16/64/128은 각각 N1입니다. Fixed192는 사전 지정한 N3입니다.',
        '실측된 두 Native-relative Gate와 Absolute95% Gate를 모두 확인합니다.',
        'N1 Point가 안전해 보여도 반복 검증된 보편적 Cut으로 제시하지 않습니다.',
        table(['Fixed192 Arm','Starts','ITL p99 ms [CI]','TPOT p95 ms [CI]','Relative5%','Absolute95%','Start별 Max ITL ms'],long_output_rows),
        '현재 Fixed192 N3 Decode Gate 통과 Arm: '+(', '.join(long_output_safe) if long_output_safe else '아직 확인된 Arm이 없습니다.')+'.',
        '이 목록은 QPS2와 해당 실행 정책에 한정됩니다. TTFT·Goodput 최적 Cut 목록이 아닙니다.',
        'ITL p99가 통과해도 큰 Max ITL이 남을 수 있습니다. Max ITL과 모든-ITL 통과율을 함께 보존합니다.','',
        '### Q7. 남은 실행 한계는 무엇인가?','',
        'P Attention 증가, D Residency, Handoff와 Hybrid DAG 의존성이 남습니다.',
        'P RMSNorm Provider Probe는 Strict KV·Hidden Gate에 실패했습니다. 성능 Primary에는 포함하지 않습니다.',
        '같은 Probe의 Greedy8/8 일치는 State 불일치를 없애지 않습니다.',
        '공유 Source 경로의 Stock Compiler Counterfactual은 기존 Strict 실패와 별도로 기록합니다.',
        '새 Online Geometry 확인은 기존 Best Last2와 Cost 기반 Last8 후보에 한정합니다.',
        'Last4·Last12의 전체 QPS 재탐색은 NOT_RUN입니다. 새 Input×Output×Cut 전수 Map을 만들었다고 주장하지 않습니다.',
        '추가 Low·Knee·High QPS 확인은 유망한 Serving 후보의 후속 조건입니다. 해당 개선 Gate를 통과하지 못하면 NOT_RUN으로 남깁니다.',
        '미래 Decode-safe Window는 검증하지 못했습니다. Active-decode Dynamic Insertion은 NOT_RUN입니다.',
        'Attention-head Offloading PoC는 별도 Operator-parallel 실험입니다. 기존 Serving 성공을 대체하지 않습니다.','']
    if cap4:
        summary+=['## 같은 Geometry의 D Admission 개입','',
            'D Cap2→4는 Source Credit과 Geometry를 유지한 별도 One-factor Discovery입니다.',
            table(['Candidate/동일 Axis Cap2','TTFT p95 Ratio','Point Decode Screen','Absolute Decode 통과율','선정'],[
                [row['candidate']+'/'+row['control'],number(row['TTFT_ratio'],6),row['point_decode_pass'],
                    number(row['absolute_decode_attainment'],6),row['selected']] for row in cap4['screens']]),
            '각 Point는 N1입니다. CI 기반 Decode 보호나 우위로 표시하지 않습니다.',
            '사전 Screen을 통과하지 못한 후보는 N3로 확장하지 않습니다.9개 Discovery Arm의 결과는 모두 보존합니다.','']
    if cache:
        summary+=['## Compiler Cache와 정합성 반증','',
            'Source 경로만 고정한 검사는 공식 Cache Root를 통일하지 못했습니다.',
            'NIXL Port를 고정하자 P와 D의 공식 Cache Root는 각각 일치했습니다.',
            '그러나 최초 컴파일 실행과 이후 캐시 재사용 실행의 KV·Hidden은 일치하지 않았습니다.',
            table(['기존 Raw 비교','사전 계획','Requests','KV Exact','Hidden Exact','Greedy Exact','Strict PASS'],[
                [row['control']+'/'+row['candidate'],row['preplanned'],row['requests'],row['KV_exact'],
                    row['hidden_exact'],row['greedy_exact'],row['passed']] for row in cache['strict_comparisons']]),
            '뒤의 두 Warm 실행 비교는 사후 CPU 진단입니다. 기존 실패 Gate를 이 비교로 덮어쓰지 않습니다.']
        warm_strict=cache.get('prospective_warm_regime')
        if warm_strict:
            summary+=['사전에 계획한 새 Warm-only3회 검사의 Strict PASS: '+str(warm_strict['passed'])+'.',
                '이는 CORRECTNESS_ONLY입니다. 책임 Operator나 N3 Serving 이득을 단독으로 증명하지 않습니다.']
        summary+=['새 Runtime은 별도의8→32→400 Gate와 실제 Warm 실행 경로 검사를 요구합니다.',
            '기존 V6 실패와 미실행 성능 Cell은 그대로 유지합니다.','']
    if poc:
        summary+=['## 추가 Attention-head Offloading PoC','',
            '실제 Q/K/V와 Paged KV를 사용한 독립 Benchmark Process3개를 측정했습니다.',
            'D에는 Active Decode가 없습니다. 전체 모델과 Serving KV Budget의 공존 비용은 재생하지 않았습니다.',
            table(['Tensor 비교','관측 수','Exact 불일치 수','최대 절대 오차'],[
                [key,row['observations'],row['nonexact_observations'],row['maximum_absolute_difference']]
                for key,row in poc.get('correctness_summary',{}).items()]),
            table(['Input','Layer 호출 수','Remote 경로','Remote KV Heads','Remote/LocalFull Wall p50 [CI]'],[
                [row['input_tokens'],row['selected_layers'],row['remote_mode'],str(row['remote_KV_heads'])+'/8',
                    estimate(row['remote_over_local_full_p50'])] for row in poc['cells'] if row['confirmed_isolated_cost_win']]),
            '위 CI의 N은 Benchmark Process 수입니다. 독립 Serving Start 수가 아닙니다.',
            '표의 Layer 수는 Attention 호출 열입니다. 전체 Transformer Layer 실행시간이 아닙니다.',
            'GEMM·FFN·다음 Hidden 준비·Native Decode 보호는 포함하지 않습니다.',
            '따라서 약5%의 고립 Attention 비용 이득을 P 전체 시간이나 Online TTFT 개선으로 확대하지 않습니다.',
            '전체 모델 Greedy·Task 품질·Online SLO-goodput은 NOT_RUN입니다.','']
    warm_plan=load('configs/common_warm_regime_qualification_v9.json')
    if warm_plan:
        qualification=[]
        for item in warm_plan['items']:
            proof=load('state/'+item['gate'],{})
            qualification.append([item['label'],proof.get('passed',False),proof.get('full400_passed',False),
                proof.get('quality_passed',False),proof.get('error','없음')])
        summary+=['## 새 Common Warm C3 확인','',
            '공식 Graph와 AOT Cache를 재사용한 실행을 별도 Family로 확인합니다.',
            '전체 Backbone과 Partial Prefix·Interval의 모든 Cache Region을 검사합니다.',
            'V8의 단일 Cache Region Reader 오류는 CPU에서 수정했습니다.',
            'V9는 완료된 GPU 결과의 Config·Source·Receipt Hash를 검사한 뒤 재사용했습니다.',
            '기존 Reader 실패와 Cold/Warm State 불일치는 삭제하지 않았습니다.',
            table(['Arm','구성 Gate','Natural400','품질 Gate','오류'],qualification),
            'Strict Control의 Controlled Pipeline과 실제 C3 Pipeline은 다릅니다.',
            'V10은 실제 C3 Source Window와 Pipeline으로 Cache를 먼저 준비합니다.',
            'Priming8은 정합성 검사입니다. 추가 독립 성능 Start가 아닙니다.',
            '세 독립 Start는 각각 GPU0/1, GPU2/3, GPU6/7을 사용합니다.',
            '각 Paired 비교는 같은 Pair와 Arrival Trace를 사용합니다.',
            '공유 Host와 병렬 Pair의 간섭을 제거했다고 주장하지 않습니다.',
            '이 Family의 Optimized Label은 Partition Geometry를 뜻합니다. Early Credit Retirement가 아닙니다.',
            '앞선 Credit·Geometry Family와 Start를 합치지 않습니다.',
            'Heldout192 Request는 앞선 확인에도 사용했습니다. 새로운 Request 일반화 검사는 아닙니다.',
            '서로 다른 Arrival Seed의 Family 차이만으로 Coordinator 개선율을 추정하지 않습니다.']
        stop=load('state/coordinator_hygiene_warm_regime_not_run_v10.json')
        if stop:
            summary+=['기록된 미실행 범위: '+str(stop.get('modes','전체'))+'.',
                '미실행 이유: '+str(stop.get('reason','기록 파일 참조'))+'.']
        else:
            summary+=['Equal-output과 Natural EOS의 실행 여부는 Stage별 완료 기록으로 확인합니다.']
        if warm:
            summary+=[table(['Workload','Arm','독립 Starts','Runtime 경로 Gate','품질 Gate'],[
                [row['mode'],row['label'],row['independent_starts'],row.get('runtime_policy_passed_all_starts',False),
                    row['quality_passed_all_starts']] for row in warm['cells']])]
            gc_pair=comparison(warm,'OriginalHybrid','HybridGCControl')
            if gc_pair:
                summary+=['### Coordinator Cyclic GC의 독립 개입','',
                    '두 Hybrid는 같은 Worker Source·Geometry·Source Window·D 정책·Buffered I/O를 사용합니다.',
                    '측정 구간의 Coordinator Cyclic GC 유예 여부만 다릅니다. Worker GC와 Reference Counting은 바꾸지 않습니다.',
                    table(['GC 유예/Normal GC','Paired Ratio [Start-level95% CI]'],[
                        ['P Arrival→Ready',estimate(gc_pair['P_critical_path_ratio'])],
                        ['TTFT p95',estimate(gc_pair['ratios']['TTFT_p95_ms'])],
                        ['SLO-goodput',estimate(gc_pair['ratios']['SLO_qualified_output_tokens_per_second'])],
                        ['ITL p99',estimate(gc_pair['ratios']['ITL_p99_ms'])]]),
                    '이 비교는 Equal-output N3입니다. Strong Stock 대비 우위와는 별개입니다.',
                    '개입 전후의 실제 Output Work·Runtime 경로·품질·두 Native Decode Gate도 별도로 검사합니다.',
                    '이 결과를 원래1.85초 Tail 전체의 원인이나 Native·Stock의 동일 개선율로 확대하지 않습니다.','']
        summary+=['Natural Stage의 추가 실행은 남은 시간으로 결정합니다. 성능 승패를 선정 기준으로 사용하지 않습니다.',
            '절약한 시간은 기존 최대10시간 안에서 재배분했습니다. 최종 GPU·Git Closeout 시간은 별도로 확보합니다.','']
    if output_cap:
        summary+=['## 긴 Decode Residency의 D Admission 개입','',
            'Output192에서 D Cap2→4를 별도로 측정했습니다.',
            '이 Family의 OriginalLayer와 OptimizedLayer는 모두 Last8입니다. 각각 D Cap2와4를 사용합니다.',
            'OriginalHybrid와 OptimizedHybrid는 모두 Last2+Suffix3/17입니다. 각각 D Cap2와4를 사용합니다.',
            '같은 Axis의 Source Credit·Geometry·Compiler·Arrival·출력량을 유지합니다.',
            '이 실험은 기존 Runtime Family입니다. 새 Warm C3 결과와 합치지 않습니다.',
            table(['Arm','Starts','TTFT p95 ms','ITL p99 ms','TPOT p95 ms'],[
                [row['label'],row['independent_starts'],estimate(row['fields']['TTFT_p95_ms']),
                    estimate(row['fields']['ITL_p99_ms']),estimate(row['fields']['TPOT_p95_ms'])]
                for row in output_cap['cells']]),
            'N1은 반복 확인된 우위나 Native-relative5% 비열등성을 증명하지 않습니다.',
            '짧은 Replay의 Cap4 Screen 실패를 긴 Output의 결과로 대체하지 않습니다.','']
    if realized_admission:
        summary+=['### 설정한 D 상한과 실제 Decode Batch','',
            table(['Arm','설정 D Cap','실제 Scheduler 상한','Pure Decode 최대 Batch','Step 가중 평균 Batch',
                'Registration 유예 요청 수','조건부 Registration 대기 Mean ms'],[
                [row['label'],row['configured_D_cap'],row['actual_scheduler_max_running'],row['pure_decode_maximum_requests'],
                    number(row['pure_decode_step_weighted_mean_requests'],4),
                    row['registration_queue']['deferred_requests'] if row['registration_queue'] is not None else '정책 미사용',
                    number((row['registration_queue'].get('conditional_wait_ms') or {}).get('mean'))
                        if row['registration_queue'] is not None else 'NOT_APPLICABLE'] for row in realized_admission['rows']]),
            '설정 상한4를 실제 Decode Batch4로 취급하지 않습니다.',
            'Registration 정책은 Output을 시작한 Running Request가 남아 있으면 새 등록을 유예합니다.',
            '이 정책과 Decode-first Scheduling 아래에서 실제 Batch가 어떻게 형성됐는지 확인합니다.',
            '상한을 높여도 Batch가 커지지 않은 결과는 GPU Compute 포화의 인과적 증거가 아닙니다.',
            'Registration 대기는 Host 구간입니다. GPU Kernel 실행시간이나 Physical Idle로 표현하지 않습니다.',
            '등록 Cohort를 바꾸는 추가 GPU 개입은 NOT_RUN입니다. 미래 Decode-safe Window를 검증하지 않았습니다.','']
    if gpu_budget:
        summary+=['## GPU 사용 예산','',
            table(['항목','관측값'],[
                ['GPU Dispatch 기록 수',len(gpu_budget['rows'])],
                ['시작부터 마지막 GPU Dispatch 종료까지 hours',number(gpu_budget['observed_dispatch_span_hours'],4)],
                ['예약 GPU Process-hour 합계',number(gpu_budget['sum_reserved_GPU_process_hours'],4)],
                ['GPU별 Interval Union의 합계 GPU-hours',number(gpu_budget['reserved_dispatch_interval_union_GPU_hours'],4)],
                ['승인한 최대 Wall-clock hours',gpu_budget['approved_maximum_wall_hours']]]),
            '실패한 Startup과 정합성 검사도 포함합니다. GPU별 중첩 구간은 Union으로 한 번만 계산합니다.',
            '이 값은 예약한 GPU 집합과 Dispatcher 경과시간의 곱입니다. GPU-exclusive Kernel 시간이 아닙니다.',
            'Attention Capture는 실제 계산 GPU가1개여도2개를 예약한 비용으로 계산합니다.',
            'Burner 시간과 CPU-only 분석·Archive·Push 시간은 이 GPU Dispatch 예산에 포함하지 않습니다.','']
    arrival_rows=[[row['mode'],row['start'],row['offered_QPS'],number(row['realized_interarrival_rate'],4),
        number(row['minimum_interarrival_ms'],3),row['maximum_arrivals_in_preceding_window']['1.0'],
        row['identical_scheduled_trace_across_arms']] for row in arrivals['traces']]
    arrival_text='\n## Offered QPS와 순간 도착 집중\n\n'+table(
        ['Workload','Start','Offered QPS','실현된 간격 기준 requests/s','최소 간격 ms','직전1초 최대 도착 수','동일 Trace'],arrival_rows)+'''
QPS2는 평균 도착률입니다. 동시에2개를 넣는 Batch 설정이 아닙니다.
Poisson Trace에는 짧은 간격의 집중 도착이 있습니다.
느린 P Service는 이런 구간의 Queue를 더 오래 유지할 수 있습니다.
그러나 Arrival Burst만 바꾼 Intervention은 실행하지 않았습니다.
이 표와 Request별 CSV는 도착 집중의 인과적 기여율을 증명하지 않습니다.
'''
    runtime_rows=[]
    for row in main['starts']:
        if row['mode']!='EqualReplay':continue
        actual=row['actual_geometry'];runtime=row['actual_runtime']
        runtime_rows.append([row['label'],row['start'],row.get('axis_cut'),row.get('hybrid_suffix_fraction'),
            row['source_credit'],row['D_running_admission_cap'],actual['prefill']['maximum_requests'],
            actual['decode']['maximum_requests'],runtime['prefill']['compile_mode'],runtime['decode']['compile_mode'],
            runtime['prefill']['cudagraph_mode'],runtime['decode']['cudagraph_mode']])
    runtime_text='\n## 실제 실행 계약 — Primary Equal-output\n\n'+table(
        ['Arm','Start','Cut','Hybrid Fraction','Source Window','D Cap','P 최대 Request/Step','D 최대 Request/Step',
            'P Compile','D Compile','P Graph','D Graph'],runtime_rows)+'''
D 최대 Request/Step은 Pure Decode Batch만을 뜻하지 않습니다.
Prefill/Decode 분리 Histogram은 Per-start CSV의 Actual Geometry에 있습니다.
Worker Config의 Compile Mode는 Partial Prefix Segment의 내부 컴파일 전체를 설명하지 않습니다.
Partial Prefix/Suffix의 별도 Compiler Flag와 Source Capsule도 함께 확인합니다.
Source Window0은 무제한 Coordinator Window입니다. KV나 Hidden Pool이 무제한이라는 뜻은 아닙니다.
'''
    bridge_rows=[[row['label'],row['independent_starts'],row['QPS'],row['seed'],row['GPU_pair'],
        number(row['TTFT_p95_ms']),number(row['ITL_p99_ms']),number(row['TPOT_p95_ms'])]
        for row in historical.get('new_R0_bridges',[])]
    runtime_text+='\n## 새 R0 재현 Bridge\n\n'+table(
        ['Arm','Starts','QPS','Arrival Seed','GPU Pair','TTFT p95 ms','ITL p99 ms','TPOT p95 ms'],bridge_rows)+'''
R0는 N1 진단 Bridge입니다. Primary N3와 합치지 않습니다.
원래 Historical N3의 Raw 재계산과 새 GPU 실행을 구분합니다.
'''
    return dict(answers='\n'.join(summary),arrivals=arrival_text,runtime=runtime_text)
