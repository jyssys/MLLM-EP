"""Produce the campaign handoff only after the actual final evidence checks."""
import argparse
from collections import Counter
import json
import time

import numpy as np

from .common import ROOT,BASE,BRANCH,digest,preserve,write
from .resource_closeout import require_final_closeout
from .token_discovery_analysis import table

REPORT='mllm_epd_prefill_heavy_token_split_final_summary.md'


def idle_comparison(analysis):
    assert analysis['observer_gate_passed']
    result=[]
    for name,roi in analysis['ROIs'].items():
        for role,value in roi['roles'].items():
            result.append(dict(ROI=name,role=role,seconds=roi['duration_seconds'],
                busy_percent=100*value['kernel_busy_fraction'],idle_percent=100*value['kernel_idle_fraction'],
                simultaneous_E_D_idle_percent_total=100*roi['E_D_simultaneous_kernel_idle_fraction'],
                simultaneous_E_D_idle_percent_of_P_busy=100*roi['E_D_simultaneous_idle_fraction_of_P_busy'],
                evidence='TRACE_ANALYSIS_QUALIFIED_LOW_LOAD_REPLAY_N1',
                scope='Full cohort versus central scheduled-arrival p10-p90. Neither is validated steady-state safe capacity.'))
    assert {r['ROI'] for r in result}=={'full_cohort','central_arrival_window'}
    return result


def static_decode_audit(token,metric):
    by_run={r['run']:r for r in metric['results']};rows=[]
    selected=[r for r in token['runtime_decisions'] if r['method']=='Static10']
    assert len(selected)==4
    for cell in selected:
        starts=[r for r in token['starts'] if all(r[k]==cell[k] for k in ['method','subset','mode'])]
        assert len(starts)==3 and len({r['block'] for r in starts})==3
        gates=[by_run[r['run']]['absolute_joint_decode_95_gate'] for r in starts]
        rows.append(dict(subset=cell['subset'],mode=cell['mode'],independent_starts=3,
            actual_per_request_decode_AND_all_starts=all(gates),relative_decode_gate=cell['decode_relative_gate'],
            scoped_decode_protection=all(gates) and cell['decode_relative_gate'],
            runtime_decision=cell['runtime_decision'],full_task_quality='HOLD',
            scope='Static10 tested admission only. Not a future safe-window predictor or general unused-capacity proof.'))
    return rows


def decisions(static):
    return dict(motivation='MOTIVATION_CONFIRMED_RUNTIME_ONLY_QUALITY_HOLD',
        prefill_heavy='PREFILL_HEAVY_NOT_CONFIRMED',physical_D_idle='D_IDLE_CONFIRMED_LOW_LOAD_ONLY',
        physical_E_idle='E_IDLE_CONFIRMED_LOW_LOAD_ONLY',
        simultaneous_idle='E_D_SIMULTANEOUS_IDLE_CONFIRMED_LOW_LOAD_ONLY',
        tested_static10_decode_protection='PASS_IN_ALL_FOUR_C37_CELLS' if all(r['scoped_decode_protection'] for r in static) else 'NOT_ALL_CELLS_PASS',
        actual_future_decode_safe_budget='NOT_VALIDATED',
        encode_safe_headroom='ENCODE_SAFE_HEADROOM_NOT_CONFIRMED',
        unrelated_probe_full_protection='NOT_CONFIRMED_ON_TESTED_PROTOCOL',
        token_split_runtime='NO_GO_ON_TESTED_GRID',full_task_quality='HOLD',
        modality_aware_followup='HOLD',dual_side_followup='HOLD')


def markdown_table(columns,rows):
    return ['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+[
        '| '+' | '.join(str(value) for value in row)+' |' for row in rows]


def prose_paragraphs(lines):
    result=[];count=0
    for line in lines:
        if not line or line.startswith(('#','|','- ')):
            count=0;result.append(line);continue
        if count==2:result.append('');count=0
        result.append(line);count+=1
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metric-version',required=True)
    parser.add_argument('--archive-version',required=True);parser.add_argument('--figure-version',required=True)
    args=parser.parse_args();preserve();require_final_closeout()
    target=ROOT.parent/REPORT;assert not target.exists()
    inputs={}
    def load(relative):
        path=ROOT/relative;inputs[relative]=digest(path);return json.loads(path.read_text())
    metric=load(f'analysis/campaign_metric_audit_{args.metric_version}.json')
    assert not metric['incomplete'],'Actual serving cohorts are still unaudited'
    archive=load(f'analysis/archive_integrity_closeout_{args.archive_version}.json');assert archive['passed']
    figures=load(f'analysis/figure_regeneration_closeout_{args.figure_version}.json');assert figures['passed']
    integrity=load('analysis/dataset_integrity_audit_v0.json');assert integrity['passed']
    batch=load('analysis/batch_COUNTER_C46_audit_v0.json');assert batch['methods']==6 and batch['metric_recalculation']=='PASS'
    token=load('analysis/token_confirmation_C37_audit_v0.json');assert len(token['starts'])==48
    baseline=load('analysis/main100_baseline_confirmation_C14_v0.json')
    physical=load('analysis/ec_LOW_CUPTI_C25_CPC_q0.5_v0_idle_analysis.json')
    quality=load('analysis/task_quality_C37_audit_v0.json');assert quality['full_task_quality']=='HOLD'
    probe=load('analysis/interference_PROBE_C40R4_audit_v0.json')
    suffix=load('analysis/suffix_QUANTA_C41R3_report_audit_v0.json')
    policy=load('analysis/ec_INLINE_CAUSAL_C44_audit_v0.json');assert policy['cohorts']==24
    lengths=load('analysis/main100_result_figures_v0.json')
    parent=load('analysis/dataset_profile.json')
    assert parent['request_count']==parent['natural_output_count']==400 and parent['natural_output_complete']
    manifest=load('data/main100_manifest_template_v2.json');rows=manifest['requests'];assert len(rows)==100
    assert integrity['main100_manifest_SHA256']==digest(ROOT/'data/main100_manifest_template_v2.json')
    statistics={r['metric']:r for r in lengths['statistics']}
    idle=idle_comparison(physical);static=static_decode_audit(token,metric);verdict=decisions(static)
    assert all(r['scoped_decode_protection'] for r in static)
    assert all(not r['observed_HTTP_protection_gate'] for r in probe['decisions'])
    assert suffix['evidence']=='DIAGNOSTIC' and not suffix['observer_gate']
    table(ROOT/'analysis/final_summary_full_vs_central_idle_v0.csv',idle)
    table(ROOT/'analysis/final_summary_static10_decode_scope_v0.csv',static)
    table(ROOT/'analysis/final_summary_main100_statistics_v0.csv',lengths['statistics'])
    source_counts=Counter(row['source'] for row in rows)
    lines=['# MLLM EPD Prefill-heavy 및 Generic Exact Token Split: 최종 연구 요약','',
        '연구 판정은 `HOLD`다. 검사한 Token Split Runtime의 순이익은 `NO_GO_ON_TESTED_GRID`다.',
        '이는 모든 MLLM Token Split의 불가능성을 뜻하지 않는다. 현재 Model·Workload·Runtime·Admission 조건에 한정한다.',
        '실제 Online Runtime Trade-off와 저부하의 물리적 Idle은 관측했다. Prefill-heavy와 활용 가능한 미래 Safe Budget은 입증하지 못했다.',
        '전체 Task Quality는 HOLD다. 부분 정합성 통과와 성능 측정을 최종 품질 PASS로 바꾸지 않았다.','',
        '## 첫 페이지: 최종 질문 9개','',
        '| 질문 | 답 |', '|---|---|',
        '| Q1. Dataset은 무엇인가 | 원본 Image-only Main400을 보존했다. 사용자 변경 후 Main100으로 실험했다. DocVQA13·Mantis25·MP-DocVQA25·MMLongBench-Doc37개다. 실제 통계는 다음 표에 있다 |',
        '| Q2. DuetServe-style Trade-off가 있는가 | Runtime 수준에서 있다. QPS2/4의 N3에서 Native EPD는 Agg DP3보다 Decode Tail이 작았다. QPS4에서는 TTFT·Output Throughput이 불리했다. 전체 품질 검증은 HOLD다 |',
        '| Q3. E가 병목이 아닌 진짜 Prefill-heavy인가 | NOT_CONFIRMED. 고부하 Kernel 계측은 5% Observer Gate를 통과하지 못했다. 적격 저부하의 P Busy는14.0682%다 |',
        '| Q4. D Idle은 얼마나 있는가 | 적격 QPS0.5 중앙 구간의 물리적 D Idle은85.6284%다. 중앙 구간157.172988초와 전체192.291269초를 구분했다. Safe Window의 측정값은 아니다 |',
        '| Q5. Exact Token Split과 이미지 내부 Split은 가능한가 | 검사한 Image-span 내부·경계·다중 페이지 조건에서 Full28 Prefix KV·MRoPE·Suffix Continuation을 검증했다. 10% N1/N8은 통과했다. 25% N8은 KV 실패로 제외했다. 전체 Batch 불변성과 품질은 미확정이다 |',
        '| Q6. Native보다 C3에서 좋은가 | 동결한 Static10·Idle10은 두 Native와 비교한 전체 Serving Gate를 통과하지 못했다. Static10의 Decode Gate 통과와 TTFT·Goodput 악화는 별개다 |',
        '| Q7. 어느 Stage가 문제인가 | Main100 Replay의 Static10은 NativeCommon보다 P HTTP가 평균305.329ms 짧았다. D Coordinator 대기는971.956ms 증가했다. 총 평균 TTFT는933.063ms 증가했다. GPU-exclusive 분해는 아니다 |',
        '| Q8. MLLM-specific 관측은 무엇인가 | 원래 Encoder의2-image→5-image Geometry만 바꿔도 같은 Pixel의 Embedding과 KV가 달라졌다. 이미지 내부 Split 자체의 Attention 불가능성은 확인되지 않았다. HTTP/Embedding 소유권과 실제 Forward Geometry를 분리해야 한다 |',
        '| Q9. Modality-aware 후속 연구는 가치가 있는가 | HOLD. 먼저 원래 Vision의 Batch-dependent 수치 차이와 Task Quality, 고부하 병목·현재 상태 Admission을 해결해야 한다. E Idle 활용 자체는 선행연구가 있다 |','',
        '## Dataset과 실제 길이','',
        'Main400은 Calibration32와 이미지·문서 Component로 분리했다. Discovery200·Heldout200도 같은 Context를 공유하지 않는다.',
        'Main100은 기존 Main400의 부분집합이다. 원문·정답·이미지 순서·Prompt·Processor·MRoPE를 변경하지 않았다.',
        '통계량을 가깝게 유지하는 선정에는 이미 측정한 Natural Output Length를 사용했다. Latency·SLO·정확도 결과로 선정하지 않았다.',
        'Main100의 평균은 부모와 비슷하다. 모든 Percentile과 Tail이 같다고 주장하지 않는다.',
        '원본 Dataset은 Production 전체 분포의 표본이 아니다. Video와 강제4×4 Matrix는 사용하지 않았다.','']
    lines+=markdown_table(['Main100 Metric','Mean','Median','Min','Max','p95'],[
        [key]+[f'{statistics[key][field]:.6f}' for field in ['mean','median','minimum','maximum','p95']]
        for key in ['input_tokens','visual_tokens','text_tokens','special_tokens','natural_output_tokens','image_count']])
    lines+=['','표의 Natural Output은 원래 Single-GPU R0 측정이다. 각 C3 Start의 생성 작업량은 별도 Raw에 있다.',
        'Main100의 Total Input1,102,959·Visual1,095,976·Image References1,251·R0 Output5,269개를 보존했다.',
        'Main100 R0의 Cap Hit은0/100이다. 부모 Main400은1/400이며 최대2048 Tokens다.',
        'Main100 R0 최대 Output은609 Tokens다. 부모 평균 Output은52.59, Main100은52.69 Tokens다.',
        '부모 Main400의 평균 Input은11021.55 Tokens다. Main100은11029.59 Tokens다.',
        '문서 PDF는 모든 페이지를72DPI RGB로 처리했다. 선택된 Main400/100의 Context Truncation은0이다.',
        '초기 Feasibility의736개 중269개 부적격은 전체 후보 Pool의 제외율이 아니다.',
        '고정 LFS 크기와 다른 mi_phone.pdf 응답은 실패로 남겼다. 원문을 임의 대체하지 않았다.',
        'Main100의 Discovery50/Heldout50은 Source 비율이 같지 않다. Heldout은 Mantis22·MMLong28개다.',
        'Main100 전체를 완전 Heldout이라고 부르지 않았다. Calibration32는 Main과 Context-disjoint다.','',
        f'Main100 SHA256: `{integrity["main100_manifest_SHA256"]}`.',
        f'Main400 SHA256: `{integrity["main_manifest_SHA256"]}`.',
        f'Calibration SHA256: `{integrity["calibration_SHA256"]}`.','',
        'p10/p25/p75/p90/p95/p99·표준편차·Task별 길이·중복 검사는 원본 Dataset Profile과 Main100 CSV에 있다.',
        'Encoder Output Shape를 Token 수로 추정해서 실제 측정값으로 채우지 않았다. C46은 별도 집계 Output Shape 검사다.','',
        '## Baseline Motivation: Natural Main100','',
        '모든 방법은 같은 H1003개와 실제 Online Encoding·Continuous Batching을 사용했다.',
        'Agg DP3는 Full MLLM Replica3개다. E+PD DP2는 별도 E와 Colocated PD2개다. Native EPD는1E+1P+1D다.',
        '동일 Offered Arrival Trace를 사용했다. TTFT는 예정 Arrival부터 계산했다.',
        'QPS2/4는 독립 초기화3회다. 나머지0.5/1/3/6은 Discovery N1이다.','']
    bfields=['TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','TPOT_p95_ms',
        'total_input_output_tokens_per_second','raw_output_tokens_per_second','SLO_goodput','JOINT_PASS']
    btable=[]
    for qps in [2.,4.]:
        for method in ['agg','e_pd','native']:
            selected=[r for r in baseline['rows'] if r['method']==method and r['QPS']==qps];assert len(selected)==3
            btable.append([method,qps,3]+[f'{np.mean([r[k] for r in selected]):.3f}' for k in bfields])
    lines+=markdown_table(['Method','Offered QPS','Starts','TTFT p50 ms','TTFT p95 ms','Mean TBT ms','ITL p99 ms',
        'TPOT p95 ms/token','Input+Output tok/s','Output tok/s','Goodput tok/s','Joint %'],btable)
    lines+=['','Start별 Percentile의 평균이다. 요청을 합쳐 계산한 Percentile이 아니다.',
        '전체 QPS Grid·실측점·95% CI와 Actual Output Tokens는 main100_C14_* CSV/PNG/PDF에 있다.',
        'Finite-cohort+Drain 처리량을 Sustainable Capacity라고 부르지 않는다. DuetServe의 수치 재현도 아니다.',
        'Full Quality와 적격 고부하 GPU 계측은 통과하지 못했다. 이 표로 Prefill-heavy를 확정하지 않는다.','',
        '## 실제 Token Split: QPS2, Paired N3','',
        '두 Native 기준을 유지했다. NativeStrong은 Stock Compiled Full-model·원래 Admission이다.',
        'NativeCommon은 후보와 공통 Reduction Compiler 설정을 사용한다. 그 설정은 BI1이 아니다.',
        'Static10은 약10% Contiguous Suffix다. Idle10은 현재 P/D HTTP 상태로 선택하고 Native로 Fallback한다.',
        '추가 Suffix는 기존 D HTTP Stream이 없을 때만 실행했다. 미래의 정확한 EOS·Arrival·Window는 사용하지 않았다.',
        '이 보수적 Admission은 검증된 Dynamic Split-size·Safe-window Predictor가 아니다.',
        'Inline E Descriptor는 NativeStrong·Token에 이미 적용됐다. 후보만 On-demand Transport를 제거한 비교가 아니다.','']
    fields=['TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','TPOT_p95_ms',
        'total_input_output_tokens_per_second','raw_output_tokens_per_second','SLO_goodput','joint_percent']
    for subset in ['Main100','Heldout50']:
        lines+=['',f'### {subset}: Natural과 Replay를 분리한 Start-mean','']
        values=[]
        for mode in ['natural','replay']:
            for method in ['NativeStrong','NativeCommon','Static10','Idle10']:
                chosen=[r for r in token['starts'] if (r['subset'],r['mode'],r['method'])==(subset,mode,method)]
                assert len(chosen)==3
                values.append([mode,method]+[f'{np.mean([r[k] for r in chosen]):.3f}' for k in fields])
        lines+=markdown_table(['Output','Method','TTFT mean ms','TTFT p50 ms','TTFT p95 ms','Mean TBT ms','ITL p99 ms',
            'TPOT p95 ms/token','Total tok/s','Output tok/s','Goodput tok/s','Joint %'],values)
    lines+=['','동일 Replay 길이는 동일 Answer String을 보장하지 않는다. Natural의 출력 작업량 차이는 유지했다.',
        '최대 Client ITL·모든-ITL 요청 통과율·완료 req/s·8개 Failure Category의 요청/Token 수는 C37 원시 CSV에 있다.',
        'Main/Heldout·Natural/Replay를 합쳐 평균내지 않았다. 3개 Start Pair의27개 Empirical Bootstrap Resample을 사용했다.',
        'N3의 CI 해상도는 제한적이다. 요청과 Token은 독립 Server Start가 아니다.','']
    lines+=markdown_table(['Subset','Output','Static10 Relative Decode','Actual Decode AND, every Start','Runtime Decision'],[
        [r['subset'],r['mode'],r['relative_decode_gate'],r['actual_per_request_decode_AND_all_starts'],r['runtime_decision']] for r in static])
    lines+=['','Static10은 네 조건에서 NativeStrong·NativeCommon 각각에 대한 ITL p99·TPOT p95 상대 Gate를 통과했다.',
        '각 Start의 요청별 All-ITL AND TPOT 통과율도95% 이상이었다. 이 범위의 Decode 보호는 실제 관측 결과다.',
        '그러나 TTFT p50 Gate와 두 Native에 대한5% Serving 개선을 통과하지 못했다.',
        'Idle10도 전체 Serving Gate를 통과하지 못했다. Heldout Natural은 Relative Decode Gate도 실패했다.',
        '일부 기준에만 유리한 Cell을 지우지 않았다. Main Replay Idle10의 NativeStrong 대비 Goodput 개선도 원시 CI에 유지했다.',
        'NativeCommon까지 포함한 동결 전체 Gate는 실패했다.','',
        '## 왜 P 절감이 Serving 순이익으로 이어지지 않았는가','',
        '같은 요청·Arrival·Replay의 Static10−NativeCommon 평균 P HTTP 차이는−305.329ms였다.',
        '추가 D Coordinator 대기는+971.956ms였다. 전체 평균 TTFT는+933.063ms,95% CI[817.498,1020.626]ms였다.',
        'NativeStrong과의 평균 TTFT 차이는+780.854ms,95% CI[639.307,929.115]ms였다.',
        'Heldout Replay의 NativeCommon 대비 D 대기는+2790.985ms, 전체 TTFT는+2698.824ms였다.',
        '이는 중복 없는 Host 경계다. P/D GPU-exclusive Operator 시간이나 TTFT p95의 가법 분해가 아니다.',
        'Source Credit·Embedding Lifetime·Transfer ACK를 조기에 해제하지 않았다. HTTP 상태를 GPU Stream Quiescence로 해석하지 않았다.',
        'Queue를0으로 만드는 Counterfactual은 Queue Feedback과 Decode 경쟁을 재실행하지 않는다. OFFLINE_ORACLE로만 남겼다.','',
        '### 별도 One-factor Native EC Policy Intervention','',
        'C44는 같은 Common Native 초기화 안에서 E Descriptor의 Inline/On-demand만 변경했다.',
        'ABBA/BAAB/ABBA의 두 Period를 Start 안에서 평균낸 후 독립 N3로 비교했다. 전체24개 Cohort를 재계산했다.',
        'Inline/On-demand TTFT p95 비율은 Natural0.482137[0.473061,0.493121],Replay0.462210[0.428544,0.493717]였다.',
        '반면 ITL p99 비율은 Natural1.044801[1.010495,1.086879],Replay1.085194[1.067145,1.105415]였다.',
        '5% Relative Decode 보호는 통과하지 못했다. TTFT 개선으로 이를 상쇄하지 않았다.',
        'Natural의 E HTTP는 평균228.691ms 증가했고, E 완료 후 D Submit까지의 Host Chain은2336.622ms 감소했다.',
        '이는 EC 정책과 Queue Feedback의 총 인과 효과다. 2초 GPU Compute 감소나 특정 Operator 효과가 아니다.',
        '기존 Stock NativeStrong·Static Token은 이미 Inline이었다. 이 결과가 Token 회귀를 해결했다고 주장하지 않는다.','',
        '## 물리적 E/D Idle: Full과 Central 분모','',
        'C25는 같은 GPU4/5/6의 QPS0.5,Replay5269 Tokens,N1이다. Clean/Profile/Clean의 전체5% Gate를 통과했다.',
        'Full은 첫 예정 Arrival부터 마지막 Drain까지다. Central은 예정 Arrival p10–p90다.',
        'Profiler Capture 밖을 Idle로 채우지 않았다. Central을 지속 가능한 Steady State로 부르지 않았다.','']
    lines+=markdown_table(['ROI','Stage','Duration s','Kernel Busy %','Kernel Idle %','E&D Idle/ROI %','E&D Idle/P Busy %'],[
        [r['ROI'],r['role']]+[f'{r[k]:.6f}' for k in ['seconds','busy_percent','idle_percent',
            'simultaneous_E_D_idle_percent_total','simultaneous_E_D_idle_percent_of_P_busy']] for r in idle])
    lines+=['','Central E/D Gap Count는72439/816866개다. Mean은1.893466/0.164757ms, p99는0.243080/0.023008ms다.',
        '최대 Gap은 E6769.806614ms,D6501.268365ms다. 매우 짧은 Gap이 Count를 지배한다.',
        '100ms 이상 Gap은 E/D Idle Duration의99.339073%/98.922669%지만 Count의0.091111%/0.007100%다.',
        'Clock Anchor 오차는 최대 약94µs다. 짧은 GPU 간 선후관계는 안전 Admission 근거가 아니다.',
        'No Pending HTTP와 Idle의 동시 관측도 완전한 GPU Quiescence를 보장하지 않는다.',
        'TRUE_ENCODE_IDLE 등5개 인과 Category의 완전 배타적 분해와 종료 원인은 NOT_MEASURED다.',
        '고부하 C11/C13/C19/C27·문서 C39는 Observer Gate 실패로 DIAGNOSTIC이다.',
        'P Service Proximity와 독립 E/P Capacity는 NOT_MEASURED다. P Busy≥80%만으로 Prefill-heavy를 확정하지 않았다.',
        'SM Active·Tensor Core Activity·HBM Bandwidth는 NOT_MEASURED다. Kernel Busy나 nvidia-smi Utilization으로 대체하지 않았다.','',
        '## 실제 E/D Prefill Intervention과 메모리','',
        'C40은 원래 Online Main100과 함께 실제128/512-token Prefix Probe를 E 또는 D에서 실행했다.',
        'Natural·Replay 각각 독립 N3다. Probe Sample은 버렸다. 실제 요청 완료나 Serving Throughput 분자에 더하지 않았다.',
        '추가 실제 작업은613회·195200 Prefix Tokens다. 5465600 Token-layer Rows는28×Tokens의 STRUCTURAL_ACCOUNTING이다.',
        '이를 물리적 GEMM Rows 실측이나 유효 Offloaded Work로 부르지 않는다.',
        '8개 조건의 전체 보호 Gate는 모두 실패했다. E HTTP p95는 Encoder-exclusive GPU Service가 아니다.',
        '24개 Probe Cohort 중4개는 ITL p99≤34ms여도 요청별 Decode AND95%에 실패했다.',
        '24개 No-probe Control에서도4개가 Decode AND95%에 실패했다. 모든 Tail을 Probe의 GPU 간섭으로 단정하지 않는다.',
        'Probe 조건 순서는 고정이었다. 시간 변화·Baseline Variation과 개별 GPU 간섭의 기여율은 미확정이다.',
        '따라서 E-safe Headroom은 미확인이다. 이미 별도 C37에서 통과한 Static10 Decode 보호를 취소하는 결과도 아니다.',
        'E의 원래 Vision Weight는1,353,100,288bytes다. 추가 Full Model은16,584,333,312bytes다.',
        '추가 LLM Weight15,231,233,024bytes와 중복 Vision1,353,100,288bytes를 구분했다.',
        'Helper KV는8,587,837,440bytes다. Native D의42GiB와 같지 않다.',
        '세 Start의 Main100 후 GPU4 Free는19,053,608,960–21,605,842,944bytes였다.',
        'Shared Device Free를 E와 Helper Process에서 중복 합산하지 않았다. Allocator Reserved는 Weight나 실제 Activation이 아니다.',
        '이 실행에서 공존 적재는 가능했다. 최대 Encoder Batch Capacity·OOM 경계·ROI Activation Peak는 NOT_MEASURED다.','',
        '## Imported-prefix Suffix Cost와 Safe-fit 한계','',
        'C41은 실제 E/P/NIXL/D와 Prefix KV Import를 사용했다. Prefix 없는 Probe와 구분한다.',
        'Input3개×Suffix128/256/512/1024/2048의15개 조건에서 Strict Full28 KV·전체 Natural Output을 검사했다.',
        '고립 비용은 Client Concurrency1·실제 D Batch1이다. 총420개 Trial과 Query 작업량 보존을 확인했다.',
        '전체 Observer Gate는 실패했다. Clean Spread도 컸다. 모든 비용은 DIAGNOSTIC으로 유지했다.',
        'Profile Forward Enclosure p95 범위는28.933984–154.926367ms다. GPU-exclusive Kernel·DMA가 아니다.',
        'HTTP p95는 Clean Before63.221277–201.262310ms,After74.810553–224.264566ms다.',
        'Non-streaming1-token Completion Wall을 SSE First-token TTFT라고 부르지 않았다.',
        '미래 Physical Gap과 비용을 비교한 Safe-fit 그림은 OFFLINE_ORACLE이다. 실제 Admission 성공이나 Online Speedup이 아니다.',
        'E Prefix와 D Imported Suffix의 의존성이 다르다. E가 D보다 큰 안전 Quantum을 처리한다고 확정하지 않았다.','',
        '## 정합성·품질·실제 Batch의 한계','',
        '모든 Token을 첫 Output 전에 처리했다. Token 순서·MRoPE·Image Grid·Prefix KV·Full28 Decode를 유지했다.',
        '고립된 Text/Image 경계·이미지 내부·다중 이미지·45-page 조건에서 검사를 통과했다.',
        'BI0 Stock Compiled의 Full KV 차이는 Unsplit Teacher에도 나타났다. Token Split 고유 오류로 단정하지 않았다.',
        '공통 Reduction 설정에서 같은 Query Geometry의 Full KV를 검사했다. 이 설정이 모든 Batch의 Invariance를 보장하지는 않는다.',
        'Frozen25% N8은7/8 요청에서 KV 허용 오차를 실패했다. Output8/8 일치를 근거로 KV 실패를 PASS로 바꾸지 않았다.',
        'C38은 같은 Target2개 이미지에 원본 Companion3개를 더한 Encoder Envelope만 변경했다.',
        '같은 Pixel·Grid·LLM Chunk에서 Target Embedding Relative L2는0.104151/0.085062였다.',
        '후속 KV 최대 Relative L2는 약0.3223이었다. 같은 Geometry 반복은 Strict였다.',
        '원래 Vision을 수정하지 않았다. 이 차이를 정상 BF16 오차나 Task Quality 무해성으로 확정하지 않았다.',
        'MMLongBench-Doc 공식 외부 답변 추출은 별도 승인·API Key가 필요하다. 유료 API 호출은0회다.',
        'DocVQA Raw ANLS와 정답 포함 여부는 다르다. Mantis 파서 민감성과 Context-cluster 품질 CI도 보존했다.',
        'Main100 Mantis Static10의 NativeCommon 대비 정확도 차이는−4pp,95% CI[−11.5385,0]pp였다.',
        '이는−2pp 품질 비열등성을 통과하지 못한다. 부분집합의 Output 일치를 전체 품질 PASS로 일반화하지 않았다.',
        'C46/C46R1은 별도 N1 Counter CPC다. 실제 Prepared Query·Encoder Output Geometry를 기록했다.',
        '원래 Clean N3의 Actual Batch를 소급 채우지 않았다. Hook Gate 실패는 DIAGNOSTIC으로 유지했다.',
        '그 Counter 구간의 D 최대 Prepared Request 수는 Static10이1,NativeStrong이4였다.',
        '이는 보수적 Admission의 직렬화와 일치하는 별도 진단 관측이다. 과거 C37 Batch나 독립적인 인과 기여율로 사용하지 않았다.',
        'V1 Native/Token Runner와 V2 Agg/PD Replica Runner의 실제 API를 구분했다.',
        'C46의 V2 API 실패는 보존했다. 완료된4개 방법은 재실행하지 않고 미실행2개만 Observer를 수정한 새 실행으로 검사했다.','',
        '## 계약 차이와 미실행','',
        'C34/C37의 보수적 C3 검사는 Prefill-heavy와 실제 Safe Budget이 확정되기 전에 수행했다.',
        '이는 원래 Phase Gate 순서의 이탈이다. 사후에 원래 조건을 충족했다고 바꾸지 않았다.',
        'Active-decode 내부 위험 삽입·Three-way Split·Context Parallelism·Wavefront·Layer/Hybrid·Modality-aware Method는 NOT_RUN이다.',
        '복잡한 Predictor 없이 현재 HTTP 상태로 No-offload Fallback을 사용했다. 진정한 Dynamic Selector의 독립 이득은 미입증이다.',
        'Whole-request Deflection도 별도 N3와 Fresh50에서 검사했다. 두 Native 기준의 Decode·Serving Gate를 통과하지 못했다.',
        '이를 Kairos 재현으로 명명하지 않았다. Whole-request와 같은 요청의 Prefix/Suffix Split은 다른 구조다.',
        'Related-work 감사는 EPD·공식 vLLM·Dynamo·HydraInfer·EAServe·FluidPD·Kairos·DuetServe 등을 구분했다.',
        'MLLM Token Split 최초 또는 E Idle 활용 최초를 주장하지 않았다.',
        'Notion 결정은 읽은 Revision과 접근 시각으로 한정한다. 이후 모든 Edit를 확인했다고 주장하지 않았다.','',
        '## 재현성·실행 종료','',
        'Model·Processor는 Qwen/Qwen2.5-VL-7B-Instruct Revision cc594898137f460bfe9f0759e9844b3ce807cfb5다.',
        'Original BF16·Greedy·Natural EOS·원본 Vision/Processor/Attention·Full28·Prefix Cache Off를 유지했다.',
        'P/D B8192/S1024,KV42GiB,Block128이다. E는 별도 B114688,T2,MM Cache0과 Sync Scheduling이다.',
        'vLLM0.26.0+cu129·PyTorch2.11.0+cu129·NIXL1.4.1을 사용했다. E를 Compiled Vision이라고 추정하지 않았다.',
        '주요 비교는 같은 H100 SXM80GB GPU4/5/6 E/P/D다. UUID·NV18 Topology·실제 Runtime/Graph Flag는 각 Lease에 있다.',
        'Arrival Seed2026112402,Order Seed2026112404와 요청별 Prompt Hash를 저장했다.',
        f'기준 Source Commit은 {BASE}다. Worker는 Boot별 불변 Source ZIP/SHA256을 실행했다.',
        '분석 코드 수정이 실행 중 Worker를 Hot-patch하지 않았다. User Checkout·Index·기존 Dense Evidence Hash를 보존했다.',
        f'최종 Metric 감사는 {metric["cohorts"]}개 Cohort·{metric["requests"]}개 요청을 Timestamp에서 다시 계산했다.',
        'Full Task Quality·Observer Validity·Metric Integrity는 서로 다른 Gate다.',
        'Figure는 새 출력 폴더에서 재생성했다. 대응 CSV와 PNG RGBA Pixel을 비교했다.',
        'PDF는 Header/EOF와 Artifact 존재를 검사했다. 별도 PDF Pixel Renderer가 없어 Full PDF Render 검증을 주장하지 않았다.',
        '원시 Tensor는 원본 Checksum과 비교했다. 기존 Gzip은 Decompression Round-trip Hash를 검사했다.',
        'Boot ZIP CRC·Member Hash·기록된 Worker Source Hash를 검사했다. 초기 일부 Source ZIP 누락은 복원한 것처럼 채우지 않았다.',
        '초기 최종 감사 v0의 실패는 그대로 보존했다. Append-only Log4개는 옛 압축본의 정확한 Prefix와 현재 전체 Snapshot을 각각 확인했다.',
        '이 분류는 Byte 보존 관계다. 과거 파일 쓰기 Syscall이나 정확한 기록 시점을 증명하는 검사가 아니다.',
        'Worker3개는 Legacy source_SHA256 Map과3개 Member가 달랐다. 원래 Map은 수정하지 않았다.',
        '해당 Boot ZIP·전체 실행용 추출 파일·시작 Ledger는 독립적으로 일치했다. Legacy Map의 정확한 불일치 원인은 미확정이다.',
        '따라서 원래 Source Map 전체가 일치했다고 주장하지 않는다. 이3건은 검증된 Archive 근거와 Provenance 경고를 함께 사용해야 한다.',
        '자세한 차이와 역사적 Source47건의 누락은 mllm_epd_final_provenance_discrepancies_v1.md에 있다.',
        '대형 원문 이미지·모델 Weight·원본 Tensor는 로컬에 보존한다. GitHub에는 Source·구조화 Raw·표·Figure·Checksum과 압축 Evidence를 올린다.',
        'GPU Lease와 Controller가 종료된 후 승인 Burner를 GPU0–7에 복원했다. 소유권·CUDA Context·실제 사용률을 확인했다.',
        f'Research Branch는 `{BRANCH}`다. 최종 Push와 Remote SHA는 별도 Publication Receipt에 기록한다.',
        '현재 요약은 미래의 최종 Commit SHA를 추정해서 기재하지 않는다.','',
        '## Authoritative 자료 안내','',
        '- Dataset: mllm_prefill_heavy400_dataset_profile.md, Main100 통계 CSV, dataset_integrity_audit_v0.json.',
        '- Runtime: mllm_epd_runtime_correctness_report.md와 Common Compiled·MRoPE·Encoder Batch 후속 보고서.',
        '- Motivation: mllm_epd_main100_baseline_motivation_v1.md와 main100_C14_* CSV/PNG/PDF.',
        '- Token N3: mllm_epd_generic_token_split_online_serving.md, token_confirmation_C37_* Raw/CSV/CI.',
        '- Idle: mllm_epd_encoder_decoder_idle_final_summary.md, mllm_epd_idle_scope_and_required_figures_v2.md.',
        '- Probe: mllm_epd_dual_side_prefill_headroom_report.md와 Probe Failure·Reference Variation 보고서.',
        '- Suffix: mllm_epd_imported_kv_suffix_quantum_cost_v1.md. Safe-fit은 Offline Oracle이다.',
        '- Causal Policy: mllm_epd_native_ec_descriptor_causal_control_v1.md.',
        '- Actual Batch: mllm_epd_actual_batch_counter_validation_v1.md. 과거 Clean N3 Batch의 대체값이 아니다.',
        '- Contract 변경·Phase Gate 이탈: mllm_epd_campaign_contract_and_deviations_v1.md.',
        '- 최종 표·입력 Hash·판정: analysis/final_summary_*_v0. 최종 Publication Receipt와 Resource Closeout은 state/에 있다.','']
    lines+=['','## 요청별 SLO·작업량: 동일 C37 N3','',
        '아래 Pass는 각 Start 비율의 평균이다. Max ITL은 세 Start에서 실제 관측한 최대값이다.',
        'Max ITL은 평균 Max나 ITL p99와 다르다. Output Count는 Start0/1/2 순서로 각각 보존했다.','']
    slo=[]
    for subset in ['Main100','Heldout50']:
        for mode in ['natural','replay']:
            for method in ['NativeStrong','NativeCommon','Static10','Idle10']:
                chosen=sorted([r for r in token['starts'] if (r['subset'],r['mode'],r['method'])==(subset,mode,method)],key=lambda r:r['block'])
                assert len(chosen)==3
                cached=[load(f'analysis/{r["run"]}_metrics.json') for r in chosen]
                slo.append([subset,mode,method,chosen[0]['count'],
                    '/'.join(str(r['output_tokens']) for r in chosen),
                    *[f'{100*np.mean([r["gate_attainment"][key] for r in cached]):.3f}' for key in ['TTFT_PASS','ITL_PASS','TPOT_PASS','JOINT_PASS']],
                    f'{max(r["ITL_max_ms"] for r in chosen):.6f}'])
    lines+=markdown_table(['Subset','Output','Method','Requests/Start','Actual Output Tokens, Start0/1/2',
        'TTFT pass %','All-ITL pass %','TPOT pass %','Joint %','Max observed Client ITL ms'],slo)
    lines+=['','## Quality와 Resource 표의 미측정 범위','',
        '저부하의 Native N1 Trace·별도 Counter N1을 Main C37의 고부하 Resource Table에 소급하지 않았다.','']
    lines+=markdown_table(['C37 Method','Full Task Quality','Qualified E/P/D Busy at QPS2','Time-weighted P/D Queue','Old clean N3 Actual Batch'],
        [[method,'HOLD','NOT_MEASURED','NOT_MEASURED','NOT_MEASURED']
            for method in ['NativeStrong','NativeCommon','Static10','Idle10']])
    lines+=['','자원 관측이 없다는 뜻은 아니다. 적격 저부하 Trace·실제 HTTP Path·진단 Counter는 각각 별도 증거 수준으로 제공했다.','']
    target.write_text('\n'.join(prose_paragraphs(lines)))
    write(ROOT/'analysis/final_summary_evidence_v0.json',dict(time=time.time(),report=REPORT,
        report_SHA256=digest(target),inputs=inputs,decisions=verdict,source_counts=source_counts,
        scope='Final scientific handoff of the completed requested experiments, failures and non-executed unsafe stages. '
            'Scientific HOLD/NO_GO is not a claim that all target hypotheses, quality or safe-headroom gates passed. '
            'Final remote publication verification is a separate receipt.'))
    preserve();print('FINAL_SCIENTIFIC_SUMMARY_FROZEN',flush=True)


if __name__=='__main__':main()
