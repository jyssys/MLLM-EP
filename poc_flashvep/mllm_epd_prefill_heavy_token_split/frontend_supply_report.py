"""Keep the negative renderer result and physical-idle limitations visible."""
import csv
import json
import time

from .common import ROOT,digest,preserve,write


def table(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    preserve()
    results=json.loads((ROOT/'analysis/frontend_supply_discovery_v0.json').read_text())
    rows=[]
    for label,value in results['results'].items():
        rows.append(dict(arm=label,starts=1,QPS=3,requests=value['completed_count'],
            output_tokens=value['total_output_tokens'],TTFT_p50_ms=value['TTFT_p50_ms'],
            TTFT_p95_ms=value['TTFT_p95_ms'],ITL_p99_ms=value['ITL_p99_ms'],TPOT_p95_ms=value['TPOT_p95_ms'],
            duration_seconds=value['duration_seconds'],all_ITL_pass=value['gate_attainment']['ITL_PASS'],
            joint_pass=value['gate_attainment']['JOINT_PASS'],raw_SHA256=value['raw_SHA256']))
    table(ROOT/'analysis/frontend_supply_controls_v0.csv',rows)
    profile='frontend_B5_cache0_T2_CPC_q3_v0'
    idle=json.loads((ROOT/'analysis'/f'{profile}_idle_analysis.json').read_text())
    qualification=json.loads((ROOT/'analysis'/f'{profile}_qualification.json').read_text())
    central=idle['ROIs']['central_arrival_window']
    counts=[]
    for roi,result in idle['ROIs'].items():
        for role,value in result['roles'].items():
            counts.append(dict(ROI=roi,role=role,QPS=3,ROI_seconds=result['duration_seconds'],
                kernel_busy=value['kernel_busy_fraction'],kernel_idle=value['kernel_idle_fraction'],
                pending_HTTP_kernel_idle=value['pending_HTTP_kernel_idle_fraction'],
                no_pending_HTTP_kernel_idle=value['no_pending_HTTP_kernel_idle_fraction'],
                scheduler_waiting_mean_per_observation=value['scheduler_waiting_mean_per_observation'],
                scheduler_waiting_max=value['scheduler_waiting_max'],
                scheduled_requests_max=value['actual_scheduled_requests_max'],
                scheduled_tokens_max=value['actual_scheduled_tokens_max']))
    table(ROOT/'analysis/frontend_B5_E_P_D_activity_v0.csv',counts)
    thresholds=[]
    for roi,result in idle['ROIs'].items():
        for role,value in result['roles'].items():
            for kind in ['idle_windows','no_pending_HTTP_idle_windows']:
                for item in value[kind].get('threshold_coverage',[]):
                    thresholds.append(dict(ROI=roi,role=role,kind=kind,**item))
    table(ROOT/'analysis/frontend_B5_idle_threshold_coverage_v0.csv',thresholds)
    memory=json.loads((ROOT/'analysis/frontend_B5_cache0_T2_worker_memory_and_scheduler.json').read_text())
    memory_rows=[]
    for role,items in memory['workers'].items():
        for value in items:
            memory_rows.append(dict(role=role,allocated_GiB=value['allocated_bytes']/2**30,
                reserved_GiB=value['reserved_bytes']/2**30,peak_allocated_GiB=value['maximum_allocated_bytes']/2**30,
                device_free_GiB=value['device_free_bytes']/2**30,
                logical_parameter_GiB=value['model_parameter_logical_bytes']/2**30))
    table(ROOT/'analysis/frontend_B5_memory_counters_v0.csv',memory_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    labels=['Cache1/T1','Cache0/T1','Cache0/T2']
    for ax,key,unit in zip(axes,['TTFT_p95_ms','ITL_p99_ms'],['TTFT p95 (ms)','ITL p99 (ms)']):
        ax.bar(labels,[r[key] for r in rows]);ax.set_ylabel(unit)
    fig.suptitle('DIAGNOSTIC | Official CPU renderer control | q3 | One start per cell')
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'frontend_supply_controls_v0.{extension}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,field,title in zip(axes,['count_fraction','duration_fraction'],['Window count coverage','Idle-time weighted coverage']):
        for role in ['encoder','decode']:
            values=[r for r in thresholds if r['ROI']=='central_arrival_window' and r['role']==role and r['kind']=='idle_windows']
            ax.plot([r['threshold_ms'] for r in values],[100*r[field] for r in values],marker='o',label=role)
        ax.set_xscale('log');ax.set_xlabel('Minimum physical gap (ms)');ax.set_ylabel('Coverage (%)');ax.set_title(title);ax.legend()
    fig.suptitle('DIAGNOSTIC | Physical inactivity, not safe-admission windows')
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'frontend_B5_idle_threshold_coverage_v0.{extension}',dpi=160)
    plt.close(fig)
    natural=json.loads((ROOT/'analysis/frontend_B5_grid_natural100_q2_v0_metrics.json').read_text())
    frozen=dict(time=time.time(),rows=rows,activity=counts,thresholds=thresholds,memory=memory_rows,
        observer_gate_passed=qualification['observer_gate_passed'],
        primary_status='DIAGNOSTIC',prefill_heavy='NOT_CONFIRMED',native_quality='PENDING',
        generic_token_split='NOT_RUN',encode_safe_headroom='NOT_MEASURED',decode_safe_headroom='NOT_MEASURED',
        interpretation='Cache0/renderer2 did not establish improved capacity or P saturation. '
            'Physical E/D idle overlap is not evidence that extra prefill can be admitted safely.')
    write(ROOT/'analysis/frontend_supply_controls_v0.json',frozen)
    lines=['# MLLM EPD Encoder Frontend Supply Control','',
        '판정은 DIAGNOSTIC이다. 전처리 Worker2는 P 포화나 Capacity 개선을 입증하지 못했다.',
        'Generic Token Split은 실행하지 않았다. Main100과 보존한 Main400은 바꾸지 않았다.','',
        '## 변경한 요소','',
        'A는 Processor Cache1GiB와 Renderer Worker1을 사용했다.',
        'B는 E의 Processor Cache만 0으로 바꿨다. P/D의 Cache1GiB는 유지했다.',
        'C는 B에서 E의 Renderer Worker만 2로 바꿨다.',
        'Cache를 켠 상태의 Worker2는 설치된 공식 Runtime이 금지한다. 이 검사는 제거하지 않았다.',
        '각 E 재시작에는 새 EC·Metadata 경로를 사용했다. 이전 Cache0의 Engine-local Image ID는 재사용하지 않았다.',
        'P/D는 E의 원래 Token·Hash·Image Grid를 전달받았다. 원본 이미지와 Vision Encoder 계산은 바꾸지 않았다.',
        'B와 C의 N8 Natural EOS는 기존 같은 Geometry Native EPD와 토큰·입력 ID Strict Match를 통과했다.',
        '세 Arm은 같은 GPU4/5/6을 순차 사용했다. Source Archive와 Boot별 UUID는 실행 Config에 보존했다.',
        'Model·Processor Revision, BF16 BI0, Prefix Cache Off와 Natural 생성 상한2,048은 유지했다.',
        'LLM의 상한은 B8192/S1024, KV42GiB/GPU, Block128이다.',
        'E는 B114688/S1024이며 실제 Chunked Prefill은 꺼져 있다. E는 KV 없는 Encoder 전용 Scheduler를 사용한다.',
        'CPU 기록 정책은 모든 Arm에서 Immediate였다. Source Credit이나 인위적인 D Cap은 추가하지 않았다.','',
        '## 동일 출력 C3 Screening','',
        'QPS3, 도착 Seed2026112402, 순서 Seed2026112404를 사용했다.',
        '각 Arm은 같은100개 요청과 5,269개 출력 Tokens를 완료했다. 이는 Natural EOS 성능표가 아니다.',
        'Cell당 독립 초기화는 1회다. 독립 Start3회 CI는 없다.','',
        '| Arm | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | All-ITL PASS % |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["arm"]} | {row["TTFT_p50_ms"]:.3f} | {row["TTFT_p95_ms"]:.3f} | '
            f'{row["ITL_p99_ms"]:.3f} | {row["TPOT_p95_ms"]:.3f} | {row["all_ITL_pass"]*100:.2f} |')
    lines+=['','Worker2는 Worker1보다 TTFT p95가 낮지 않았다. 따라서 단순 CPU Worker 수를 해결책으로 채택하지 않는다.',
        '## GPU 활동과 관찰자 검사','',
        'C의 Clean/Profile/Clean은 각각100개 요청과 같은 출력 작업량을 사용했다.',
        'Profile과 두 Clean 평균의 차이는 모든 동결 지표에서 5% 이하였다.',
        '그러나 Clean 간 TPOT p95 차이는 7.73%였다. 동결한 관찰자 안정성 Gate를 통과하지 못했다.',
        '이는 Profiler 자체의 간섭이 5%를 넘었다는 증거와는 다르다. 그래도 GPU 결과는 Diagnostic으로 제한한다.',
        '중앙 도착 구간은 전체 요청의 예정 도착 시각 p10–p90이다. 지속 가능한 Steady-state를 입증한 구간은 아니다.','',
        '| Stage | Kernel Busy % | Pending HTTP + Kernel Idle % | No Pending HTTP + Kernel Idle % |',
        '|---|---:|---:|---:|']
    for role,value in central['roles'].items():
        lines.append(f'| {role} | {100*value["kernel_busy_fraction"]:.3f} | '
            f'{100*value["pending_HTTP_kernel_idle_fraction"]:.3f} | {100*value["no_pending_HTTP_kernel_idle_fraction"]:.3f} |')
    lines += ['',f'E/D 동시 Kernel Idle은 P Busy 시간의 {100*central["E_D_simultaneous_idle_fraction_of_P_busy"]:.3f}%였다.',
        '이 값은 물리적 Kernel Inactivity의 겹침이다. 실행 가능한 추가 Prefill Budget은 아니다.',
        'HTTP Pending에는 전처리·Transfer·Scheduler·Service가 모두 포함된다.',
        '이 자료만으로 E Idle을 다섯 원인에 정확히 분배할 수 없다. 미분리 부분은 NOT_MEASURED다.',
        'SM Active, Tensor Core Active, HBM 대역폭 포화율은 이 계측에서 측정하지 않았다.',
        'P Busy가 80% 미만이며 지속적인 P Queue도 입증하지 못했다. PREFILL_HEAVY_NOT_CONFIRMED를 유지한다.',
        '실제 추가 Prefill 삽입은 없었다. Encode-safe와 Decode-safe Headroom은 NOT_MEASURED다.','',
        '## 별도 Natural EOS 확인','',
        f'C의 QPS2 Natural100은 {natural["total_output_tokens"]:,}개 Tokens를 출력했다.',
        f'TTFT p95는 {natural["TTFT_p95_ms"]:.3f}ms, ITL p99는 {natural["ITL_p99_ms"]:.3f}ms였다.',
        f'All-ITL PASS는 {100*natural["gate_attainment"]["ITL_PASS"]:.2f}%, Joint PASS는 {100*natural["gate_attainment"]["JOINT_PASS"]:.2f}%였다.',
        '이는 N1 확인이다. 상대 비열등성 CI와 전체 Task Quality Gate는 아직 미완료다.',
        '자연 출력 작업량이 다른 과거 실행과 섞어 평균을 계산하지 않는다.','',
        '## 메모리 범위','',
        '| Stage | Allocated GiB | Reserved GiB | Peak Allocated GiB | Logical Parameter GiB |',
        '|---|---:|---:|---:|---:|']
    for row in memory_rows:
        lines.append(f'| {row["role"]} | {row["allocated_GiB"]:.3f} | {row["reserved_GiB"]:.3f} | '
            f'{row["peak_allocated_GiB"]:.3f} | {row["logical_parameter_GiB"]:.3f} |')
    lines+=['','이는 Quiescent 시점의 Allocator·Device Counter다. E에 추가 LLM을 올린 Co-residency 결과는 아니다.',
        '현재 E는 Vision Weight만 보유했다. 추가 LLM Weight·KV·Workspace의 동시 Peak는 아직 측정하지 않았다.',
        '논리 Weight 크기만으로 안전한 추가 Memory Budget을 확정하지 않는다.','',
        '## 다음 원인 검증','',
        '설치된 ECExampleConnector는 Debug 구현이다. E에서 이미지마다 동기 CPU 복사와 Safetensors 저장을 수행한다.',
        '새 관측은 이 경로의 정확한 인과 기여율을 확정하지 않는다.',
        '별도 GPU0/1에서 CUDA IPC의 Tensor·View·Lifetime 프로토콜을 먼저 검사했다.',
        '이 프로토콜 검사는 Serving Connector나 Generic Token Split 구현을 뜻하지 않는다.',
        '이후 Transport 변경을 진행하면 N1→N8, KV·MRoPE, Natural 품질과 독립 A/B 검사를 다시 통과해야 한다.',
        '기존 Native의 Task Quality 문제와 Observer 실패는 보존한다. 이 보고서는 Gate를 완화하지 않는다.','',
        'CSV: `mllm_epd_prefill_heavy_token_split_20261124/analysis/frontend_*_v0.csv`.',
        'PNG/PDF: `mllm_epd_prefill_heavy_token_split_20261124/figures/frontend_*_v0.*`.', '']
    (ROOT.parent/'mllm_epd_encoder_frontend_supply_control.md').write_text('\n'.join(lines))
    preserve();print('FRONTEND_SUPPLY_REPORT_WRITTEN',len(rows))


if __name__=='__main__':main()
