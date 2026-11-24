"""Report qualified low-load physical activity without a safe-window claim."""
import csv
import json
import time

from .common import ROOT,digest,preserve,write


def main():
    preserve();name='ec_LOW_CUPTI_C25_CPC_q0.5_v0'
    source=ROOT/'analysis'/f'{name}_idle_analysis.json'
    value=json.loads(source.read_text())
    qualified=ROOT/'analysis'/f'{name}_qualification.json'
    gate=json.loads(qualified.read_text())
    assert gate['observer_gate_passed'] and value['observer_gate_passed']
    assert all(r['within_5_percent'] for r in gate['comparison'].values())
    roi=value['ROIs']['central_arrival_window'];rows=[];coverage=[]
    for role,stage in roi['roles'].items():
        rows.append(dict(role=role,ROI_seconds=roi['duration_seconds'],QPS=.5,independent_starts=1,
            kernel_busy_percent=100*stage['kernel_busy_fraction'],kernel_idle_percent=100*(1-stage['kernel_busy_fraction']),
            kernel_and_copy_quiet_percent=100*stage['kernel_and_copy_quiet_fraction'],
            no_pending_HTTP_kernel_idle_percent=100*stage['no_pending_HTTP_kernel_idle_fraction'],
            pending_HTTP_kernel_idle_percent=100*stage['pending_HTTP_kernel_idle_fraction'],
            observed_waiting_mean=stage['scheduler_waiting_mean_per_observation'],
            observed_waiting_max=stage['scheduler_waiting_max'],actual_scheduled_requests_max=stage['actual_scheduled_requests_max'],
            evidence='TRACE_ANALYSIS, qualified low-load CUPTI observer; task quality pending'))
        for item in stage['idle_windows']['threshold_coverage']:
            coverage.append(dict(role=role,threshold_ms=item['threshold_ms'],
                window_count_fraction=item['count_fraction'],idle_duration_fraction=item['duration_fraction']))
    for stem,records in [('stage_activity',rows),('threshold_coverage',coverage)]:
        with (ROOT/'figures'/f'low_load_C25_{stem}_v0.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(7,4))
    axis.bar([r['role'] for r in rows],[r['kernel_busy_percent'] for r in rows])
    axis.set_ylabel('Kernel busy (% of central arrival window)');axis.set_ylim(0,100)
    axis.set_title('Qualified CUPTI observer | QPS0.5 | N1 | Quality pending')
    axis.grid(axis='y',alpha=.2);fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'low_load_C25_stage_activity_v0.{extension}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for role in ['encoder','decode']:
        selected=[r for r in coverage if r['role']==role]
        for axis,key,title in zip(axes,['window_count_fraction','idle_duration_fraction'],['Window count','Idle-duration weight']):
            axis.semilogx([r['threshold_ms'] for r in selected],[100*r[key] for r in selected],marker='o',label=role)
            axis.set_title(title);axis.set_xlabel('Physical kernel gap threshold (ms)');axis.grid(alpha=.2)
    axes[0].set_ylabel('Coverage (%)');axes[1].legend()
    fig.suptitle('Qualified low-load observation; not safe-admission budgets');fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'low_load_C25_threshold_coverage_v0.{extension}',dpi=180)
    plt.close(fig)
    write(ROOT/'analysis/low_load_C25_evidence_v0.json',dict(time=time.time(),source_SHA256=digest(source),
        qualification_SHA256=digest(qualified),observer_gate='PASS',task_quality='PENDING',
        physical_encoder_idle='E_IDLE_CONFIRMED on tested low-load trace',
        simultaneous_physical_idle='E_D_SIMULTANEOUS_IDLE_CONFIRMED on tested low-load trace',
        prefill_heavy='NOT_CONFIRMED at this low load',encode_safe_headroom='NOT_MEASURED',
        decode_safe_headroom='NOT_MEASURED',additional_online_prefill='NOT_RUN',
        central_ROI=roi,independent_starts=1))
    report=ROOT.parent/'mllm_epd_low_load_kernel_idle_validation_v1.md';assert not report.exists()
    lines=['# Low-load 실제 Kernel Idle 검증 — C25','',
        'QPS0.5의 Full CUPTI Clean/Profile/Clean이 고정한 5% 계측 Gate를 통과했다.',
        'E/D의 물리적 Kernel Idle과 동시 Idle은 관측됐다. 이 조건은 Prefill-heavy가 아니다.',
        '추가 Prefill은 실행하지 않았다. Encode-safe와 Decode-safe Headroom은 `NOT_MEASURED`다.','',
        '## 계약과 증거 수준','',
        '증거 수준은 `TRACE_ANALYSIS`다. 실제 Main100 Online EPD 실행의 Kernel/Copy Activity를 사용했다.',
        'Task Quality는 미확정이다. 이 보고서는 품질 PASS 또는 Online Token Split 성과가 아니다.',
        'Original Qwen2.5-VL-7B BF16, BI0, Full28, 원본 Processor와 동일한 Arrival을 유지했다.',
        '모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.',
        'GPU4/5/6은 E/P/D다. B8192/S1024, KV42GiB, Block128과 원래 Compiled 경로를 사용했다.',
        '각 실행은 Main100과 고정한 Output5269 Tokens다. 독립 초기화는 1회다.','',
        '## 계측 Gate','',
        '| Metric | Profile Difference % | Clean Spread % | Gate |',
        '|---|---:|---:|---|']
    for metric,item in gate['comparison'].items():
        lines.append(f"| {metric} | {100*item['profile_relative_difference']:.4f} | "
            f"{100*item['clean_relative_spread']:.4f} | PASS |")
    lines+=['','차이는 비계측 두 실행의 평균 대비 값이다. Clean Spread도 같은 평균을 분모로 사용한다.',
        '모든 Metric과 Collector 검사가 통과했다. 이전 QPS2/4의 실패 판정은 변경하지 않았다.','',
        '## 같은 시간 구간의 E/P/D 상태','',
        f"Central Arrival Window는 예정 Arrival p10–p90 구간이다. 길이는 {roi['duration_seconds']:.6f}초다.",
        '마지막 Drain과 긴 초기 구간은 아래 분모에서 제외했다. Sustainable Steady State를 뜻하지 않는다.','',
        '| Stage | Kernel Busy % | Kernel Idle % | Kernel+Copy Quiet % | Observed Waiting Mean | Waiting Max | Actual Scheduled Max |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['role']} | {row['kernel_busy_percent']:.4f} | {row['kernel_idle_percent']:.4f} | "
            f"{row['kernel_and_copy_quiet_percent']:.4f} | {row['observed_waiting_mean']:.6f} | "
            f"{row['observed_waiting_max']} | {row['actual_scheduled_requests_max']} |")
    lines+=['','Waiting은 Scheduler 관측별 평균이다. 정확한 시간 가중 Queue 평균이 아니다.',
        'Actual Scheduled Max는 관측한 요청 수다. B/S 상한 또는 Prefill/Decode Batch가 모두 같다는 뜻이 아니다.','',
        f"E/D 동시 Kernel Idle은 Central Window의 {100*roi['E_D_simultaneous_kernel_idle_fraction']:.4f}%다.",
        f"그 시간과 P Busy의 겹침은 전체 Central Window의 {100*roi['E_D_simultaneous_idle_during_P_busy_fraction_total']:.4f}%다.",
        f"같은 겹침은 P Busy 시간의 {100*roi['E_D_simultaneous_idle_fraction_of_P_busy']:.4f}%다.",
        '두 분모를 혼합하지 않는다. 동시 Idle이 동일 요청의 3GPU 병렬화를 증명하지는 않는다.','',
        '## Window 해석과 한계','',
        'Kernel Gap 수에는 작업 내부의 매우 짧은 Gap이 많다. Count CDF와 Duration 가중 Coverage를 구분했다.',
        '100ms 이상 Gap은 E Idle Time의 99.3391%, D Idle Time의 98.9227%를 차지했다.',
        '같은 기준의 Window Count 비율은 E0.091111%, D0.007100%다.',
        '이 값은 미래 Idle Duration을 미리 아는 Scheduler가 실행 가능하다는 뜻이 아니다.','',
        'Pending HTTP가 없는 Kernel Idle과 Pending HTTP가 있는 Kernel Idle을 따로 기록했다.',
        'Host Scheduler, CPU Media Preparation, Transfer Wait와 Downstream Backpressure의 정확한 분해는 미완료다.',
        '따라서 모든 Gap에 TRUE_ENCODE_IDLE 또는 Actionable Slack Label을 부여하지 않았다.',
        'Kernel Busy는 SM Active, Tensor Core Activity 또는 HBM Bandwidth Utilization이 아니다. 그 Counter는 미측정이다.',
        'GPU 간 짧은 Gap의 선후관계에는 Clock Anchor 오차가 있다. 원시 Anchor 폭은 JSON에 보존했다.',
        '추가 Prefill과 Encoder/Decode SLO 비열등성은 별도 Online Intervention이 필요하다.','',
        '원시 Trace와 Collector Counter는 `raw/profiles/ec_LOW_CUPTI_C25_CPC_q0.5_v0/`에 있다.',
        '대응 Window CSV는 `analysis/ec_LOW_CUPTI_C25_CPC_q0.5_v0_idle_windows.csv`다.',
        'Stage 및 Threshold Figure는 `figures/low_load_C25_*_v0.{png,pdf,csv}`다.',
        '모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.','']
    report.write_text('\n'.join(lines));preserve()
    print('LOW_LOAD_QUALIFIED_KERNEL_IDLE_REPORTED',flush=True)


if __name__=='__main__':main()
