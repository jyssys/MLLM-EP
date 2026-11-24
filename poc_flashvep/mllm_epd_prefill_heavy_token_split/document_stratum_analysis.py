"""Preserve failed observer gates in the predeclared secondary document stratum."""
import json
import subprocess
import time

import numpy as np

from .common import ROOT,PYTHON,digest,preserve,write
from .document_stratum_control import TAG
from .token_discovery_analysis import table


def resource_observations(rows,left,right,role):
    selected=[r for r in rows if left<=r['begin_ns']/1e9<=right]
    before=[r['before'] for r in selected]
    ratios=[r['free_KV_blocks']/r['total_KV_blocks'] for r in before
        if r.get('free_KV_blocks') is not None and r.get('total_KV_blocks',0)>0]
    return dict(observations=len(selected),
        decode_active_max=max((r.get('decode_active',0) for r in before),default=None) if role=='decode' else None,
        decode_ready_waiting_max=max((r.get('decode_ready_waiting',0) for r in before),default=None) if role=='decode' else None,
        remote_KV_waiting_max=max((sum(n for k,n in r.get('waiting_status',{}).items() if 'REMOTE' in k) for r in before),default=None),
        free_KV_fraction_min=min(ratios) if ratios and role!='encoder' else None,
        free_KV_fraction_max=max(ratios) if ratios and role!='encoder' else None,
        encoder_cache_free_slots_min=min((r['encoder_free_slots'] for r in before if r.get('encoder_free_slots') is not None),default=None)
            if role=='encoder' else None,
        scope='Observed scheduler states only. Encoder dummy KV pool is not an LLM KV budget. No interval interpolation.')


def main():
    preserve();source=ROOT/'analysis'/f'{TAG}_summary.json';summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE'
    assert (ROOT/'state'/f'{TAG}_controller_complete.json').exists()
    names=[]
    for qps,label in [(1.,'PILOT'),(2.,'PROFILE1'),(3.,'PROFILE2')]:
        name=TAG+f'_CPC_q{qps}_v0';names.append(name)
        derived=ROOT/'analysis'/f'{name}_idle_analysis.json'
        if not derived.exists():
            subprocess.run([str(PYTHON),'-m','poc_flashvep.mllm_epd_prefill_heavy_token_split.profile_analysis',
                name,'--proxy-name',TAG+'_'+label+'_proxy'],check=True)
    activity=[];overlap=[];coverage=[];qualification=[];resources=[];cdf=[];pilots=[]
    raw_profiles=[]
    for name in names:
        data=json.loads((ROOT/'analysis'/f'{name}_idle_analysis.json').read_text())
        gate=json.loads((ROOT/'analysis'/f'{name}_qualification.json').read_text())
        qualified=gate['observer_gate_passed']
        scope='TRACE_ANALYSIS qualified observer' if qualified else 'DIAGNOSTIC TRACE_ANALYSIS: observer gate failed'
        raw_profiles.append(dict(name=name,QPS=data['QPS'],observer_gate=qualified,scope=scope))
        for metric,value in gate['comparison'].items():
            qualification.append(dict(QPS=data['QPS'],metric=metric,observer_gate=qualified,**value))
        for roi,part in data['ROIs'].items():
            overlap.append(dict(QPS=data['QPS'],ROI=roi,duration_seconds=part['duration_seconds'],observer_gate=qualified,
                simultaneous_idle_fraction_total=part['E_D_simultaneous_kernel_idle_fraction'],
                simultaneous_idle_during_P_busy_fraction_total=part['E_D_simultaneous_idle_during_P_busy_fraction_total'],
                simultaneous_idle_fraction_of_P_busy=part['E_D_simultaneous_idle_fraction_of_P_busy'],
                P_busy_E_idle_fraction_total=part['P_busy_E_idle_fraction_total'],
                P_busy_D_idle_fraction_total=part['P_busy_D_idle_fraction_total'],
                all_kernel_active_fraction_total=part['all_kernel_active_fraction'],scope=scope,
                actionable_slack='NOT_MEASURED. Kernel gap overlap is not admission or three-way parallelism proof.'))
            for role,value in part['roles'].items():
                activity.append(dict(QPS=data['QPS'],ROI=roi,role=role,duration_seconds=part['duration_seconds'],
                    kernel_busy_fraction=value['kernel_busy_fraction'],kernel_idle_fraction=value['kernel_idle_fraction'],
                    kernel_and_copy_quiet_fraction=value['kernel_and_copy_quiet_fraction'],
                    no_pending_HTTP_kernel_idle_fraction=value['no_pending_HTTP_kernel_idle_fraction'],
                    pending_HTTP_kernel_idle_fraction=value['pending_HTTP_kernel_idle_fraction'],
                    waiting_observation_mean=value['scheduler_waiting_mean_per_observation'],
                    waiting_max=value['scheduler_waiting_max'],actual_scheduled_requests_max=value['actual_scheduled_requests_max'],
                    actual_scheduled_tokens_max=value['actual_scheduled_tokens_max'],
                    observer_gate=qualified,scope=scope))
                sched=json.loads((ROOT/'raw/profiles'/name/role/'scheduler.json').read_text())
                resources.append(dict(QPS=data['QPS'],ROI=roi,role=role,observer_gate=qualified,
                    **resource_observations(sched['rows'],part['left_monotonic'],part['right_monotonic'],role)))
                coverage.extend(dict(QPS=data['QPS'],ROI=roi,role=role,observer_gate=qualified,scope=scope,**row)
                    for row in value['idle_windows']['threshold_coverage'])
        import csv
        with (ROOT/'analysis'/f'{name}_idle_windows.csv').open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row['ROI']=='central_arrival_window' and row['kind']=='kernel_idle' and row['role'] in ['encoder','decode']:
                    cdf.append(dict(QPS=data['QPS'],role=row['role'],duration_ms=float(row['duration_ms']),
                        observer_gate=qualified,scope=scope))
    for cell in summary['results']:
        if cell['label']=='warmup':continue
        metric=cell['metrics'];path=ROOT/'raw'/f'{metric["run"]}_requests.jsonl'
        assert digest(path)==metric['raw_SHA256'] and metric['completed_count']==37 and metric['failed_count']==0
        pilots.append(dict(QPS=float(cell['label'].removeprefix('natural_q')),requests=37,
            output_tokens=metric['total_output_tokens'],TTFT_p95_ms=metric['TTFT_p95_ms'],
            ITL_p99_ms=metric['ITL_p99_ms'],TPOT_p95_ms=metric['TPOT_p95_ms'],
            raw_output_tokens_per_second=metric['raw_output_tokens_per_second'],SLO_goodput=metric['SLO_goodput'],
            **{key:100*value for key,value in metric['gate_attainment'].items()},
            scope='MEASURED_ONLINE Native secondary source-stratum Natural N1. Quality pending. Not Main100.'))
    for name,rows in [('activity',activity),('overlap',overlap),('threshold_coverage',coverage),
            ('qualification',qualification),('resource_observations',resources),('idle_windows',cdf),('natural_pilots',pilots)]:
        table(ROOT/'analysis'/f'document_C39_{name}_v0.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));frows=[]
    for axis,key,label in zip(axes,['kernel_busy_fraction','waiting_observation_mean'],
            ['Kernel busy (% of central arrival window)','Waiting requests (mean per observation)']):
        for role in ['encoder','prefill','decode']:
            selected=[r for r in activity if r['ROI']=='central_arrival_window' and r['role']==role]
            multiplier=100 if key=='kernel_busy_fraction' else 1
            axis.plot([r['QPS'] for r in selected],[multiplier*r[key] for r in selected],marker='o',label=role)
            frows.extend(dict(QPS=r['QPS'],role=role,metric=key,value=multiplier*r[key],observer_gate=r['observer_gate'],scope=r['scope']) for r in selected)
        axis.set_xlabel('Offered QPS');axis.set_ylabel(label);axis.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Secondary document37 | N1 | DIAGNOSTIC: observer gates failed | Not safe headroom')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'document_C39_activity_queue_v0.{ext}',dpi=180)
    plt.close(fig);table(ROOT/'figures/document_C39_activity_queue_v0.csv',frows)
    fig,axes=plt.subplots(1,3,figsize=(13,4));figrows=[]
    for axis,qps in zip(axes,[1.,2.,3.]):
        for role in ['encoder','decode']:
            values=np.sort([r['duration_ms'] for r in cdf if r['QPS']==qps and r['role']==role])
            y=np.arange(1,len(values)+1)/len(values)
            axis.step(values,y,where='post',label=role)
            figrows.extend(dict(QPS=qps,role=role,duration_ms=float(v),CDF=float(p),
                scope='DIAGNOSTIC physical kernel gap CDF. Not safe executable windows.') for v,p in zip(values,y))
        axis.set_xscale('log');axis.set_title(f'QPS{qps:g}');axis.set_xlabel('Physical kernel gap (ms)');axis.grid(alpha=.2)
    axes[0].set_ylabel('Window-count CDF');axes[-1].legend();fig.suptitle('Secondary document37 | Failed observer gates retained')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'document_C39_idle_CDF_v0.{ext}',dpi=180)
    plt.close(fig);table(ROOT/'figures/document_C39_idle_CDF_v0.csv',figrows)
    write(ROOT/'analysis/document_C39_audit_v0.json',dict(time=time.time(),summary_SHA256=digest(source),
        profiles=raw_profiles,main100_replaced=False,prefill_heavy='NOT_CONFIRMED',
        service_proximity='NOT_MEASURED: no independent P capacity reference',
        SM_Tensor_HBM='NOT_MEASURED',actionable_E_D_slack='NOT_MEASURED',
        encode_safe_and_decode_safe_headroom='NOT_MEASURED in this source-stratum study',
        quality='PENDING',source_stratum='All37 pre-existing parent discovery MMLongBench-Doc requests. No performance selection.'))
    report=ROOT.parent/'mllm_epd_secondary_document_prefill_profile_v1.md';assert not report.exists()
    lines=['# Secondary Document37 — Native E/P/D 계측','',
        'Main100을 교체하지 않았다. 기존 Parent Discovery의 MMLongBench-Doc37개를 모두 사용했다.',
        'Source Stratum은 성능 결과 전에 동결했다. Natural N1 Pilot과 Equal-output CPC를 구분했다.',
        '전체 Task Quality는 PENDING이다. 이 실행은 Token Split 성능 비교가 아니다.','',
        '## Natural Online Pilot','',
        '| Offered QPS | Requests | Output Tokens | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Joint % |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in pilots:
        lines.append(f'| {row["QPS"]:g} | 37 | {row["output_tokens"]} | {row["TTFT_p95_ms"]:.3f} | '+
            f'{row["ITL_p99_ms"]:.3f} | {row["TPOT_p95_ms"]:.3f} | {row["JOINT_PASS"]:.3f} |')
    lines+=['','## CPC와 같은 시간 구간의 GPU 상태','',
        '각 QPS는 새 E/P/D 초기화를 사용했다. Worker당 Collector 시작·종료는 1회다.',
        'Profile 차이와 두 Clean의 Spread가 모두 5% 이하여야 Primary 계측으로 인정했다.',
        'QPS1/2/3 모두 이 Gate를 통과하지 못했다. 모든 Kernel 수치는 `DIAGNOSTIC`이다.',
        'Clean Spread 실패만으로 Profiler가 그 차이를 일으켰다고 단정하지 않는다.',
        '아래 분모는 예정 Arrival p10–p90 구간이다. Sustainable Steady State로 확정하지 않았다.','',
        '| QPS | E Busy % | P Busy % | D Busy % | E Waiting Max | P Waiting Max | D Waiting Max | E&D Idle % | E&D Idle/P Busy % |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for qps in [1.,2.,3.]:
        roles={r['role']:r for r in activity if r['QPS']==qps and r['ROI']=='central_arrival_window'}
        both=next(r for r in overlap if r['QPS']==qps and r['ROI']=='central_arrival_window')
        lines.append(f'| {qps:g} | '+ ' | '.join(f'{100*roles[r]["kernel_busy_fraction"]:.3f}' for r in ['encoder','prefill','decode'])+
            ' | '+' | '.join(str(roles[r]['waiting_max']) for r in ['encoder','prefill','decode'])+
            f' | {100*both["simultaneous_idle_fraction_total"]:.3f} | {100*both["simultaneous_idle_fraction_of_P_busy"]:.3f} |')
    lines+=['','Kernel Busy는 Kernel 구간 Union이다. SM Active·Tensor Core Activity·HBM Bandwidth는 미측정이다.',
        'Waiting Max와 평균은 Scheduler 관측값이다. 50ms Sampling을 시간 가중 Queue로 바꾸지 않았다.',
        'D Decode-ready, Remote-KV 대기와 Free KV Block은 별도 Resource CSV에 있다.',
        'E의 Dummy KV Pool은 추가 LLM Weight의 KV Capacity가 아니다.',
        '전체 Cohort와 중앙 Arrival 결과는 따로 보존했다. Warm-up은 모든 Main 결과에서 제외했다.','',
        '## 판정과 한계','',
        '이 Profile만으로 E 비병목·P 포화·D Decode-safe 여유를 동시에 확정할 수 없다.',
        '독립 P Capacity를 측정하지 않았다. Service Proximity≥85%도 미측정이다.',
        '큰 E/D Kernel Gap과 동시 Idle은 향후 실행 가능 Window 또는 3GPU 병렬화의 증거가 아니다.',
        'No-pending HTTP Idle과 Pending HTTP Idle은 분리했다.',
        'Host Scheduler·Input·Transfer·작업 내부 Gap·Downstream Backpressure의 완전한 인과 분류는 미측정이다.',
        '해당 Label을 모든 Window에 임의로 부여하지 않았다. 실제 추가 Prefill 안전성은 별도 Intervention이 필요하다.',
        '이 분석은 `PREFILL_HEAVY_NOT_CONFIRMED`다. 미확정을 확인된 부재로 해석하지 않는다.','',
        '## 재현 계약','',
        'Qwen2.5-VL-7B Revision cc594898137f460bfe9f0759e9844b3ce807cfb5를 사용했다.',
        'Original BF16, BI0, 원본 Vision/Processor, Full28, B8192/S1024, KV42GiB, Block128이다.',
        'P/D는 공통 Compiled Reduction 설정이다. GPU4/5/6은 E/P/D다.',
        'Order Seed2026112509와 Arrival Seed2026112510을 사용했다.',
        'Manifest는 `data/secondary_document37_discovery_v0.json`이다.',
        'CPC의 Output 수는 기존 R0 Natural 결과로 고정했다. Natural Pilot과 평균을 합치지 않았다.',
        'Raw Profile은 `raw/profiles/ec_DOCUMENT_C39R1_CPC_*`에 있다.',
        '집계 CSV/JSON은 `analysis/document_C39_*_v0`에 있다. Figure는 `figures/document_C39_*_v0`에 있다.','']
    report.write_text('\n'.join(lines));preserve();print('DOCUMENT_SECONDARY_PROFILE_ANALYZED',flush=True)


if __name__=='__main__':main()
