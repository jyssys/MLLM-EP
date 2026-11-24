"""Show measured idle observations without joining different runtime contracts."""
import argparse
import hashlib
import json
import time

from .common import ROOT,digest,preserve,write
from .profile_analysis import load_activity,union
from .token_discovery_analysis import table

TAGS=['ec_LOW_CUPTI_C25_CPC_q0.5_v0','ec_ROI_CUPTI_C27_CPC_q3_v0',
    'ec_NATIVE_C11_S2_CPC_q4_v0']
FLAGS=['--dtype','--max-model-len','--max-num-batched-tokens','--max-num-seqs',
    '--kv-cache-memory-bytes','--block-size','--mm-processor-cache-gb','--renderer-num-workers',
    '--compilation-config','--no-enable-prefix-caching','--enable-chunked-prefill',
    '--enforce-eager','--no-async-scheduling']
BOOL_FLAGS=set(FLAGS[-4:])
CORE_FILES=['proxy.py','processor_handoff.py','cuda_ipc_ec.py']


def runtime_fingerprint(config):
    """Compare explicit engine flags and three named core files, not all code."""
    roles={};source={}
    for server in config['servers'].values():
        command=server['command'];fields={}
        for flag in FLAGS:
            fields[flag]=(True if flag in BOOL_FLAGS else command[command.index(flag)+1]) if flag in command else None
        roles[server['role']]=dict(GPU=server['gpu'],explicit_flags=fields)
        hashes={name:server['source_SHA256'].get(name) for name in CORE_FILES}
        assert all(hashes.values())
        if source:assert source==hashes
        source=hashes
    content=dict(roles=roles,core_source_SHA256=source,
        manifest_SHA256=config['manifest_SHA256'],reference_SHA256=config['reference_SHA256'])
    encoded=json.dumps(content,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(encoded).hexdigest(),content


def activity_rows(tag,data,gate,fingerprint):
    roi=data['ROIs']['central_arrival_window'];rows=[]
    assert gate['observer_gate_passed']==data['observer_gate_passed']
    evidence='TRACE_ANALYSIS' if gate['observer_gate_passed'] else 'DIAGNOSTIC'
    for role,value in roi['roles'].items():
        rows.append(dict(tag=tag,QPS=data['QPS'],role=role,independent_initializations=1,
            runtime_flag_and_core_fingerprint=fingerprint,evidence=evidence,
            observer_gate_passed=gate['observer_gate_passed'],
            ROI='central_arrival_window_observed_intersection' if data.get('frozen_completion_roi') else 'central_arrival_p10_p90',
            ROI_seconds=roi['duration_seconds'],kernel_busy_percent=100*value['kernel_busy_fraction'],
            kernel_idle_percent=100*value['kernel_idle_fraction'],
            observed_scheduler_waiting_mean=value['scheduler_waiting_mean_per_observation'],
            observed_scheduler_waiting_max=value['scheduler_waiting_max'],
            observed_scheduled_requests_max=value['actual_scheduled_requests_max'],
            simultaneous_idle_percent_total=100*roi['E_D_simultaneous_kernel_idle_fraction'],
            simultaneous_idle_P_busy_percent_total=100*roi['E_D_simultaneous_idle_during_P_busy_fraction_total'],
            simultaneous_idle_percent_of_P_busy=100*roi['E_D_simultaneous_idle_fraction_of_P_busy'],
            SM_active='NOT_MEASURED',tensor_core_activity='NOT_MEASURED',HBM_bandwidth='NOT_MEASURED',
            scope='Actual kernel union and observed scheduler states. No safe-window or sustainable-capacity claim.'))
    return rows


def save_figure(fig,name,rows):
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
    table(ROOT/'figures'/f'{name}.csv',rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v1')
    args=parser.parse_args();version=args.version
    assert version.startswith('v') and version[1:].isdigit()
    preserve();rows=[];contracts=[]
    for tag in TAGS:
        path=ROOT/'configs'/f'{tag}.json'
        if not path.exists():path=ROOT/'configs'/f'{tag}_profile.json'
        config=json.loads(path.read_text());fingerprint,content=runtime_fingerprint(config)
        data_path=ROOT/'analysis'/f'{tag}_idle_analysis.json'
        gate_path=ROOT/'analysis'/f'{tag}_qualification.json'
        data=json.loads(data_path.read_text());gate=json.loads(gate_path.read_text())
        rows.extend(activity_rows(tag,data,gate,fingerprint))
        contracts.append(dict(tag=tag,config_SHA256=digest(path),analysis_SHA256=digest(data_path),
            qualification_SHA256=digest(gate_path),fingerprint=fingerprint,contract=content))
    assert contracts[0]['fingerprint']==contracts[1]['fingerprint']
    assert contracts[0]['fingerprint']!=contracts[2]['fingerprint']
    table(ROOT/'analysis'/f'idle_scope_activity_{version}.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families=list(dict.fromkeys(r['runtime_flag_and_core_fingerprint'] for r in rows))
    colors=dict(encoder='tab:blue',prefill='tab:green',decode='tab:orange')
    for metric,label,name in [('kernel_busy_percent','Kernel busy (% of actual observed ROI)',f'idle_scope_busy_vs_QPS_{version}'),
            ('observed_scheduler_waiting_mean','Scheduler waiting (mean per observation)',f'idle_scope_queue_vs_QPS_{version}')]:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),squeeze=False)
        for axis,family in zip(axes[0],families):
            for row in rows:
                if row['runtime_flag_and_core_fingerprint']!=family:continue
                marker='o' if row['observer_gate_passed'] else 'x'
                axis.scatter(row['QPS'],row[metric],marker=marker,color=colors[row['role']],s=55)
                axis.annotate(row['role']+' '+('PASS' if row['observer_gate_passed'] else 'DIAG'),
                    (row['QPS'],row[metric]),xytext=(5,{'encoder':-15,'prefill':2,'decode':17}[row['role']]),
                    textcoords='offset points',fontsize=7)
            axis.set_title('Core/flag fingerprint '+family[:10]);axis.set_xlabel('Offered QPS (requests/s)')
            axis.set_ylabel(label);axis.grid(alpha=.2);axis.margins(x=.35)
            if metric=='kernel_busy_percent':axis.set_ylim(0,100)
        fig.suptitle('Observed points only | o: observer PASS, x: DIAGNOSTIC | ROIs differ; no joined curve')
        fig.tight_layout();save_figure(fig,name,rows);plt.close(fig)
    unique=[r for r in rows if r['role']=='encoder']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    metrics=['simultaneous_idle_percent_total','simultaneous_idle_percent_of_P_busy']
    for axis,metric,title in zip(axes,metrics,['E&D idle / observed ROI','E&D idle during P busy / P busy']):
        for row in unique:
            index=families.index(row['runtime_flag_and_core_fingerprint'])
            axis.scatter(row['QPS'],row[metric],marker='o' if row['observer_gate_passed'] else 'x',
                s=55,color=['tab:blue','tab:purple'][index])
            axis.annotate(('C25' if row['QPS']==.5 else 'C27' if row['QPS']==3 else 'C11')+
                (' PASS' if row['observer_gate_passed'] else ' DIAG'),(row['QPS'],row[metric]),
                xytext=(5,4),textcoords='offset points',fontsize=8)
        axis.set_title(title);axis.set_xlabel('Offered QPS (requests/s)');axis.set_ylabel('Physical overlap (%)')
        axis.set_ylim(0,100);axis.grid(alpha=.2);axis.margins(x=.3)
    fig.suptitle('No actionable-slack inference | Blue/purple: different core/flag fingerprints | Different ROIs')
    fig.tight_layout();save_figure(fig,f'idle_scope_simultaneous_vs_QPS_{version}',unique);plt.close(fig)
    low=ROOT/'analysis'/f'{TAGS[0]}_idle_analysis.json';data=json.loads(low.read_text())
    roi=data['ROIs']['central_arrival_window'];center=(roi['left_monotonic']+roi['right_monotonic'])/2
    lo,hi=center-2,center+2;segments=[]
    fig,axis=plt.subplots(figsize=(13,4.2))
    for index,role in enumerate(['encoder','prefill','decode']):
        kernels,copies,anchor,_=load_activity(ROOT/'raw/profiles'/TAGS[0]/role)
        for kind,activity,offset,height,color in [('kernel',kernels,0,.55,colors[role]),('copy_or_memset',copies,.6,.2,'tab:red')]:
            values=union(activity,lo,hi)
            axis.broken_barh([(a-lo,b-a) for a,b in values],(index+offset,height),facecolors=color)
            segments.extend(dict(tag=TAGS[0],role=role,kind=kind,start_monotonic=a,end_monotonic=b,
                relative_begin_seconds=a-lo,duration_ms=(b-a)*1000,ROI_left_monotonic=lo,ROI_right_monotonic=hi,
                clock_endpoint_bracket_max_us=anchor['endpoint_host_bracket_max_us'],
                evidence='TRACE_ANALYSIS',scope='Qualified C25; fixed 4s at central-ROI midpoint. Not actionable slack.') for a,b in values)
    axis.set_yticks([.3,1.3,2.3],['E GPU4','P GPU5','D GPU6'])
    axis.set_xlim(0,4);axis.set_xlabel('Seconds from fixed ROI left edge');axis.grid(axis='x',alpha=.2)
    axis.set_title('Actual CUPTI: kernel union + red copy/memset | QPS0.5 N1 | No prefill-heavy/safe-window claim')
    fig.tight_layout();save_figure(fig,f'idle_scope_C25_aligned_GPU_timeline_{version}',segments);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True);queue_rows=[]
    for axis,role in zip(axes,['encoder','prefill','decode']):
        source=ROOT/'raw/profiles'/TAGS[0]/role/'scheduler.json'
        scheduler=json.loads(source.read_text())
        selected=[r for r in scheduler['rows'] if roi['left_monotonic']<=r['begin_ns']/1e9<=roi['right_monotonic']]
        times=[r['begin_ns']/1e9-roi['left_monotonic'] for r in selected]
        for metric,marker in [('waiting','o'),('running','x')]:
            axis.plot(times,[r['before'][metric] for r in selected],linestyle='None',marker=marker,
                markersize=2,label='before-schedule '+metric)
        queue_rows.extend(dict(tag=TAGS[0],role=role,begin_monotonic=r['begin_ns']/1e9,
            relative_seconds=r['begin_ns']/1e9-roi['left_monotonic'],waiting=r['before']['waiting'],
            running=r['before']['running'],scheduled_requests=r['actual_scheduled_requests'],
            scheduled_tokens=r['scheduled_tokens'],scheduler_sampling_interval_ms=scheduler['sampling_interval_ms'],
            evidence='TRACE_ANALYSIS',scope='Observed pre-schedule state only. No interpolation, time-weighted mean or execution-state claim.') for r in selected)
        axis.set_ylabel(role+' requests');axis.legend(fontsize=8);axis.grid(alpha=.2)
    axes[-1].set_xlabel('Seconds from central arrival p10');axes[-1].set_xlim(0,roi['duration_seconds'])
    fig.suptitle('C25 QPS0.5 | Observed scheduler states only | No state inferred between samples')
    fig.tight_layout();save_figure(fig,f'idle_scope_C25_aligned_queue_timeline_{version}',queue_rows);plt.close(fig)
    audit=ROOT/'analysis'/f'idle_scope_figure_audit_{version}.json';assert not audit.exists()
    write(audit,dict(time=time.time(),contracts=contracts,
        scope='Narrow explicit-flag and three-core-file equivalence only. No full software equivalence claim.',
        same_fingerprint_does_not_match_ROIs=True,qualified_high_load=False,
        original_frozen_reports_and_profiles_unchanged=True,actual_high_load_P_heavy='NOT_CONFIRMED',
        SM_Tensor_HBM='NOT_MEASURED'))
    report=ROOT.parent/f'mllm_epd_idle_scope_and_required_figures_v{int(version[1:])+1}.md'
    report_text='\n'.join(['# E/P/D Idle Figure: Source와 관측 범위 감사','',
        '증거는 실제 Kernel·Copy Timeline의 TRACE_ANALYSIS다. 실패한 계측은 DIAGNOSTIC으로 표시했다.',
        '같은 Manifest·Output Replay를 사용해도 Runtime Source와 관측 구간은 다를 수 있다.',
        '낮은 Busy를 Decode-safe 또는 Encode-safe Window로 해석하지 않았다.','',
        'C25 QPS0.5와 C27 QPS3의 명시적 Engine Flag와 세 핵심 Source Hash는 같다.',
        '비교한 Source는 proxy.py, processor_handoff.py, cuda_ipc_ec.py다. 전체 Software 동등성을 증명한 것은 아니다.',
        'C11 QPS4는 이 Hash 조합이 다르다. 별도 Panel과 Fingerprint로 구분했다.',
        'C25는 예정 Arrival p10–p90 전체를 관측했다. C27은 실제 Short ROI와 그 구간의 교집합만 관측했다.',
        'C27과 C11은 고정한 5% Observer Gate를 통과하지 않았다. 두 점은 Primary GPU 시간으로 사용하지 않는다.',
        'QPS 곡선은 연결하지 않았다. 측정하지 않은 QPS의 값은 보간하지 않았다.',
        'Scheduler Waiting은 Step 관측별 평균이다. 시간 가중 Queue 또는 GPU-exclusive 대기가 아니다.','',
        'Queue Timeline은 실제 Schedule 직전의 Waiting·Running 표본이다. 표본 사이의 상태는 보간하지 않았다.',
        'Schedule의 낙관적 Token Count 갱신을 현재 실행 중인 Decode 수로 해석하지 않았다.',
        'Aligned Figure는 적격 C25 중앙 구간의 Midpoint 전후 2초다. 결과를 보고 유리한 구간을 고르지 않았다.',
        'GPU 간 매우 짧은 선후관계에는 Clock Anchor 오차가 있다. 대응 CSV에 Anchor 폭을 보존했다.',
        '붉은 구간은 Copy 또는 Memset이다. 이를 실제 Network DMA 또는 Prefill 가능 시간으로 대체하지 않았다.',
        'Main400은 그대로 보존했다. 이 Figure는 후속 Main100 실험이다.',
        'SM Active·Tensor Core Activity·HBM Bandwidth는 미측정이다. Kernel Busy와 동일하다고 쓰지 않았다.','',
        f'Figure와 CSV는 figures/idle_scope_*_{version}에 있다. 모든 원시 Profile은 기존 Namespace에 남아 있다.',
        f'Config·원시 집계·Qualification SHA256과 제한된 Fingerprint 입력은 analysis/idle_scope_figure_audit_{version}.json에 있다.',
        '재생성할 때는 --version에 아직 없는 새 버전을 지정한다. 기존 CSV와 보고서는 덮어쓰지 않는다.',''])
    if report.exists():assert report.read_text()==report_text,'Preserve a report that differs from this generator.'
    else:report.write_text(report_text)
    preserve();print('IDLE_SCOPE_FIGURES_AND_SOURCE_COMPARABILITY_AUDITED',flush=True)


if __name__=='__main__':main()
