"""Recalculate real imported-KV suffix costs and label physical fit as offline only."""
import argparse
import csv
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .token_discovery_analysis import table


def suffix_events(folder,trials):
    with (folder/'forward_enclosures.csv').open(newline='') as stream:events=list(csv.DictReader(stream))
    geometry={r['index']:r['requests'] for r in json.loads((folder/'geometry.json').read_text())['rows']}
    real=[];empty=[]
    for row in events:
        requests=geometry[int(row['index'])]
        assert int(row['scheduled_requests'])==len(requests)
        assert int(row['scheduled_tokens'])==sum(requests.values())
        if not requests:
            empty.append(dict(**row,scope='Empty-schedule engine work. Not assigned to a suffix request.'))
            continue
        assert len(requests)==1,'The isolated client contract permits one D request only'
        request_id,count=next(iter(requests.items()))
        matched=[r for r in trials if r['run']+':D' in request_id]
        assert len(matched)==1,(request_id,len(matched))
        trial=matched[0]
        real.append(dict(run=trial['run'],request_id=request_id,scheduled_query_tokens=count,
            actual_scheduled_requests=1,GPU_enclosure_ms=float(row['GPU_enclosure_ms']),
            host_execute_ms=(float(row['host_end_ns'])-float(row['host_begin_ns']))/1e6,
            suffix_tokens=int(trial['suffix_tokens']),input_tokens=int(trial['input_tokens']),
            scope='CUDA forward enclosure, not exclusive GPU kernel or transfer time.'))
    return real,empty


def physical_fit(window_ms,cost_ms):
    durations=np.asarray(window_ms,dtype=float)
    assert cost_ms>0 and durations.size and np.all(durations>0)
    fits=durations>=cost_ms
    return dict(window_count=len(durations),fitting_windows=int(fits.sum()),
        fitting_window_fraction=float(fits.mean()),idle_duration_fraction_in_fitting_windows=float(durations[fits].sum()/durations.sum()),
        evidence='OFFLINE_ORACLE',
        scope='Future physical gap is known. No KV-ready, stream, memory, host or deadline feasibility check. Not safe admission or measured speedup.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='suffix_QUANTA_C41R3')
    args=parser.parse_args();tag=args.tag;preserve()
    cost_path=ROOT/'analysis'/f'{tag}_cost_v0.json';cost=json.loads(cost_path.read_text())
    correctness_path=ROOT/'analysis'/f'{tag}_correctness_v0.json';correct=json.loads(correctness_path.read_text())
    assert correct['passed'] and len(correct['cases'])==15
    assert all(case['strict_pass'] for case in correct['cases'])
    assert (ROOT/'state'/f'{tag}_controller_complete.json').exists()
    with (ROOT/'analysis'/f'{tag}_requests_v0.csv').open(newline='') as stream:requests=list(csv.DictReader(stream))
    assert len(requests)==420
    profiled=[r for r in requests if r['stage']=='profile'];assert len(profiled)==135
    events,empty=suffix_events(ROOT/'raw/gpu_enclosures'/tag/'decode',profiled)
    per_trial=[]
    for trial in profiled:
        selected=[r for r in events if r['run']==trial['run']]
        assert selected
        tokens=sum(r['scheduled_query_tokens'] for r in selected)
        per_trial.append(dict(run=trial['run'],index=int(trial['index']),suffix_tokens=int(trial['suffix_tokens']),
            input_tokens=int(trial['input_tokens']),actual_D_query_tokens=tokens,
            query_work_conserved=tokens==int(trial['suffix_tokens']),actual_D_batch_max=1,
            GPU_forward_enclosure_sum_ms=sum(r['GPU_enclosure_ms'] for r in selected),
            host_execute_sum_ms=sum(r['host_execute_ms'] for r in selected),
            scope='Sum of same-request isolated D forward enclosures. Not exclusive kernel activity.'))
    geometry_pass=all(r['query_work_conserved'] for r in per_trial)
    evidence='ISOLATED_GPU_MEASUREMENT' if cost['observer_gate_passed'] and geometry_pass else 'DIAGNOSTIC'
    summaries=[];fit=[]
    low='ec_LOW_CUPTI_C25_CPC_q0.5_v0';idle_path=ROOT/'analysis'/f'{low}_idle_windows.csv'
    with idle_path.open(newline='') as stream:
        windows=[float(r['duration_ms']) for r in csv.DictReader(stream) if r['ROI']=='central_arrival_window' and r['role']=='decode' and r['kind']=='kernel_idle']
    for index in [0,6,31]:
        for quantum in [128,256,512,1024,2048]:
            selected=[r for r in requests if int(r['index'])==index and int(r['suffix_tokens'])==quantum]
            http={stage:[float(r['D_first_token_HTTP_ms']) for r in selected if r['stage']==stage]
                for stage in ['clean_before','profile','clean_after']}
            assert all(len(v)==9 for v in http.values())
            measured=[r for r in per_trial if r['index']==index and r['suffix_tokens']==quantum];assert len(measured)==9
            row=dict(index=index,input_tokens=measured[0]['input_tokens'],suffix_tokens=quantum,
                prefix_tokens=measured[0]['input_tokens']-quantum,independent_initializations=1,
                clean_before_HTTP_p50_ms=float(np.median(http['clean_before'])),
                clean_before_HTTP_p95_ms=float(np.percentile(http['clean_before'],95)),
                clean_after_HTTP_p50_ms=float(np.median(http['clean_after'])),
                clean_after_HTTP_p95_ms=float(np.percentile(http['clean_after'],95)),
                profile_HTTP_p95_ms=float(np.percentile(http['profile'],95)),
                profile_forward_enclosure_p50_ms=float(np.median([r['GPU_forward_enclosure_sum_ms'] for r in measured])),
                profile_forward_enclosure_p95_ms=float(np.percentile([r['GPU_forward_enclosure_sum_ms'] for r in measured],95)),
                actual_D_batch_max=1,all_query_work_conserved=all(r['query_work_conserved'] for r in measured),
                whole_observer_gate=cost['observer_gate_passed'],evidence=evidence,
                scope='Actual E/P/NIXL/D. D-only nonstreaming one-token completion wall includes import, scheduling, sampling and delivery.')
            summaries.append(row)
            tail=max(row[k] for k in ['clean_before_HTTP_p95_ms','clean_after_HTTP_p95_ms','profile_HTTP_p95_ms'])
            fit.append(dict(index=index,input_tokens=row['input_tokens'],suffix_tokens=quantum,
                cost_reference_ms=tail,cost_evidence=evidence,idle_source=low,**physical_fit(windows,tail)))
    for name,rows in [('summary',summaries),('real_forward_events',events),('request_geometry',per_trial),('physical_gap_fit_oracle',fit)]:
        table(ROOT/'analysis'/f'{tag}_{name}_v0.csv',rows)
    if empty:table(ROOT/'analysis'/f'{tag}_empty_schedule_events_v0.csv',empty)
    write(ROOT/'analysis'/f'{tag}_report_audit_v0.json',dict(time=time.time(),cost_SHA256=digest(cost_path),
        correctness_SHA256=digest(correctness_path),observer_gate=cost['observer_gate_passed'],query_work_conserved=geometry_pass,
        evidence=evidence,empty_schedule_events=len(empty),empty_enclosure_ms=sum(float(r['GPU_enclosure_ms']) for r in empty),
        actual_client_concurrency=1,actual_D_batch_max=1,original_token_order=True,
        imported_KV_transfer_exclusive_ms='NOT_MEASURED',exclusive_suffix_kernel_ms='NOT_MEASURED',
        encode_safe_and_decode_safe_budget='NOT_MEASURED',serving_speedup=False,full_task_quality='PENDING'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));figure_rows=[]
    for index in [0,6,31]:
        selected=[r for r in summaries if r['index']==index];label=f'Input{selected[0]["input_tokens"]}'
        for axis,key,ylabel in zip(axes,['clean_after_HTTP_p95_ms','profile_forward_enclosure_p95_ms'],
                ['Nonstreaming one-token D HTTP p95 (ms)','D forward enclosure p95 (ms)']):
            axis.plot([r['suffix_tokens'] for r in selected],[r[key] for r in selected],marker='o',label=label)
            axis.set_xlabel('Actual suffix query tokens');axis.set_ylabel(ylabel);axis.grid(alpha=.2)
            figure_rows.extend(dict(index=index,input_tokens=r['input_tokens'],suffix_tokens=r['suffix_tokens'],metric=key,value=r[key],
                evidence=evidence,scope='C0 only. Enclosures are not exclusive kernels. Not C3 serving latency.') for r in selected)
    axes[0].legend();fig.suptitle(f'{evidence} | Imported-prefix suffix | N1 initialization | No serving speedup claim')
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{tag}_cost_v0.{extension}',dpi=180)
    plt.close(fig);table(ROOT/'figures'/f'{tag}_cost_v0.csv',figure_rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for index in [0,6,31]:
        selected=[r for r in fit if r['index']==index];label=f'Input{selected[0]["input_tokens"]}'
        for axis,key,title in zip(axes,['fitting_window_fraction','idle_duration_fraction_in_fitting_windows'],
                ['Physical window-count fraction','Physical idle-duration fraction']):
            axis.plot([r['suffix_tokens'] for r in selected],[100*r[key] for r in selected],marker='o',label=label)
            axis.set_xlabel('Actual suffix query tokens');axis.set_ylabel(title+' (%)');axis.set_ylim(0,100)
            axis.grid(alpha=.2)
    axes[0].legend();fig.suptitle('OFFLINE_ORACLE | Future C25 gaps known | No safe-admission or serving-speedup claim')
    fig.tight_layout()
    for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{tag}_physical_gap_fit_oracle_v0.{extension}',dpi=180)
    plt.close(fig);table(ROOT/'figures'/f'{tag}_physical_gap_fit_oracle_v0.csv',fit)
    report=ROOT.parent/'mllm_epd_imported_kv_suffix_quantum_cost_v1.md';assert not report.exists()
    lines=['# 실제 Prefix KV를 가져온 Suffix Quantum 비용','',
        f'전체 증거 수준은 `{evidence}`다. 독립 초기화는 1회다.',
        '15개 요청·Quantum 조합의 Strict 상태 검사를 먼저 통과했다. 이후 Cost 표본을 별도로 측정했다.',
        '각 Cost 요청은 실제 원래 E, P Prefix, NIXL KV Import와 Full28 D Suffix를 실행했다.',
        'Output1 Sample은 폐기했다. Natural EOS나 C3 Online Serving 결과가 아니다.','',
        '## 실제 비용','',
        '| Input | Suffix | Clean Before HTTP p95 ms | Clean After HTTP p95 ms | Profile Forward p95 ms | Query Work Conserved |',
        '|---:|---:|---:|---:|---:|---|']
    for row in summaries:
        lines.append(f'| {row["input_tokens"]} | {row["suffix_tokens"]} | {row["clean_before_HTTP_p95_ms"]:.3f} | '+
            f'{row["clean_after_HTTP_p95_ms"]:.3f} | {row["profile_forward_enclosure_p95_ms"]:.3f} | {row["all_query_work_conserved"]} |')
    lines+=['','D HTTP 구간은 Nonstreaming 한 Token 응답 완료까지다. SSE의 첫 Token TTFT와 정의가 다르다.',
        'KV Import·Embedding 소비·Admission·Forward·Sampling·Delivery가 이 구간에 들어 있다.',
        'Forward Event는 Stream Enclosure다. GPU-exclusive Kernel이나 DMA 시간으로 바꾸지 않는다.',
        '각 Cell의 Clean/Profile/Clean 표본은 단계별 9개다. 반복 9개를 독립 Server Start9회로 계산하지 않았다.',
        '고정한 30개 p50/p95 검사 중 하나라도 실패하면 전체 계측을 Diagnostic으로 유지한다.',
        f'원래 전체 Observer Gate는 {cost["observer_gate_passed"]}다. 실제 Query Conservation은 {geometry_pass}다.',
        f'요청이 없는 정리 Execute {len(empty)}개는 따로 보존했다. 해당 비용을 Suffix에 임의로 배분하지 않았다.','',
        '## Physical Gap 비교의 한계','',
        '별도의 QPS0.5 Main100 D Kernel Gap과 비용을 비교한 값은 `OFFLINE_ORACLE`이다.',
        '미래 Gap 길이를 안다고 가정했다. KV 준비·Stream·메모리·새 Arrival·Deadline을 검사하지 않았다.',
        '비용에는 Prefix KV Import가 포함된다. 실제 미리 준비된 KV에 대한 더 작은 Service 비용이라고 바꾸지 않는다.',
        '이 물리적 Fit Rate로 Decode-safe Window, Active Decode 삽입 또는 실제 Speedup을 주장하지 않는다.',
        'E의 Prefix Probe와 이 D Suffix는 서로 다른 작업이다. E/D가 같은 Suffix를 처리한 성능 비교가 아니다.','',
        '## 재현 계약','',
        'Original BF16 Qwen2.5-VL-7B, BI0, Full28과 원본 Vision/Processor를 유지했다.',
        'Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. GPU4/5/6은 E/P/D다.',
        'B8192/S1024, KV42GiB, Block128과 공통 Compiled Reduction 설정이다.',
        'Calibration Request0/6/31의 원문을 유지했다. Input 길이를 맞추기 위한 Padding이나 Truncation은 하지 않았다.',
        '논리적 Prefix KV Bytes는 Structural Accounting이다. 실제 전송된 Padding·DMA Bytes는 미측정이다.',
        f'원시 요청과 검사 CSV는 `analysis/{tag}_*_v0`에 있다.',
        f'실제 Forward Event는 `raw/gpu_enclosures/{tag}/decode/`에 있다.',
        f'Figure와 대응 CSV는 `figures/{tag}_cost_v0`에 있다.','']
    report.write_text('\n'.join(lines));preserve();print('IMPORTED_PREFIX_SUFFIX_COST_RECALCULATED',flush=True)


if __name__=='__main__':main()
