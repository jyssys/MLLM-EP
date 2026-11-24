"""Recalculate genuine whole-request serving without inventing a P stage."""
import argparse
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .metrics import gates
from .native_common_pilot_report import arrival_contract_SHA256
from .quality_audit import scorers,score_request
from .token_confirmation_analysis import paired_interval,mean_interval,CATEGORIES
from .token_discovery_analysis import table

METHODS=['NativeStrong','NativeOnDemand','WholeDeflection']
KEYS=['TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','ITL_max_ms','TPOT_p95_ms',
    'raw_output_tokens_per_second','total_input_output_tokens_per_second','completed_requests_per_second',
    'SLO_goodput','joint_percent','decode_pass_percent']
COMMON_STAGES=['arrival_to_proxy','proxy_before_E','E_HTTP_enclosure','after_E_to_D_submit','D_submit_to_first']


def host_path(request,trace):
    first=request['token_times'][0]
    boundaries=[request['scheduled_arrival'],trace['received'],trace['E_submit'],trace['E_complete'],trace['D_submit'],first]
    result={key:1000*(b-a) for key,a,b in zip(COMMON_STAGES,boundaries,boundaries[1:])}
    assert all(v>=-1e-5 for v in result.values())
    assert abs(sum(result.values())-request['TTFT_ms'])<1e-5
    selected=trace.get('deflected',False)
    if selected:
        assert 'P_submit' not in trace and 'P_ready' not in trace
        assert trace['P_stage'].startswith('NOT_APPLICABLE') and trace['KV_handoff'].startswith('NONE')
        assert trace['D_admission']['extra_prefill']
        assert trace['D_admission']['before']['active_http_requests']==0
        prefix=None;source_wait=None;P_HTTP=None
    else:
        assert trace['P_ready']>=trace['P_submit']>=trace['E_complete']
        prefix=1000*(trace['P_submit']-trace['E_complete'])
        P_HTTP=1000*(trace['P_ready']-trace['P_submit'])
        source_wait=1000*(trace['D_submit']-trace['P_ready'])
    gate=trace.get('D_admission')
    return dict(**result,deflected=selected,P_stage='NOT_APPLICABLE' if selected else 'FULL_PREFILL',
        E_complete_to_P_submit_ms=prefix,P_HTTP_enclosure_ms=P_HTTP,P_ready_to_D_submit_ms=source_wait,
        D_coordinator_wait_ms=1000*(gate['admitted']-gate['begin']) if gate else None,
        scope='First five fields form a disjoint host TTFT partition. Detail fields overlap that partition. Not exclusive GPU costs.')


def decision(pairs,starts,subset,mode):
    def value(reference,key):
        return next(r for r in pairs if r['subset']==subset and r['mode']==mode and r['reference']==reference and r['metric']==key)
    controls=['NativeStrong','NativeOnDemand']
    decode=all(value(control,key)['paired_start_bootstrap95_CI'][1]<=1.05 for control in controls
        for key in ['ITL_p99_ms','TPOT_p95_ms'])
    median=all(value(control,'TTFT_p50_ms')['paired_start_bootstrap95_CI'][1]<=1.05 for control in controls)
    chosen=[r for r in starts if r['subset']==subset and r['mode']==mode and r['method']=='WholeDeflection']
    assert len(chosen)==3
    absolute=all(r['decode_pass_percent']>=95 for r in chosen)
    gains={}
    for control in controls:
        latency=value(control,'TTFT_p95_ms');goodput=value(control,'SLO_goodput')
        gains[control]=bool(latency['ratio_of_start_means']<=.95 and latency['paired_start_bootstrap95_CI'][1]<1 or
            goodput['ratio_of_start_means']>=1.05 and goodput['paired_start_bootstrap95_CI'][0]>1)
    collateral=[dict(reference=control,metric=metric,ratio=value(control,metric)['ratio_of_start_means'])
        for control in controls for metric in ['raw_output_tokens_per_second','SLO_goodput','joint_percent']
        if value(control,metric)['ratio_of_start_means']<1]
    passed=decode and median and absolute and all(gains.values())
    return dict(subset=subset,mode=mode,relative_decode_gate=decode,TTFT_p50_gate=median,
        absolute_decode_all_starts=absolute,minimum_5percent_gain_vs_each_native=gains,
        observed_collateral_point_decreases=collateral,
        runtime_decision='RUNTIME_THRESHOLD_PASS_PENDING_QUALITY_AND_FULL_SERVING_REVIEW' if passed else 'NO_GO_ON_TESTED_RUNTIME',
        full_task_quality='PENDING',full_GO=False,
        scope='Actual whole-request control, not a Generic Token Split or modality-aware serving success. All collateral metrics remain visible.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='whole_DEFLECTION_C43R4');args=parser.parse_args()
    tag=args.tag;preserve();source=ROOT/'analysis'/f'{tag}_summary.json';summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE' and len(summary['results'])==36
    assert (ROOT/'state'/f'{tag}_controller_complete.json').exists()
    parent=json.loads((ROOT/'data/main_manifest_template_v2.json').read_text())
    lookup={r['request_id']:r for r in parent['requests']}
    counts={r['request_id']:r['output_tokens'] for r in map(json.loads,
        (ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl').read_text().splitlines())}
    scoring=scorers();starts=[];paths=[];failures=[];quality=[];choices=[];raw={};arrivals={}
    for cell in summary['results']:
        run=cell['run'];method=cell['method'];subset=cell['subset'];mode=cell['mode'];metric=cell['metrics']
        block=int(run.split('_B',1)[1].split('_',1)[0]);stem=f'{tag}_B{block}_{method}'
        file=ROOT/'raw'/f'{run}_requests.jsonl';requests=[json.loads(line) for line in file.read_text().splitlines()]
        expected=100 if subset=='Main100' else 50
        assert len(requests)==expected and all(r['status']=='MEASURED_ONLINE' for r in requests)
        assert digest(file)==metric['raw_SHA256']
        assert sum(r['output_tokens'] for r in requests)==metric['total_output_tokens']
        assert all(r['output_tokens']==len(r['token_ids'])==r['usage']['completion_tokens'] for r in requests)
        assert all(r['prompt_SHA256']==lookup[r['request_id']]['prompt_SHA256'] for r in requests)
        assert np.percentile([r['TTFT_ms'] for r in requests],95)==metric['TTFT_p95_ms']
        assert np.percentile([v for r in requests for v in r['ITL_ms']],99)==metric['ITL_p99_ms']
        assert np.percentile([r['TPOT_ms'] for r in requests],95)==metric['TPOT_p95_ms']
        assert sum(r['output_tokens'] for r in requests if gates(r)['JOINT_PASS'])/metric['duration_seconds']==metric['SLO_goodput']
        if mode=='replay':assert all(r['output_tokens']==counts[r['request_id']] for r in requests)
        arrivals.setdefault(subset,set()).add(arrival_contract_SHA256(ROOT/'configs'/f'{run}_arrival.json'))
        row=dict(block=block,method=method,subset=subset,mode=mode,run=run,requests=expected,
            output_tokens=metric['total_output_tokens'],duration_seconds=metric['duration_seconds'],
            **{key:metric[key] for key in KEYS if key in metric},joint_percent=100*metric['gate_attainment']['JOINT_PASS'],
            decode_pass_percent=100*sum(gates(r)['ITL_PASS'] and gates(r)['TPOT_PASS'] for r in requests)/expected,
            evidence='MEASURED_ONLINE runtime; full task quality PENDING; actual clean GPU batch NOT_MEASURED')
        starts.append(row);raw[(block,method,subset,mode)]=requests
        proxy_name=stem+'_proxy' if method=='NativeStrong' else stem+'_whole_proxy'
        proxy=ROOT/'raw'/f'proxy_{proxy_name}.jsonl'
        traces={r['runtime_request_id']:r for r in map(json.loads,proxy.read_text().splitlines())}
        for request in requests:
            trace=traces[request['runtime_request_id']];assert trace['status']=='MEASURED_ONLINE'
            stage=host_path(request,trace)
            paths.append(dict(block=block,method=method,subset=subset,mode=mode,request_id=request['request_id'],
                TTFT_ms=request['TTFT_ms'],input_tokens=request['usage']['prompt_tokens'],output_tokens=request['output_tokens'],**stage))
            if method!='NativeStrong':
                chose=trace['decision']
                assert chose['after_encode'] and chose['selected']==stage['deflected']
                if method=='NativeOnDemand':assert not stage['deflected']
                choices.append(dict(block=block,method=method,subset=subset,mode=mode,request_id=request['request_id'],
                    selected=stage['deflected'],P_HTTP_outstanding=chose['P_HTTP_outstanding'],D_HTTP_outstanding=chose['D_HTTP_outstanding'],
                    full_prompt_tokens_on_D=request['usage']['prompt_tokens'] if stage['deflected'] else None,
                    scope='Current-state full-request choice after actual E. HTTP pressure is not exact GPU queue or future slack.'))
            if mode=='natural':
                result=score_request(lookup[request['request_id']],request['text'],scoring)
                quality.append(dict(block=block,method=method,subset=subset,request_id=request['request_id'],source=request['source'],
                    score=result['score'],metric=result['metric'],complete_task_scorer=result['complete_task_scorer'],full_quality='PENDING'))
        for category in CATEGORIES:
            selected=[r for r in requests if gates(r)['category']==category]
            failures.append(dict(block=block,method=method,subset=subset,mode=mode,category=category,
                requests=len(selected),request_denominator=expected,output_tokens=sum(r['output_tokens'] for r in selected),
                total_output_tokens=metric['total_output_tokens']))
    assert all(len(value)==1 for value in arrivals.values())
    pairs=[];statistics=[]
    for subset in ['Main100','FreshHoldout']:
        for mode in ['natural','replay']:
            for method in METHODS:
                selected=sorted([r for r in starts if r['subset']==subset and r['mode']==mode and r['method']==method],key=lambda r:r['block'])
                assert len(selected)==3
                statistics.extend(dict(subset=subset,mode=mode,method=method,metric=key,
                    **mean_interval([r[key] for r in selected])) for key in KEYS)
            actual=sorted([r for r in starts if (r['subset'],r['mode'],r['method'])==(subset,mode,'WholeDeflection')],key=lambda r:r['block'])
            for reference in ['NativeStrong','NativeOnDemand']:
                control=sorted([r for r in starts if (r['subset'],r['mode'],r['method'])==(subset,mode,reference)],key=lambda r:r['block'])
                for key in KEYS:pairs.append(dict(subset=subset,mode=mode,reference=reference,metric=key,
                    **paired_interval([r[key] for r in actual],[r[key] for r in control])))
    decisions=[decision(pairs,starts,subset,mode) for subset in ['Main100','FreshHoldout'] for mode in ['natural','replay']]
    for label,rows in [('starts',starts),('host_paths',paths),('failure_categories',failures),('quality',quality),
            ('choices',choices),('paired_ratios',pairs),('start_mean_CI',statistics)]:
        table(ROOT/'analysis'/f'{tag}_{label}_v0.csv',rows)
    write(ROOT/'analysis'/f'{tag}_audit_v0.json',dict(time=time.time(),source_SHA256=digest(source),
        decisions=decisions,arrivals={key:next(iter(value)) for key,value in arrivals.items()},
        full_task_quality='PENDING',full_GO=False,actual_clean_GPU_batch='NOT_MEASURED',
        metric_recalculation='PASS',host_partition='Five common nonoverlapping fields; branch detail fields are separate and may overlap.',
        absent_P_stage='NOT_APPLICABLE and null durations, never fabricated zero-ms measurements'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for subset in ['Main100','FreshHoldout']:
        for key,label in [('TTFT_p95_ms','TTFT p95 (ms)'),('ITL_p99_ms','Client ITL p99 (ms)'),
                ('TPOT_p95_ms','Request TPOT p95 (ms/token)'),('raw_output_tokens_per_second','Raw output tokens/s'),
                ('SLO_goodput','SLO-qualified output tokens/s'),('joint_percent','Joint SLO attainment (%)')]:
            fig,axes=plt.subplots(1,2,figsize=(11,4));figure_rows=[]
            for axis,mode in zip(axes,['natural','replay']):
                for index,method in enumerate(METHODS):
                    stat=next(r for r in statistics if (r['subset'],r['mode'],r['method'],r['metric'])==(subset,mode,method,key))
                    lo,hi=stat['CI'];center=stat['mean']
                    axis.errorbar(index,center,yerr=[[center-lo],[hi-center]],fmt='o',capsize=4)
                    figure_rows.append(dict(subset=subset,mode=mode,method=method,metric=key,start_mean=center,CI_low=lo,CI_high=hi,
                        independent_starts=3,scope='Actual whole-request C3 control. Full task quality pending. Not token-split speedup.'))
                axis.set_xticks(range(3),METHODS,rotation=15);axis.set_ylabel(label);axis.set_title(mode);axis.grid(alpha=.2)
            fig.suptitle(f'{subset} | QPS2 | Independent N3 | Runtime only, quality pending')
            fig.tight_layout();name=f'{tag}_{subset}_{key}_v0'
            for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
            plt.close(fig);table(ROOT/'figures'/f'{name}.csv',figure_rows)
    report=ROOT.parent/'mllm_epd_whole_deflection_online_serving_v1.md';assert not report.exists()
    lines=['# Whole-request Deflection — 독립 N3 C3 Serving','',
        '실제 Online E와 Continuous P/D를 사용했다. Main100과 새 Context Holdout50을 별도로 평가했다.',
        'Natural EOS와 동일 Output 수의 Replay를 분리했다. 각 Method의 독립 초기화는 3회다.',
        '전체 Task Quality는 PENDING이다. 아래 Runtime Threshold 판정을 Full GO로 바꾸지 않는다.','',
        '## 모든 Start 결과','',
        '| Subset | Output | Method | Start | Tokens | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Goodput tok/s | Joint % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in starts:
        lines.append(f'| {row["subset"]} | {row["mode"]} | {row["method"]} | {row["block"]} | {row["output_tokens"]} | '+
            ' | '.join(f'{row[key]:.3f}' for key in ['TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput','joint_percent'])+' |')
    lines+=['','## 고정 Gate 판정','',
        '| Subset | Output | Relative Decode | Absolute Decode | TTFT p50 | Runtime Decision |',
        '|---|---|---|---|---|---|']
    for item in decisions:lines.append(f'| {item["subset"]} | {item["mode"]} | {item["relative_decode_gate"]} | '+
        f'{item["absolute_decode_all_starts"]} | {item["TTFT_p50_gate"]} | {item["runtime_decision"]} |')
    lines+=['','NativeStrong은 원래 Inline E Transport와 Stock Compiler를 유지했다.',
        'NativeOnDemand는 후보와 같은 On-demand E Transport·Coordinator·공통 Compiler를 사용하되 Deflection하지 않았다.',
        '두 Native를 같은 실행 정책이라고 부르지 않는다. 후보는 두 기준에 대해 각각 Decode와 TTFT p50 Gate를 검사했다.',
        '선택은 실제 E 완료 후 현재 P HTTP 작업과 D Stream 상태를 사용했다. 미래 EOS나 Idle Duration은 사용하지 않았다.',
        'D Full Prefill은 이전 D Stream이 없을 때만 시작했다. 새 Native D는 첫 실제 생성 Token 이후에 진입할 수 있다.',
        '이 상태 보호만으로 모든 후속 Decode의 비열등성을 가정하지 않았다. 실제 ITL/TPOT와 모든-ITL 실패를 검사했다.',
        '필수 SLO는 TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms/token이다.',
        'Absolute Decode는 Request별 All-ITL과 TPOT의 AND다. 모든 Start에서 95% 이상이어야 한다.',
        'Paired CI는 같은 Block의 독립 Start3쌍을 사용했다. 요청을 독립 Start로 취급하지 않았다.',
        '모든 27개 Empirical Resample을 사용했다. N3 CI 해상도는 제한적이다.',
        '다른 주요 지표의 Point 감소도 Audit에 남겼다. TTFT 개선으로 Decode나 Joint SLO 악화를 상쇄하지 않는다.','',
        '## 경로·품질·적용 한계','',
        'Deflected 요청은 P Forward와 P Prefix KV Import가 없다. 해당 P 시간은 NOT_APPLICABLE로 남겼다.',
        '다섯 공통 Host 구간은 TTFT와 정확히 합산된다. 세부 P 구간은 공통 구간과 겹치므로 다시 더하지 않는다.',
        'Host Enclosure를 GPU-exclusive Forward나 DMA 시간으로 해석하지 않는다.',
        'Whole-request의 성공이나 실패는 Generic Token Split의 결과가 아니다. Kairos 재현이나 새로운 Modality-aware Method도 아니다.',
        '조건부 Same-E N1+N8에서 28개 KV Layer와 전체 Greedy 출력을 검사했다. C3의 다른 Batch에서 같은 출력을 보장하지는 않는다.',
        'MMLongBench-Doc 공식 외부 Answer Extraction은 미실행이다. Raw Task Score를 전체 품질 PASS로 바꾸지 않았다.',
        'FreshHoldout은 Parent400 중 Main100·Calibration 이미지/문서와 겹치지 않는 새 50개다.',
        '원래 Single-GPU Natural 길이 Profiling은 존재한다. 이번 C3 방법 비교에는 새 Context다.',
        'Main100이나 이전 Holdout을 성능 결과에 따라 교체하지 않았다.',
        '실제 Clean GPU Batch는 미측정이다. HTTP Outstanding 또는 B/S를 Actual GPU Batch로 대체하지 않았다.','',
        '## 재현 자료','',
        'Original BF16 Qwen2.5-VL-7B Revision cc594898137f460bfe9f0759e9844b3ce807cfb5를 사용했다.',
        'GPU4/5/6, B8192/S1024, KV42GiB, Block128, Prefix Cache Off, 원래 Vision/Processor를 유지했다.',
        f'Config는 `configs/{tag}.json`이다. Dataset Hash·실행 순서·Arrival은 결과 전에 동결했다.',
        f'CSV와 Audit은 `analysis/{tag}_*_v0`에 있다. Figure와 대응 CSV는 `figures/{tag}_*_v0`에 있다.',
        '모든 원시 실패와 세부 Request Token 수를 보존했다. Finite-cohort Throughput을 Sustainable Capacity로 부르지 않는다.','']
    report.write_text('\n'.join(lines));preserve();print('WHOLE_N3_RUNTIME_RECALCULATED',flush=True)


if __name__=='__main__':main()
