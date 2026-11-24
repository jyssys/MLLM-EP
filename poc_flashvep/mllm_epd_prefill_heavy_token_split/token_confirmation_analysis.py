"""Recalculate complete N3 cohorts with paired start-level uncertainty and all SLO failures."""
import itertools
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .metrics import gates
from .native_common_pilot_report import arrival_contract_SHA256
from .quality_audit import scorers,score_request
from .token_confirmation_control import TAG,METHODS
from .token_discovery_analysis import host_stages,STAGES,table

KEYS=['TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','ITL_max_ms',
    'TPOT_p95_ms','raw_output_tokens_per_second','completed_requests_per_second',
    'total_input_output_tokens_per_second','SLO_goodput','joint_percent','ITL_pass_percent','TPOT_pass_percent']
CATEGORIES=['JOINT_PASS','TTFT_FAIL','ITL_FAIL','TPOT_FAIL','TTFT+ITL_FAIL',
    'TTFT+TPOT_FAIL','ITL+TPOT_FAIL','TTFT+ITL+TPOT_FAIL']


def paired_interval(actual,reference):
    actual=np.asarray(actual,dtype=float);reference=np.asarray(reference,dtype=float)
    assert actual.shape==reference.shape==(3,) and np.all(reference>0)
    samples=[float(actual[list(i)].mean()/reference[list(i)].mean())
        for i in itertools.product(range(3),repeat=3)]
    return dict(ratio_of_start_means=float(actual.mean()/reference.mean()),
        paired_start_bootstrap95_CI=np.percentile(samples,[2.5,97.5]).tolist(),
        independent_start_pairs=3,empirical_resamples=27,
        caveat='All27 empirical paired resamples. N3 gives limited uncertainty resolution. Requests are not independent starts.')


def mean_interval(values):
    values=np.asarray(values,dtype=float);assert values.shape==(3,)
    samples=[float(values[list(i)].mean()) for i in itertools.product(range(3),repeat=3)]
    return dict(mean=float(values.mean()),CI=np.percentile(samples,[2.5,97.5]).tolist())


def runtime_decision(pairs,starts,method,subset,mode):
    def result(baseline,key):
        return next(r for r in pairs if r['method']==method and r['baseline']==baseline and
            r['subset']==subset and r['mode']==mode and r['metric']==key)
    controls=['NativeStrong','NativeCommon']
    decode=all(result(b,k)['paired_start_bootstrap95_CI'][1]<=1.05 for b in controls
        for k in ['ITL_p99_ms','TPOT_p95_ms'])
    median=all(result(b,'TTFT_p50_ms')['paired_start_bootstrap95_CI'][1]<=1.05 for b in controls)
    chosen=[r for r in starts if r['method']==method and r['subset']==subset and r['mode']==mode]
    absolute=all(r['ITL_pass_percent']>=95 and r['TPOT_pass_percent']>=95 for r in chosen)
    gains={}
    for baseline in controls:
        latency=result(baseline,'TTFT_p95_ms');goodput=result(baseline,'SLO_goodput')
        gains[baseline]=bool(
            latency['ratio_of_start_means']<=.95 and latency['paired_start_bootstrap95_CI'][1]<1 or
            goodput['ratio_of_start_means']>=1.05 and goodput['paired_start_bootstrap95_CI'][0]>1)
    supported=decode and median and absolute and all(gains.values())
    return dict(method=method,subset=subset,mode=mode,decode_relative_gate=decode,
        TTFT_p50_gate=median,absolute_decode_all_starts=absolute,
        minimum_5percent_gain_vs_each_native=gains,
        runtime_decision='RUNTIME_CONDITIONAL_QUALITY_PENDING' if supported else 'NO_GO_ON_TESTED_RUNTIME',
        full_task_quality='PENDING',full_GO=False,
        caveat='Runtime criteria only. Natural/replay and Main/Heldout remain separate. No final GO without quality and other required gates.')


def main():
    preserve();source=ROOT/'analysis'/f'{TAG}_summary.json';summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE' and len(summary['cells'])==48
    manifest=ROOT/'data/main100_manifest_template_v2.json';data=json.loads(manifest.read_text())
    lookup={r['request_id']:r for r in data['requests']}
    reference_raw=ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    reference_outputs={r['request_id']:r for r in map(json.loads,reference_raw.read_text().splitlines())}
    scoring=scorers();starts=[];failures=[];paths=[];quality=[];decisions=[];request_rows=[];raw={};arrivals={}
    for cell in summary['cells']:
        run=cell['run'];method=cell['method'];subset=cell['subset'];mode=cell['mode'];metric=cell['metrics']
        block=int(run.split('_B',1)[1].split('_',1)[0])
        path=ROOT/'raw'/f'{run}_requests.jsonl';requests=[json.loads(line) for line in path.read_text().splitlines()]
        expected=100 if subset=='Main100' else 50
        assert len(requests)==expected and all(r['status']=='MEASURED_ONLINE' for r in requests)
        assert digest(path)==metric['raw_SHA256']
        assert all(len(r['token_ids'])==r['output_tokens']==r['usage']['completion_tokens'] for r in requests)
        assert sum(r['output_tokens'] for r in requests)==metric['total_output_tokens']
        assert np.percentile([r['TTFT_ms'] for r in requests],95)==metric['TTFT_p95_ms']
        assert np.percentile([v for r in requests for v in r['ITL_ms']],99)==metric['ITL_p99_ms']
        assert np.percentile([r['TPOT_ms'] for r in requests],95)==metric['TPOT_p95_ms']
        assert sum(r['output_tokens'] for r in requests if gates(r)['JOINT_PASS'])/metric['duration_seconds']==metric['SLO_goodput']
        if mode=='replay':assert all(r['output_tokens']==reference_outputs[r['request_id']]['output_tokens'] for r in requests)
        arrival=arrival_contract_SHA256(ROOT/'configs'/f'{run}_arrival.json')
        arrivals.setdefault((subset,mode),set()).add(arrival)
        row=dict(block=block,method=method,subset=subset,mode=mode,run=run,
            count=expected,output_tokens=metric['total_output_tokens'],duration_seconds=metric['duration_seconds'],
            **{k:metric[k] for k in KEYS if k in metric},
            joint_percent=100*metric['gate_attainment']['JOINT_PASS'],
            ITL_pass_percent=100*metric['gate_attainment']['ITL_PASS'],
            TPOT_pass_percent=100*metric['gate_attainment']['TPOT_PASS'],
            raw_SHA256=digest(path),arrival_contract_SHA256=arrival,
            evidence='MEASURED_ONLINE runtime. Full task quality PENDING. Actual clean GPU batch NOT_MEASURED.')
        starts.append(row);raw[(block,method,subset,mode)]=requests
        proxy=ROOT/'raw'/f'proxy_{TAG}_B{block}_{method}_proxy.jsonl'
        proxy_rows={p['runtime_request_id']:p for p in map(json.loads,proxy.read_text().splitlines())}
        for request in requests:
            p=proxy_rows[request['runtime_request_id']];stage=host_stages(request,p)
            paths.append(dict(block=block,method=method,subset=subset,mode=mode,request_id=request['request_id'],
                **stage,TTFT_ms=request['TTFT_ms'],scope='Disjoint HTTP host boundaries, not GPU-exclusive costs.'))
            gate=gates(request)
            request_rows.append(dict(block=block,method=method,subset=subset,mode=mode,request_id=request['request_id'],
                source=request['source'],arrival=request['scheduled_arrival'],input_tokens=request['usage']['prompt_tokens'],
                output_tokens=request['output_tokens'],TTFT_ms=request['TTFT_ms'],
                max_ITL_ms=max(request['ITL_ms'],default=0),TPOT_ms=request['TPOT_ms'],**gate))
            if mode=='natural':
                value=score_request(lookup[request['request_id']],request['text'],scoring)
                quality.append(dict(block=block,method=method,subset=subset,request_id=request['request_id'],
                    source=request['source'],score=value['score'],complete_task_scorer=value['complete_task_scorer'],
                    raw_reference_tokens_strict=request['token_ids']==reference_outputs[request['request_id']]['token_ids'],
                    metric=value['metric'],full_quality='PENDING'))
            if method in ['Static10','Idle10']:
                decision=p['token_decision'];admission=p['D_admission']
                if p['token_enabled']:
                    assert admission['extra_prefill'] and admission['before']['active_http_requests']==0
                partition=p.get('token_partition') or {}
                decisions.append(dict(block=block,method=method,subset=subset,mode=mode,request_id=request['request_id'],
                    selected=p['token_enabled'],P_HTTP_outstanding=decision['P_HTTP_outstanding'],
                    D_HTTP_outstanding=decision['D_HTTP_outstanding'],
                    coordinator_D_wait_ms=stage['D_coordinator_gate_wait'],
                    offloaded_prompt_tokens=partition.get('suffix_tokens',0),
                    scope='Current-state conservative admission. Not a predicted GPU safe window.'))
        for category in CATEGORIES:
            selected=[r for r in requests if gates(r)['category']==category]
            failures.append(dict(block=block,method=method,subset=subset,mode=mode,category=category,
                requests=len(selected),request_denominator=expected,output_tokens=sum(r['output_tokens'] for r in selected),
                total_output_tokens=metric['total_output_tokens'],duration_seconds=metric['duration_seconds']))
    assert all(len(v)==1 for v in arrivals.values())
    pairs=[];statistics=[]
    for subset in ['Main100','Heldout50']:
        for mode in ['natural','replay']:
            for method in METHODS:
                chosen=sorted([r for r in starts if r['method']==method and r['subset']==subset and r['mode']==mode],key=lambda r:r['block'])
                assert len(chosen)==3
                statistics.extend(dict(method=method,subset=subset,mode=mode,metric=k,**mean_interval([r[k] for r in chosen])) for k in KEYS)
                if method not in ['Static10','Idle10']:continue
                for baseline in ['NativeStrong','NativeCommon']:
                    control=sorted([r for r in starts if r['method']==baseline and r['subset']==subset and r['mode']==mode],key=lambda r:r['block'])
                    for key in KEYS:
                        pairs.append(dict(method=method,baseline=baseline,subset=subset,mode=mode,metric=key,
                            **paired_interval([r[key] for r in chosen],[r[key] for r in control])))
    decisions_runtime=[runtime_decision(pairs,starts,method,subset,mode)
        for method in ['Static10','Idle10'] for subset in ['Main100','Heldout50'] for mode in ['natural','replay']]
    for name,rows in [('starts',starts),('failure_categories',failures),('request_host_paths',paths),
            ('request_SLO',request_rows),('raw_quality',quality),('decisions',decisions),('paired_ratios',pairs),('start_mean_CI',statistics)]:
        table(ROOT/'analysis'/f'token_confirmation_C37_{name}_v0.csv',rows)
    audit=ROOT/'analysis/token_confirmation_C37_audit_v0.json'
    write(audit,dict(time=time.time(),status='N3_RUNTIME_RECALCULATED',manifest_SHA256=digest(manifest),
        summary_SHA256=digest(source),starts=starts,paired_ratios=pairs,start_mean_CI=statistics,runtime_decisions=decisions_runtime,
        quality='PENDING',full_GO='HOLD',observer='No tensor capture or GPU profiling in timed cohorts',
        Actual_batch='NOT_MEASURED in clean runs. Do not substitute an earlier calibration batch.',
        interpretation='No result changes the original SLO, frozen fraction, Main100 split or full quality requirement.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_metrics=[('TTFT_p50_ms','TTFT p50 (ms)'),('TTFT_p95_ms','TTFT p95 (ms)'),
        ('ITL_p99_ms','Client ITL p99 (ms)'),('TPOT_p95_ms','Request TPOT p95 (ms/token)'),
        ('raw_output_tokens_per_second','Raw output tokens/s'),('SLO_goodput','SLO-qualified output tokens/s'),
        ('joint_percent','Joint SLO attainment (%)')]
    for subset in ['Main100','Heldout50']:
        for key,label in figure_metrics:
            fig,axes=plt.subplots(1,2,figsize=(11,4));figure_rows=[]
            for axis,mode in zip(axes,['natural','replay']):
                for x,method in enumerate(METHODS):
                    stat=next(r for r in statistics if r['method']==method and r['subset']==subset and r['mode']==mode and r['metric']==key)
                    low,high=stat['CI'];axis.errorbar(x,stat['mean'],yerr=[[stat['mean']-low],[high-stat['mean']]],fmt='o',capsize=4)
                    figure_rows.append(dict(subset=subset,mode=mode,method=method,metric=key,start_mean=stat['mean'],CI_low=low,CI_high=high))
                axis.set_xticks(range(4),METHODS,rotation=15);axis.set_title(mode);axis.set_ylabel(label);axis.grid(alpha=.2)
            fig.suptitle(f'{subset} | QPS2 | Independent N3 | Runtime only, full quality pending')
            fig.tight_layout();name=f'token_confirmation_C37_{subset}_{key}_v0'
            for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
            plt.close(fig);table(ROOT/'figures'/f'{name}.csv',figure_rows)
        fig,axes=plt.subplots(1,2,figsize=(11,4));cdf_rows=[]
        for axis,mode in zip(axes,['natural','replay']):
            for method in METHODS:
                values=sorted(r['TTFT_ms'] for block in range(3) for r in raw[(block,method,subset,mode)])
                probability=np.arange(1,len(values)+1)/len(values)
                axis.step(values,probability,where='post',label=method)
                cdf_rows.extend(dict(method=method,subset=subset,mode=mode,TTFT_ms=float(value),CDF=float(p),
                    requests_pooled=len(values),independent_starts=3,scope='Descriptive pooled ECDF. No request-level CI.')
                    for value,p in zip(values,probability))
            axis.axvline(5000,color='gray',linestyle='--',label='TTFT SLO5s')
            axis.set_xlabel('TTFT (ms)');axis.set_ylabel('Request CDF');axis.set_title(mode);axis.grid(alpha=.2)
        axes[0].legend(fontsize=8);fig.suptitle(f'{subset} | QPS2 | Pooled descriptive CDF, not independent request inference')
        fig.tight_layout();name=f'token_confirmation_C37_{subset}_TTFT_CDF_v0'
        for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
        plt.close(fig);table(ROOT/'figures'/f'{name}.csv',cdf_rows)
    lines=['# Protected Token Split — 독립 N3 Runtime 확인','',
        '실제 Online E/P/D의 Natural EOS와 Equal-output Replay를 별도로 재계산했다.',
        '각 Method는 GPU4/5/6에서 독립 서버 초기화 3회로 실행했다. 요청을 독립 Start로 취급하지 않았다.',
        '전체 Task Quality는 PENDING이다. 아래 Runtime 결과만으로 최종 GO를 선언하지 않는다.','',
        '## 계약','',
        'Qwen2.5-VL-7B, Original BF16, BI0, 원본 Vision/Processor, Full28을 유지했다.',
        'Revision cc594898137f460bfe9f0759e9844b3ce807cfb5, B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.',
        'Main100 Manifest SHA256은 '+digest(manifest)+'다.',
        'NativeStrong은 Stock Compiled다. NativeCommon과 후보는 공통 Deterministic Reduction 설정을 사용했다.',
        '두 Native는 원래 NIXL과 기본 Admission을 유지했다. Token Adapter나 인위적인 D Cap을 추가하지 않았다.',
        '후보는 기존 D HTTP Stream이 없을 때만 추가 Suffix를 시작했다.',
        '이 정책은 미래 Decode-safe Window 또는 Dynamic Split-size Predictor가 아니다.',
        'Main100은 Discovery50과 Heldout50을 포함한다. Heldout50은 별도의 고정 Arrival로도 검사했다.',
        '두 Split의 Source 분포는 불균형하다. Main100 전체 대표성이 각 Split의 대표성을 뜻하지 않는다.',
        '모든 비율, 순서, Arrival Seed와 SLO는 Heldout 결과 전에 고정했다.',
        '측정 중 GPU Profiler와 CPU Tensor Capture는 껐다. 실제 Clean GPU Batch는 NOT_MEASURED다.',
        '같은 B/S나 Continuous Scheduler 이름을 같은 Actual Batch로 해석하지 않았다.','',
        '## Start별 모든 결과','',
        '| Subset | Output | Method | Start | Tokens | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | Goodput tok/s | Joint % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in starts:
        lines.append(f'| {r["subset"]} | {r["mode"]} | {r["method"]} | {r["block"]} | {r["output_tokens"]} | '+
            ' | '.join(f'{r[k]:.3f}' for k in ['TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_goodput','joint_percent'])+' |')
    lines+=['','## Runtime Gate 판정','',
        '| Subset | Output | Method | Relative Decode | Absolute Decode | TTFT p50 | Runtime Decision |',
        '|---|---|---|---|---|---|---|']
    for d in decisions_runtime:
        lines.append(f'| {d["subset"]} | {d["mode"]} | {d["method"]} | {d["decode_relative_gate"]} | '
            f'{d["absolute_decode_all_starts"]} | {d["TTFT_p50_gate"]} | {d["runtime_decision"]} |')
    lines+=['',
        'Runtime의 개선 조건은 TTFT p95 최소5% 감소 또는 Goodput 최소5% 증가와 우위 방향의 Paired CI다.',
        '두 Native 기준, 상대 Decode, TTFT p50 비열등성 및 Absolute Decode를 각각 검사했다.',
        '품질 PENDING을 Runtime 실패의 대체 설명으로 사용하지 않았다. Runtime 실패와 품질 미확정을 분리했다.',
        'CDF는 같은 크기의 독립 Start 요청을 합친 기술통계다. CI의 독립 단위는 여전히 Start다.',
        '', '## Paired CI와 한계','',
        '각 Paired Ratio는 Start-mean의 비율이다. 같은 Block의 Baseline과 후보를 함께 Resampling했다.',
        'N3의 가능한 27개 Empirical Bootstrap 조합을 모두 사용했다. 95% CI의 해상도는 제한적이다.',
        'ITL p99와 TPOT p95의 상대 1.05 Gate는 NativeStrong 및 NativeCommon에 각각 적용해야 한다.',
        'Absolute Gate는 모든 Client ITL≤34ms 및 Request TPOT≤33ms/token이다.',
        'TTFT≤5000ms와 두 Decode Gate의 AND가 Joint다. 모든 실패 8개 조합의 요청 수와 Token 수를 저장했다.',
        'Goodput 분모는 첫 예정 Arrival부터 Drain까지다. Finite-cohort Throughput은 Sustainable Capacity가 아니다.',
        'Natural Output 수는 실제 생성값이다. Replay는 요청별 Token 수만 같고 Answer String은 같다고 보장하지 않는다.',
        'Task Quality의 공식 외부 답변 추출은 승인 없이 실행하지 않았다. Raw Task Score를 완전한 품질 Gate로 대체하지 않았다.',
        '25% 후보는 이전 N8 KV 실패로 제외했다. 해당 실패를 삭제하거나 기준을 완화하지 않았다.',
        'Raw Metric, Arrival Contract와 SLO Goodput 분자를 재계산했다. 원래 저장값과 일치했다.',
        'Host Stage 합계도 요청 TTFT와 일치했다. HTTP Enclosure는 GPU-exclusive Compute가 아니다.',
        'Derived 파일은 `analysis/token_confirmation_C37_*_v0`에 있다.',
        'Figure와 대응 CSV는 `figures/token_confirmation_C37_*_v0`에 있다.',
        '본 보고서는 최종 Motivation, Prefill-heavy 및 Encode-safe Headroom 판정을 대신하지 않는다.','']
    report=ROOT.parent/'mllm_epd_protected_token_confirmation_v1.md';assert not report.exists()
    report.write_text('\n'.join(lines));preserve();print('TOKEN_CONFIRM_N3_RECALCULATED',flush=True)


if __name__=='__main__':main()
