"""Reproduce discovery points and host paths without inventing GPU costs."""
import csv
import json
import time

import numpy as np

from .common import ROOT, digest, preserve, write
from .metrics import gates
from .native_common_pilot_report import arrival_contract_SHA256
from .quality_audit import score_request, scorers

TAG='token_DISCOVERY_C34'
STAGES=['arrival_to_coordinator','coordinator_before_E','E_HTTP_enclosure','E_ready_to_P_submit',
    'P_HTTP_enclosure','P_ready_before_D_gate','D_coordinator_gate_wait','after_gate_to_D_submit',
    'D_submit_to_client_first']


def host_stages(request,proxy):
    first=request['token_times'][0]
    gate=proxy.get('D_admission')
    if gate is None:
        times=[request['scheduled_arrival'],proxy['received'],proxy['E_submit'],proxy['E_complete'],
            proxy['P_submit'],proxy['P_ready'],proxy['D_submit'],proxy['D_submit'],proxy['D_submit'],first]
    else:
        times=[request['scheduled_arrival'],proxy['received'],proxy['E_submit'],proxy['E_complete'],
            proxy['P_submit'],proxy['P_ready'],gate['begin'],gate['admitted'],proxy['D_submit'],first]
    values={key:1000*(b-a) for key,a,b in zip(STAGES,times,times[1:])}
    assert all(v>=-1e-5 for v in values.values())
    assert abs(sum(values.values())-request['TTFT_ms'])<1e-5
    return values


def table(path,rows):
    assert rows
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    preserve();source=ROOT/'analysis'/f'{TAG}_summary.json';summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE' and len(summary['results'])==14
    manifest_path=ROOT/'data/main100_manifest_template_v2.json';manifest=json.loads(manifest_path.read_text())
    ids=set(manifest['discovery_ids']);lookup={r['request_id']:r for r in manifest['requests'] if r['request_id'] in ids}
    assert len(lookup)==50 and ids.isdisjoint(manifest['heldout_ids'])
    scoring=scorers();metrics_rows=[];stages=[];failures=[];decisions=[];quality=[];oracle=[];arrivals=[]
    categories=['JOINT_PASS','TTFT_FAIL','ITL_FAIL','TPOT_FAIL','TTFT+ITL_FAIL',
        'TTFT+TPOT_FAIL','ITL+TPOT_FAIL','TTFT+ITL+TPOT_FAIL']
    for cell in summary['results']:
        metric=cell['metrics'];run=metric['run'];fraction=cell['fraction'];method=cell['method'];mode=cell['mode']
        path=ROOT/'raw'/f'{run}_requests.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows)==50 and {r['request_id'] for r in rows}==ids and digest(path)==metric['raw_SHA256']
        assert all(r['status']=='MEASURED_ONLINE' for r in rows)
        arrivals.append(arrival_contract_SHA256(ROOT/'configs'/f'{run}_arrival.json'))
        if method=='Native':proxy_name=TAG+'_Native_proxy'
        elif method=='Static Token':proxy_name=TAG+f'_Token_F{round(100*fraction):02d}_proxy'
        else:proxy_name=TAG+'_Token_idle_proxy'
        proxy_path=ROOT/'raw'/f'proxy_{proxy_name}.jsonl'
        raw_proxy=[json.loads(line) for line in proxy_path.read_text().splitlines()]
        by_id={r['runtime_request_id']:r for r in raw_proxy}
        cell_stage=[]
        for row in rows:
            p=by_id[row['runtime_request_id']];part=host_stages(row,p)
            cell_stage.append(part)
            stages.append(dict(method=method,fraction=fraction,mode=mode,request_id=row['request_id'],
                TTFT_ms=row['TTFT_ms'],output_tokens=row['output_tokens'],**part,
                evidence='MEASURED_ONLINE host boundaries, not GPU-exclusive costs'))
            if mode=='natural':
                q=score_request(lookup[row['request_id']],row['text'],scoring)
                quality.append(dict(method=method,fraction=fraction,request_id=row['request_id'],source=row['source'],
                    score=q['score'],complete_task_scorer=q['complete_task_scorer'],metric=q['metric'],
                    evidence='RAW_TASK_SCORE_ANALYSIS, full quality pending'))
            if method=='Idle-fallback Token':
                decisions.append(dict(mode=mode,request_id=row['request_id'],source=row['source'],
                    selected=p['token_enabled'],P_HTTP_outstanding=p['token_decision']['P_HTTP_outstanding'],
                    D_HTTP_outstanding=p['token_decision']['D_HTTP_outstanding'],
                    coordinator_D_gate_wait_ms=part['D_coordinator_gate_wait'],TTFT_ms=row['TTFT_ms']))
        for category in categories:
            selected=[r for r in rows if gates(r)['category']==category]
            failures.append(dict(method=method,fraction=fraction,mode=mode,category=category,
                requests=len(selected),request_denominator=50,output_tokens=sum(r['output_tokens'] for r in selected),
                total_output_tokens=metric['total_output_tokens']))
        metrics_rows.append(dict(method=method,fraction=fraction,mode=mode,independent_starts=1,
            output_tokens=metric['total_output_tokens'],**{key:metric[key] for key in ['TTFT_mean_ms','TTFT_p50_ms',
                'TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','ITL_max_ms','TPOT_p95_ms','raw_output_tokens_per_second','SLO_goodput']},
            **{key:100*value for key,value in metric['gate_attainment'].items()},
            quality='PENDING',CI='NOT_AVAILABLE',raw_SHA256=digest(path)))
        no_wait=[r['TTFT_ms']-part['D_coordinator_gate_wait'] for r,part in zip(rows,cell_stage)]
        oracle.append(dict(method=method,fraction=fraction,mode=mode,
            observed_TTFT_p95_ms=metric['TTFT_p95_ms'],zero_coordinator_D_wait_TTFT_p95_ms=float(np.percentile(no_wait,95)),
            evidence='OFFLINE_ORACLE',assumptions='Remove observed coordinator wait only. No queue feedback, safety or service recomputation.'))
    assert len(set(arrivals))==1
    mean_stages=[]
    for cell in metrics_rows:
        values=[r for r in stages if r['method']==cell['method'] and r['mode']==cell['mode'] and r['fraction']==cell['fraction']]
        mean_stages.append(dict(method=cell['method'],fraction=cell['fraction'],mode=cell['mode'],
            **{key:float(np.mean([r[key] for r in values])) for key in STAGES},
            mean_TTFT_ms=cell['TTFT_mean_ms'],scope='Disjoint host boundary means; not a percentile or GPU-exclusive decomposition'))
    subset_counts=[]
    for label,selected in [('discovery',set(manifest['discovery_ids'])),('heldout',set(manifest['heldout_ids']))]:
        for name in sorted({r['source'] for r in manifest['requests']}):
            subset_counts.append(dict(subset=label,source=name,requests=sum(r['source']==name and r['request_id'] in selected for r in manifest['requests'])))
    for name,data in [('metrics',metrics_rows),('host_request_paths',stages),('host_stage_means',mean_stages),
        ('failure_categories',failures),('idle_decisions',decisions),('raw_task_scores',quality),
        ('zero_D_gate_oracle',oracle),('split_source_counts',subset_counts)]:
        table(ROOT/'analysis'/f'token_discovery_C34_{name}_v0.csv',data)
    report_data=ROOT/'analysis/token_discovery_C34_report_audit_v0.json'
    write(report_data,dict(time=time.time(),source_SHA256=digest(source),manifest_SHA256=digest(manifest_path),
        arrival_contract_SHA256=arrivals[0],independent_initializations_per_family=1,
        selection=summary['selection'],metrics=metrics_rows,split_source_counts=subset_counts,
        decisions={mode:dict(selected=sum(r['selected'] for r in decisions if r['mode']==mode),requests=50)
            for mode in ['natural','replay']},quality='PENDING',heldout_performance='NOT_RUN',
        scope='All five measured static fractions retained. Idle-fallback is a current-state rule, not a future-window or dynamic-size predictor.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for mode in ['natural','replay']:
        native=next(r for r in metrics_rows if r['mode']==mode and r['method']=='Native')
        static=sorted([r for r in metrics_rows if r['mode']==mode and r['method']=='Static Token'],key=lambda r:r['fraction'])
        idle=next(r for r in metrics_rows if r['mode']==mode and r['method']=='Idle-fallback Token')
        fig,axes=plt.subplots(1,3,figsize=(12,3.6))
        for axis,key,label in zip(axes,['TTFT_p95_ms','ITL_p99_ms','SLO_goodput'],
            ['TTFT p95 (ms)','Client ITL p99 (ms)','SLO-qualified output tokens/s']):
            axis.plot([100*r['fraction'] for r in static],[r[key] for r in static],marker='o',label='Protected static token')
            axis.axhline(native[key],linestyle='--',color='gray',label='Common Native')
            axis.scatter([100*idle['fraction']],[idle[key]],marker='*',s=110,label='Idle fallback')
            axis.set_xlabel('Requested suffix fraction (%)');axis.set_ylabel(label);axis.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'Discovery50 | {mode} | QPS2 | N1 per family | Quality pending; no CI')
        fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'token_discovery_C34_{mode}_v0.{ext}',dpi=180)
        plt.close(fig)
        table(ROOT/'figures'/f'token_discovery_C34_{mode}_v0.csv',[r for r in metrics_rows if r['mode']==mode])
    lines=['# Protected Token Split — Discovery50 Online 결과','',
        '5개 Static 비율과 Idle-fallback을 실제 E/P/D Online 경로에서 측정했다.',
        '증거 수준은 `MEASURED_ONLINE` Runtime Screening이다. 각 Family의 독립 초기화는 1회다.',
        '전체 품질과 독립 N3 CI는 미확정이다. 이 결과로 GO를 선언하지 않는다.','',
        '## 고정 계약','',
        'Main100의 기존 Discovery50만 사용했다. Heldout50 성능은 사용하지 않았다.',
        'Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.',
        'GPU4/5/6, Original BF16, BI0, Full28, 원본 E와 Processor를 유지했다.',
        'P/D는 공통 Compiled 설정, B8192/S1024, KV42GiB, Block128, Prefix Cache Off다.',
        'Native에는 Token Adapter나 새 D Cap을 적용하지 않았다. Stock NativeStrong도 변경하지 않았다.',
        '모든 Arm의 요청 순서, QPS2.0과 상대 Arrival은 같았다.',
        '각 Static 비율은 같은 Chunk 경계의 N1 Teacher와 KV/Output Strict 검사를 통과한 뒤 측정했다.',
        '측정 중 CPU Tensor Capture와 GPU Profiler는 껐다. Warm-up은 표에서 제외했다.',
        '보호 정책은 기존 D HTTP Stream이 없는 경우에만 추가 Suffix를 시작한다.',
        'Native 요청의 대기는 첫 생성 Token에서 해제한다. 전체 Decode 동안 모든 Native를 막지는 않는다.',
        '이 정책은 미래 Safe Window 또는 Dynamic Split-size Predictor가 아니다.','',
        '## 모든 측정 지점','',
        '| Output | Method | D Fraction | Output Tokens | TTFT p50 ms | TTFT p95 ms | ITL p99 ms | Goodput tok/s | Joint % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics_rows:
        lines.append(f'| {r["mode"]} | {r["method"]} | {r["fraction"]:.2f} | {r["output_tokens"]} | '+
            ' | '.join(f'{r[k]:.3f}' for k in ['TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','SLO_goodput','JOINT_PASS'])+' |')
    lines+=['','## 사전 정의한 선택과 해석','',
        'Static 비율 10/20/25/30/50%를 모두 보존했다. 모든 비율이 Runtime Point Gate를 모두 통과하지는 못했다.',
        'Point Gate에는 TTFT p50, ITL p99와 TPOT p95의 Native 대비 1.05 제한이 있었다.',
        '모든 Static 비율의 Replay TTFT p50이 이 제한을 넘었다.',
        '실패 시 사전 규칙에 따라 Replay p95가 가장 작은 10%를 진단용으로 동결했다.',
        '25%는 별도의 최고 Replay Goodput 비율이었다. 두 비율 모두 확정된 Serving Winner가 아니다.',
        'Idle-fallback은 이 10%와 현재 P/D HTTP 상태만 사용했다.',
        'P에 Outstanding 요청이 있고 D HTTP Stream이 없으면 Split을 계획했다. 그 외에는 Native를 유지했다.',
        '이는 정확한 P GPU Queue 또는 미래 EOS를 입력으로 사용한 것이 아니다.',
        '계획 이후에도 실제 D 시작 전에는 기존 D Stream의 완료를 기다렸다.',
        'Idle-fallback Replay p95는 2917.444ms였다. Native는 3204.322ms였다.',
        '차이는 약 8.95%의 단일 실행 Point Gain이다. 독립 반복과 Heldout에서 재현해야 한다.',
        'Replay Goodput은 Native50.192, Idle49.933 tok/s였다.',
        'Natural의 Goodput 차이에는 생성 길이 변화도 포함된다. 이를 계산 Speedup으로 단정하지 않는다.','',
        '## Host 경로와 한계','',
        '요청별 TTFT를 중복 없는 Host 경계로 나눴다. 합계가 원시 TTFT와 일치했다.',
        'E/P HTTP Enclosure에는 Queue, Host 처리와 GPU 작업이 함께 들어 있다.',
        'D Submit 이후 구간에도 KV 전송, Backend Admission, Forward와 Client Delivery가 함께 들어 있다.',
        '이 구간을 GPU-exclusive Kernel Time으로 표시하지 않았다.',
        'Coordinator D Gate 대기를 제거한 값은 별도 `OFFLINE_ORACLE` CSV다.',
        '이 Oracle은 Queue Feedback과 Decode 경쟁을 다시 계산하지 않는다. 실제 개선율이 아니다.',
        '실패 8개 조합의 Request Count와 Output Token Weight도 저장했다.',
        'ITL p99가 낮아도 모든-ITL 조건에서 탈락한 요청은 삭제하지 않았다.','',
        '## Split 구성과 품질 한계','',
        '기존 Discovery/Heldout은 Source 비율이 다르다. 분할을 사후 변경하지 않았다.','',
        '| Split | Source | Requests |','|---|---|---:|']
    for r in subset_counts:lines.append(f'| {r["subset"]} | {r["source"]} | {r["requests"]} |')
    lines+=['',
        '이 분포 차이는 Generalization 한계다. Main100 전체가 Production Traffic을 대표한다고 주장하지 않는다.',
        'Raw Task Score를 같은 원래 Scorer로 계산했다. MMLongBench-Doc 공식 답변 추출은 미실행이다.',
        'DocVQA Raw ANLS Floor도 남아 있다. 전체 Task Quality는 계속 `PENDING`이다.',
        'Heldout 전에 비율과 현재 상태 Rule을 동결했다. 새 SLO를 결과에 맞춰 설정하지 않았다.',
        'CSV는 `analysis/token_discovery_C34_*_v0.csv`다. 실제 Figure는 `figures/token_discovery_C34_*_v0`에 있다.','']
    report=ROOT.parent/'mllm_epd_protected_token_discovery_v1.md';assert not report.exists()
    report.write_text('\n'.join(lines));preserve();print('TOKEN_DISCOVERY_RECALCULATED',flush=True)


if __name__=='__main__':main()
