"""Natural baseline sweep with start-level intervals and explicit SLO failures."""
import csv
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .metrics import summarize,gates

LABELS={'agg':'Agg DP3','e_pd':'E + colocated PD DP2','native':'Native EPD'}
FIELDS=['TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','TPOT_p95_ms',
    'raw_output_tokens_per_second','total_input_output_tokens_per_second','completed_requests_per_second','SLO_goodput',
    'TTFT_PASS','ITL_PASS','TPOT_PASS','JOINT_PASS']


def save_csv(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def bootstrap(values):
    values=np.asarray(values,float)
    if len(values)<3:return None
    rng=np.random.default_rng(2026112411)
    samples=values[rng.integers(len(values),size=(20000,len(values)))].mean(axis=1)
    return np.percentile(samples,[2.5,97.5]).tolist()


def verify_timing(record):
    times=record['token_times']
    assert len(times)==len(record['token_ids'])==record['output_tokens']>0
    assert record['usage']['completion_tokens']==len(times)
    assert all(b>=a for a,b in zip(times,times[1:]))
    ttft=(times[0]-record['scheduled_arrival'])*1000
    itl=[(b-a)*1000 for a,b in zip(times,times[1:])]
    tpot=(times[-1]-times[0])*1000/(len(times)-1) if len(times)>1 else 0
    assert abs(ttft-record['TTFT_ms'])<1e-7 and abs(tpot-record['TPOT_ms'])<1e-7
    assert itl==record['ITL_ms']


def main():
    preserve()
    complete=ROOT/'state/main100_baseline_confirmation_C14_complete.json'
    assert json.loads(complete.read_text())['status']=='COMPLETE'
    result_path=ROOT/'analysis/main100_baseline_confirmation_C14_v0.json'
    assert not result_path.exists(),'Keep prior summaries frozen'
    selected=[]
    for family in ['agg','e_pd','native']:
        for start in [1,2,3]:
            path=ROOT/'analysis'/f'base_C14_{family}_S{start}.json'
            value=json.loads(path.read_text());assert value['status']=='COMPLETE'
            selected.extend((family,start,row['run']) for row in value['rows'] if '_natural100_' in row['run'])
            if family=='native':selected.append((family,start,f'ec_NATIVE_C11_S{start}_natural100_q2_v0'))
    rows=[];failures=[];requests=[];trace_by_qps={}
    categories=['JOINT_PASS','TTFT_FAIL','ITL_FAIL','TPOT_FAIL','TTFT+ITL_FAIL','TTFT+TPOT_FAIL','ITL+TPOT_FAIL','TTFT+ITL+TPOT_FAIL']
    for family,start,run in selected:
        value=summarize(run)
        raw=[json.loads(text) for text in (ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines()]
        for actual in raw:verify_timing(actual)
        done=json.loads((ROOT/'state'/f'{run}_complete.json').read_text())
        assert done['raw_SHA256']==value['raw_SHA256']
        assert abs(max(r['finish'] for r in raw)-done['monotonic_start']-done['duration_seconds'])<1e-8
        assert value['primary_execution_valid'] and value['completed_count']==100
        config=json.loads((ROOT/'configs'/f'{run}.json').read_text());qps=float(config['QPS'])
        arrival=json.loads((ROOT/'configs'/f'{run}_arrival.json').read_text())
        canonical=json.dumps(arrival['trace'],sort_keys=True)
        if qps in trace_by_qps:assert trace_by_qps[qps]==canonical
        else:trace_by_qps[qps]=canonical
        row=dict(method=family,start=start,QPS=qps,run=run,requests=100,output_tokens=value['total_output_tokens'],
            duration_seconds=value['duration_seconds'],raw_SHA256=value['raw_SHA256'],ITL_max_ms=value['ITL_max_ms'],
            **{k:value[k] for k in FIELDS if k in value},
            **{k:100*v for k,v in value['gate_attainment'].items()})
        rows.append(row)
        for category in categories:
            count=value['categories'].get(category,0);tokens=value['category_output_tokens'].get(category,0)
            failures.append(dict(method=family,start=start,QPS=qps,run=run,category=category,
                request_count=count,request_fraction=count/100,output_tokens=tokens,
                generated_token_fraction=tokens/value['total_output_tokens'],
                token_rate=tokens/value['duration_seconds']))
        if qps in [2.,4.]:
            for actual in raw:
                gate=gates(actual)
                requests.append(dict(method=family,start=start,QPS=qps,request_id=actual['request_id'],
                    index=actual['index'],scheduled_arrival=actual['scheduled_arrival'],TTFT_ms=actual['TTFT_ms'],
                    output_tokens=actual['output_tokens'],TPOT_ms=actual['TPOT_ms'],
                    ITL_max_ms=max(actual['ITL_ms'],default=0),**gate))
    summary=[]
    for family in ['agg','e_pd','native']:
        for qps in sorted(trace_by_qps):
            cells=[r for r in rows if r['method']==family and r['QPS']==qps]
            if not cells:continue
            for metric in FIELDS:
                values=[r[metric] for r in cells];ci=bootstrap(values)
                summary.append(dict(method=family,QPS=qps,metric=metric,start_count=len(cells),
                    requests_per_start=100,start_mean=float(np.mean(values)),
                    CI95_low=ci[0] if ci else None,CI95_high=ci[1] if ci else None,
                    evidence='Actual Natural C3 runtime; full task quality PENDING'))
    write(result_path,dict(time=time.time(),status='RUNTIME_MEASURED_QUALITY_PENDING',rows=rows,summary=summary,
        QPS_grid=sorted(trace_by_qps),GPU_group=[4,5,6],natural_EOS=True,
        scope='No token split. Independent N3 at QPS2/4, discovery N1 elsewhere. '
            'Start-mean confidence intervals do not pool requests. Different Natural output work is retained. '
            'Finite-cohort throughput is not sustainable capacity. Full official task quality remains unresolved.'))
    save_csv(ROOT/'analysis/main100_baseline_confirmation_C14_starts_v0.csv',rows)
    save_csv(ROOT/'analysis/main100_baseline_confirmation_C14_summary_v0.csv',summary)
    save_csv(ROOT/'analysis/main100_baseline_confirmation_C14_failure_categories_v0.csv',failures)
    save_csv(ROOT/'analysis/main100_baseline_confirmation_C14_requests_v0.csv',requests)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in FIELDS:
        fig,ax=plt.subplots(figsize=(7,4));figure_rows=[]
        for family in ['agg','e_pd','native']:
            cells=sorted((r for r in summary if r['method']==family and r['metric']==metric),key=lambda r:r['QPS'])
            figure_rows.extend(cells)
            x=[r['QPS'] for r in cells];y=[r['start_mean'] for r in cells]
            line=ax.plot(x,y,'o-',label=LABELS[family])[0]
            repeated=[r for r in cells if r['start_count']>=3]
            if repeated:
                ax.errorbar([r['QPS'] for r in repeated],[r['start_mean'] for r in repeated],
                    yerr=[[r['start_mean']-r['CI95_low'] for r in repeated],
                          [r['CI95_high']-r['start_mean'] for r in repeated]],
                    fmt='none',capsize=3,color=line.get_color())
        ax.set_xlabel('Offered QPS (requests/s)')
        ax.set_ylabel(metric+' (%)' if metric.endswith('PASS') else metric)
        ax.set_title('Qwen2.5-VL-7B | 3 H100 | Natural Main100, quality pending',fontsize=10)
        ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.text(.01,.01,'N3 at QPS2/4, N1 elsewhere. CI: independent starts. Finite cohort and drain.',fontsize=8)
        fig.tight_layout(rect=[0,.04,1,1])
        stem=ROOT/'figures'/f'main100_C14_natural_{metric}_v0'
        for suffix in ['png','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=170)
        save_csv(stem.with_suffix('.csv'),figure_rows);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    cdf_rows=[]
    for family in ['agg','e_pd','native']:
        values=sorted(r['TTFT_ms'] for r in requests if r['method']==family and r['QPS']==2.)
        ax.plot(values,np.arange(1,len(values)+1)/len(values),label=LABELS[family])
        cdf_rows.extend(dict(method=family,QPS=2,TTFT_ms=value,CDF=(i+1)/len(values),requests=len(values))
                        for i,value in enumerate(values))
    ax.axvline(5000,color='black',ls=':',label='Frozen TTFT SLO')
    ax.set_xlabel('TTFT (ms)');ax.set_ylabel('Request CDF')
    ax.set_title('QPS2 Main100 | Pooled descriptive CDF, not start CI')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(ROOT/'figures'/f'main100_C14_TTFT_CDF_q2_v0.{suffix}',dpi=170)
    save_csv(ROOT/'figures/main100_C14_TTFT_CDF_q2_v0.csv',cdf_rows)
    plt.close(fig)
    lines=['# Main100 baseline motivation — Natural C3 confirmation','',
        '세 방법은 같은 H100 3개와 동일한 Main100을 사용했다. QPS2와 QPS4는 독립 서버 초기화 3회다.',
        '나머지 QPS는 Discovery N1이다. 전체 Task Quality Gate는 미확정이다.',
        '이 표에는 Generic Token Split이 없다. DuetServe의 수치 재현도 아니다.','',
        '| Method | QPS | Starts | TTFT p95 ms | ITL p99 ms | TPOT p95 ms | Raw tok/s | SLO tok/s | Joint % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for family in ['agg','e_pd','native']:
        for qps in sorted(trace_by_qps):
            cells=[r for r in rows if r['method']==family and r['QPS']==qps]
            if not cells:continue
            keys=['TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','raw_output_tokens_per_second','SLO_goodput','JOINT_PASS']
            lines.append('| '+LABELS[family]+f' | {qps:g} | {len(cells)} | '+
                ' | '.join(f'{np.mean([r[k] for r in cells]):.3f}' for k in keys)+' |')
    lines+=['','표는 Start별 Metric의 평균이다. 요청 300개를 합친 Percentile이 아니다.',
        'CI와 Start별 실제 Output Tokens는 대응 CSV와 JSON에 보존했다.',
        '모든 방법의 같은 QPS는 상대 Arrival Trace와 요청 순서가 같았다.',
        'Natural 출력 길이 차이는 유지했다. Equal-output Replay는 별도 결과다.',
        'Throughput은 첫 예정 Arrival부터 Drain까지의 유한 집합 값이다. Sustainable Capacity로 해석하지 않는다.',
        'GPU Kernel Busy와 Queue의 기존 동시 계측은 5% 기준 미통과로 진단용이다.',
        '이 표만으로 Prefill-heavy 또는 Decode-safe Headroom을 확정하지 않는다.',
        'Task Quality 미확정을 Timing 표로 대체하지 않는다.','']
    report=ROOT.parent/'mllm_epd_main100_baseline_motivation_v1.md'
    assert not report.exists();report.write_text('\n'.join(lines));preserve()
    print('C14_NATURAL_BASELINE_ANALYSIS_COMPLETE',len(rows),len(summary),flush=True)


if __name__=='__main__':main()
