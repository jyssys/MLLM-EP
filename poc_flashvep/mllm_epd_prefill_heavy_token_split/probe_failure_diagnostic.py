"""Explain request-level probe failures without inferring a GPU-tail cause."""
import csv
import json
import time

from .common import ROOT,digest,preserve,write
from .token_confirmation_analysis import mean_interval
from .token_discovery_analysis import table

TAG='interference_PROBE_C40R4'
ARMS=['NativeStrong','NativeCommon','clean_before','E128','E512','D128','D512','clean_after']


def failure_counts(rows):
    assert len(rows)==8 and len({r['category'] for r in rows})==8
    denominators={int(r['request_denominator']) for r in rows};assert len(denominators)==1
    total=denominators.pop();assert sum(int(r['requests']) for r in rows)==total
    output={int(r['total_output_tokens']) for r in rows};assert len(output)==1
    tokens=output.pop();assert sum(int(r['output_tokens']) for r in rows)==tokens
    def count(gates):
        selected=[r for r in rows if all(gate not in r['category'] for gate in gates)]
        return sum(int(r['requests']) for r in selected),sum(int(r['output_tokens']) for r in selected)
    values={}
    for label,gates in [('TTFT',['TTFT']),('ITL',['ITL']),('TPOT',['TPOT']),
            ('DECODE',['ITL','TPOT']),('JOINT',['TTFT','ITL','TPOT'])]:
        requests,qualified=count(gates)
        values[label+'_pass_requests']=requests;values[label+'_pass_percent']=100*requests/total
        values[label+'_qualified_tokens']=qualified
    return dict(requests=total,output_tokens=tokens,**values)


def main():
    preserve();source=ROOT/'analysis'/f'{TAG}_audit_v0.json';audit=json.loads(source.read_text())
    assert len(audit['decisions'])==8 and (ROOT/'state'/f'{TAG}_controller_complete.json').exists()
    start_path=ROOT/'analysis'/f'{TAG}_starts_v0.csv'
    failure_path=ROOT/'analysis'/f'{TAG}_failure_categories_v0.csv'
    with start_path.open(newline='') as stream:starts=list(csv.DictReader(stream))
    with failure_path.open(newline='') as stream:failures=list(csv.DictReader(stream))
    records=[]
    for row in starts:
        if row['arm'] not in ARMS:continue
        selected=[r for r in failures if all(r[key]==row[key] for key in ['block','mode','arm'])]
        counts=failure_counts(selected)
        assert abs(counts['DECODE_pass_percent']-float(row['decode_pass_percent']))<1e-9
        records.append(dict(block=int(row['block']),mode=row['mode'],arm=row['arm'],**counts,
            client_ITL_p99_ms=float(row['ITL_p99_ms']),client_max_ITL_ms=float(row['ITL_max_ms']),
            p99_below_34ms=float(row['ITL_p99_ms'])<=34,
            actual_request_decode_95_gate=counts['DECODE_pass_percent']>=95,
            evidence='TRACE_ANALYSIS of actual online probe controls; not a GPU kernel-cause attribution.'))
    assert len(records)==48
    table(ROOT/'analysis/probe_failure_tail_C40_v0.csv',records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,label,name,threshold in [('DECODE_pass_percent','Requests passing every ITL AND TPOT (%)',
            'probe_C40_actual_decode_attainment_v0',95),
            ('client_max_ITL_ms','Largest client ITL in each start (ms)',
            'probe_C40_client_max_ITL_v0',34)]:
        fig,axes=plt.subplots(1,2,figsize=(13,4.8));points=[]
        for axis,mode in zip(axes,['natural','replay']):
            for index,arm in enumerate(ARMS):
                rows=[r for r in records if r['mode']==mode and r['arm']==arm]
                assert len(rows)==3;values=[r[key] for r in rows];stat=mean_interval(values);lo,hi=stat['CI']
                axis.errorbar(index,stat['mean'],yerr=[[stat['mean']-lo],[hi-stat['mean']]],fmt='o',capsize=4)
                axis.plot([index]*3,values,'x',markersize=4,color='gray',alpha=.7)
                points.append(dict(mode=mode,arm=arm,metric=key,mean=stat['mean'],CI_low=lo,CI_high=hi,
                    actual_start_min=min(values),actual_start_max=max(values),independent_initializations=3,
                    evidence='INTERFERENCE_PROBE; not a client task speedup or GPU-exclusive latency.'))
            axis.axhline(threshold,color='red',linestyle='--',label=f'Frozen threshold {threshold}')
            axis.set_xticks(range(len(ARMS)),ARMS,rotation=35,ha='right');axis.set_ylabel(label)
            axis.set_title(mode);axis.grid(alpha=.2);axis.legend(fontsize=8)
            if key=='DECODE_pass_percent':axis.set_ylim(max(0,min(r[key] for r in records)-2),101)
        fig.suptitle('Independent N3 | Grey x: actual starts | Start mean CI | No GPU/host/delivery cause inferred')
        fig.tight_layout()
        for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
        plt.close(fig);table(ROOT/'figures'/f'{name}.csv',points)
    misleading=[r for r in records if r['arm'] in ['E128','E512','D128','D512'] and
        r['p99_below_34ms'] and not r['actual_request_decode_95_gate']]
    write(ROOT/'analysis/probe_failure_tail_C40_audit_v0.json',dict(time=time.time(),source_SHA256=digest(source),
        starts_SHA256=digest(start_path),failure_categories_SHA256=digest(failure_path),
        actual_probe_cohorts=24,p99_below_threshold_but_request_decode_gate_failed=len(misleading),
        examples=misleading,per_request_all_ITL_gate_preserved=True,original_N3_verdicts_unchanged=True,
        GPU_interference_cause='NOT_MEASURED',client_vs_worker_per_token_delivery_cause='NOT_MEASURED',
        scope='Metric failure attribution only. Fixed period order and batch/natural-output variation prevent a unique hardware-cause claim.'))
    report=ROOT.parent/'mllm_epd_probe_decode_tail_and_failure_attribution_v1.md';assert not report.exists()
    lines=['# E/D Probe: ITL p99와 요청별 Decode 실패의 차이','',
        '증거는 실제 C40 N3의 TRACE_ANALYSIS다. 새 GPU 실행이나 Profiler는 추가하지 않았다.',
        '원래 8개 Natural/Replay 보호 Gate 판정은 변경하지 않았다. 모든 조건은 전체 Gate 미통과다.',
        '측정한 Probe는 Main 요청을 완료하지 않는다. 별도의 Throughput 개선은 아니다.','',
        '## 다른 분모와 통계량','',
        'ITL p99는 해당 Start의 모든 Client Token Interval을 모은 99th Percentile이다.',
        'All-ITL Pass는 한 요청의 모든 Interval이 34ms 이하여야 한다.',
        'Absolute Decode는 All-ITL과 TPOT≤33ms/token의 요청별 AND다. 각 Start에서 95% 이상이어야 한다.',
        '따라서 일부 요청의 드문 큰 Gap은 전체 ITL p99에 나타나지 않아도 해당 요청을 실패시킨다.',
        f'실제 Probe Cohort24개 중 {len(misleading)}개는 ITL p99≤34ms였지만 요청별 Decode 95% Gate를 실패했다.','',
        '| Start | Mode | Probe | ITL p99 ms | Client Max ITL ms | All-ITL AND TPOT Pass % |',
        '|---:|---|---|---:|---:|---:|']
    for row in misleading:
        lines.append(f'| {row["block"]} | {row["mode"]} | {row["arm"]} | {row["client_ITL_p99_ms"]:.3f} | '+
            f'{row["client_max_ITL_ms"]:.3f} | {row["DECODE_pass_percent"]:.1f} |')
    lines+=['','전체 보호 Gate 실패에는 Absolute 실패뿐 아니라 Relative CI 상한 실패도 포함된다.',
        'Point Ratio가 1.05 이하여도 CI 상한이 1.05를 넘으면 비열등성이 입증되지 않는다.',
        '그 결과를 모든 Metric의 유의한 악화라고 바꾸지 않는다. N3 CI 해상도도 제한적이다.',
        'E512 Natural의 일부 Point·CI 개선을 선택해 다른 실패 Gate를 삭제하지 않았다.','',
        '## 확정하지 못한 원인','',
        '실제 GPU Kernel 경쟁, Remote READ, Host Scheduling, Client Delivery를 Token별로 분해하지 못했다.',
        'Clean C40에서는 해당 상세 계측을 활성화하지 않았다. 다른 Diagnostic 실행의 Operator 비율로 대체하지 않는다.',
        'E128·E512·D128·D512 순서는 고정했다. 기간 효과와 다른 Batch·Natural Output 차이가 남아 있다.',
        '따라서 이 표는 Metric 실패 원인을 설명한다. 드문 Gap의 단일 Runtime 또는 Hardware 원인을 확정하지 않는다.',
        'Counterfactual Gate 제거, GPU Cut 변경 또는 미래 안전 Window의 효과로 해석하지 않는다.','',
        '## 원시 자료','',
        'analysis/probe_failure_tail_C40_v0.csv는 48개 실제 Control/Probe Cohort를 모두 포함한다.',
        '각 요청·Token의 8개 실패 조합은 원래 interference_PROBE_C40R4_failure_categories_v0.csv에 있다.',
        '대응 Figure와 CSV는 figures/probe_C40_actual_decode_attainment_v0 및 probe_C40_client_max_ITL_v0다.','']
    report.write_text('\n'.join(lines));preserve();print('PROBE_REQUEST_DECODE_FAILURE_RECALCULATED',len(misleading),flush=True)


if __name__=='__main__':main()
