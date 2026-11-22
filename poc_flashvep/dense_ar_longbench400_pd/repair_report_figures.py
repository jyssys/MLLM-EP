"""Present independent adaptive repair without replacing the failed main arm."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,REPO
from .confirmed_figures import mean_ci
from .figures import save
from .final_report_tables import interval


def main():
    assert (ROOT/'state/hybrid_READ_repair_layer_control_complete_v1.json').exists()
    data=json.loads((ROOT/'analysis/hybrid_READ_repair_results_v1.json').read_text())
    assert data['complete'],'Keep quality-invalid or failed repair outcomes separate. Do not draw them as normal confirmations.'
    group=next(r for r in data['groups'] if r['stage']=='INDEPENDENT_REPAIR_CONFIRMATION')
    arms=('NativeStrong','NativeCompute','NativeREAD','NativeCompletion','LayerControl',
        'StockToken','HybridCompute','HybridREAD')
    values={a:[json.loads((ROOT/'analysis'/(s['arms'][a]['name']+'_metrics.json')).read_text())
        for s in group['starts']]for a in arms}
    path=REPO/'poc_flashvep/reports/dense_ar_longbench400_decode_tail_recovery.md'
    marker='## 10. Adaptive Repair의 독립 Confirmation'
    prefix=path.read_text().split(marker)[0].rstrip()
    lines=['',marker,'',
        '이 검사는 원래 140-start Main의 결과를 교체하지 않는다.',
        'Main400의 단일 Tail 증상 이후 설계한 별도 Adaptive Follow-up이다.',
        '모든 새 Arrival Seed와 대조군은 새 성능 측정 전에 동결했다.',
        'QPS1.8, Natural EOS, Cap2048, BI0와 B8192/S1024를 사용했다.',
        '새 Seed는 2026112661/2662/2663이다. 각 Arm은 독립 Start 3회다.',
        '각 Start의 원본 Request는 400개다. 모든 새 Start는 GPU6/7을 사용했다.',
        'Layer Control은 원래 Discovery에서 동결한 후보다.',
        'Layer Control 3회는 7-arm Block 뒤에 실행했다. 같은 Randomized Order로 처리하지 않는다.',
        '실제 Continuous Batch는 강제하지 않았다. Runtime과 Actual Work를 별도로 기록한다.','',
        '| Arm | TTFT p95(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | SLO-goodput(output tokens/s) | 실제 Output Tokens/Start |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for arm in arms:
        row=group['arms'][arm];m=row['means']
        m=dict(m,observed_global_max_ITL_ms=max(v['max_ITL_ms'] for v in values[arm]))
        fields=('TTFT_p95_ms','ITL_p99_ms','max_ITL_ms','observed_global_max_ITL_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')
        lines.append('| '+arm+' | '+' | '.join(f'{m[f]:.6f}' for f in fields)+' | '+str(row['output_tokens'])+' |')
    lines += ['','표는 Start별 Metric의 평균이다. Max 열도 3개 Start Max의 평균이다.',
        '관측 전체 Max 열은 별도로 모든 유효 Start의 단일 최댓값이다. 모집단의 최악 지연 상한은 아니다.',
        'Start-mean Max를 관측 전체 Max 또는 Pooled p99로 해석하지 않는다.',
        '정확한 각 Start의 Max와 원시 Token Event는 Raw에 보존했다.',
        'Natural Output 수가 다르면 동일한 생성 작업량을 가정하지 않는다.','',
        '| Arm / Reference | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | TTFT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |',
        '|---|---|---|---|---|---|---|']
    for name,test in group['comparisons'].items():
        fields=('ITL_p99_ms','TPOT_p95_ms','TTFT_p95_ms','SLO_qualified_output_tokens_per_second')
        lines.append('| '+name.replace('_over_',' / ')+' | '+' | '.join(interval(test['ratios'][f])for f in fields)+
            ' | '+str(test['decode_5percent_CI_pass'])+' | '+str(test['absolute_decode_pass'])+' |')
    lines += ['','비율 CI는 같은 Seed Block을 재표본한 Pointwise Empirical 95% CI다.',
        'N3에서 가능한 Bootstrap 27개 재표본을 완전 열거했다. 다중 비교 보정은 하지 않았다.',
        'HybridREAD와 LayerControl은 NativeStrong과 NativeREAD를 모두 검사한다.',
        'Stock Token은 NativeStrong과 같은 Completion Guard를 가진 NativeCompletion을 모두 검사한다.',
        '빠른 TTFT만으로 안전한 Method로 처리하지 않는다. 두 Decode Metric과 Absolute Gate를 함께 사용한다.','',
        '### Runtime과 Actual Geometry','',
        '| Arm | Source Credit | Compile Family | P/D 최대 Request Batch | P/D 최대 Padded Query |',
        '|---|---|---|---|---|']
    for arm in arms:
        result=group['arms'][arm]
        maximum=lambda role,field:max(v['actual_geometry'][role][field]for v in values[arm])
        batch='/'.join(str(maximum(r,'maximum_requests'))for r in ('prefill','decode'))
        padded='/'.join(str(maximum(r,'maximum_padded_queries'))for r in ('prefill','decode'))
        lines.append('| '+arm+' | '+str(sorted({c['credit']for c in result['contracts']}))+' | '+
            ','.join(sorted({c['family']for c in result['contracts']}))+' | '+batch+' | '+padded+' |')
    lines += ['','이 비교는 Best-versus-best Continuous Family 비교다. Actual Batch1/1/1 또는 Exact Iso-work Control이 아니다.',
        'READ 지연은 관련 Transfer와 Stream 시점도 바꾼다. 순수 Host 또는 DMA 비용의 단독 원인으로 해석하지 않는다.',
        '같은 Source의 READ-only A/B는 Runtime 경로의 인과적 근거를 제공한다.',
        '모든 GPU Operator의 비용 비중이나 미래 Decode-safe Window를 식별한 것은 아니다.',
        '이 Repair의 성능을 원래 맹검 Held-out Main의 성능으로 표시하지 않는다.',
        '동일 Request의 새 Arrival Seed 검증이다. 완전히 새로운 Request Population의 검증은 아니다.','',
        '전체 Source, Request, 실제 Geometry와 Host Tail 근거는 [Repair 분석](dense_ar_longbench400_2d_20261121/analysis/hybrid_READ_repair_results_v1.json)에 있다.','']
    path.write_text(prefix+'\n'+'\n'.join(lines),encoding='utf-8')
    colors=['#777777','#aaaaaa','#009e73','#56b4e9','#0072b2','#cc79a7','#d55e00','#e69f00']
    for field,ylabel,stem in (('TTFT_p95_ms','TTFT p95 (ms)','ttft'),('ITL_p99_ms','Client-visible ITL p99 (ms)','itl'),
            ('TPOT_p95_ms','Request TPOT p95 (ms/token)','tpot'),
            ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','goodput')):
        fig,ax=plt.subplots(figsize=(11,5.4));points=[]
        for i,(arm,color)in enumerate(zip(arms,colors)):
            mean,ci=mean_ci([v[field]for v in values[arm]])
            ax.bar(i,mean,color=color,alpha=.8)
            ax.errorbar(i,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='none',ecolor='black',capsize=3)
            points.append(dict(arm=arm,mean=mean,CI95=ci,starts=3))
        ax.set(xticks=range(len(arms)),xticklabels=arms,ylabel=ylabel,
            title='Adaptive decode repair | Main400 | QPS1.8 | Three new paired starts on6/7')
        ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
        save(fig,'adaptive_repair_confirmed_'+stem,
            'MEASURED ONLINE ADAPTIVE REPAIR | N400/start | Natural EOS,cap2048 | BI0,B8192/S1024,KV42GiB.\n'
            'Error bars:pointwise start-mean empirical95% CI,27 resamples. Original unsafe main arms remain unchanged.\n'
            'LayerControl follows the seven-arm time blocks. Actual batches,compile family,credit and output counts are recorded,not forced equal.\n'
            'This is not the original blind test,new-request generalization,sustainable capacity,or dynamic-window selection.',
            dict(evidence='MEASURED_ONLINE_ADAPTIVE_REPAIR',metric=field,points=points,source=group))
    print('ADAPTIVE_REPAIR_REPORT_AND_FIGURES',4,'PNG/PDF pairs',flush=True)


if __name__=='__main__':
    main()
