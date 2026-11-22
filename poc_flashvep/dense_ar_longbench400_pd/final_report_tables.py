"""Render final start-level metrics without changing raw measurements."""
import json
from .common import ROOT,digest


def number(value):
    return 'NOT_AVAILABLE' if value is None else f'{value:,.3f}'


def interval(row):
    if row is None:return 'NOT_AVAILABLE'
    estimate='NOT_AVAILABLE' if row['estimate'] is None else f"{row['estimate']:.6f}"
    return estimate+' ['+','.join(f'{v:.6f}' for v in row['CI95'])+']' if row['CI95'] is not None else estimate+' [N1, CI 없음]'


def main():
    assert all((ROOT/'state'/f'final_performance_lane{i}_complete_v1.json').exists() for i in range(4))
    source=json.loads((ROOT/'analysis/final_continuous_strong_confirmation_v1.json').read_text())
    plan_path=ROOT/'configs/final_continuous_execution_plan_v1.json'
    plan=json.loads(plan_path.read_text())
    lines=['# Final C3: 실측 Table','',
        '이 문서는 Raw Client Timestamp에서 다시 계산한 값만 표시한다.',
        '같은 Cell의 Arm은 Dataset, Arrival Trace와 GPU Pair가 같다.',
        '기본 Metric은 Start별 통계량의 평균이다. 관측 전체 Max 열만 모든 유효 Start의 단일 최댓값이다.',
        'Request를 독립 Start로 재표본하지 않는다.',
        'N3의 비율 CI는 Paired Start Bootstrap 27개 재표본을 완전 열거했다.',
        '이 CI는 Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않았다.',
        'N1에는 Start-level CI가 없다. 유한 Cohort 처리량은 Sustainable Capacity가 아니다.','',
        '표의 Metric은 소수 3자리, 비율은 소수 6자리로 표시한다. Gate는 반올림 전 값으로 판정한다.','',
        'Source Capsule SHA256: `'+source['source_capsule_SHA256']+'`.',
        'Execution Plan SHA256: `'+digest(plan_path)+'`.',
        'Main Dataset SHA256: `'+digest(ROOT/'data/longbench_balanced400_v1.json')+'`.',
        '모든 Worker는 BI0, Original BF16, B8192/S1024, KV42GiB, Block128을 사용한다.',
        'Natural EOS를 허용한다. Output Cap은 2048이다.',
        'Task-quality Gate는 다른 BI0 Batch의 Greedy Token 완전 일치를 뜻하지 않는다.','']
    for cell in source['cells']:
        label=f"{cell['workload']} / QPS{cell['QPS']}"
        lines += ['## '+label,'',f"Cell 상태: `{cell['status']}`. 예정 독립 Start: {cell['required_starts']}.",'',
            '| Arm | 유효 Start | TTFT p95(ms) | Mean TBT(ms) | ITL p99(ms) | Start-mean Max ITL(ms) | 관측 전체 Max ITL(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO output tokens/s | Joint SLO | Completed req/s |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm,result in cell['arm_results'].items():
            m=dict(result['means'])
            m['observed_global_max_ITL_ms']=max((json.loads((ROOT/'analysis'/(
                block['arms'][arm]['name']+'_metrics.json')).read_text())['max_ITL_ms']
                for block in cell['starts'] if block['arms'][arm]['status']=='MEASURED_ONLINE'),default=None)
            fields=('TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','max_ITL_ms','observed_global_max_ITL_ms','TPOT_p95_ms',
                'generated_output_tokens_per_second','SLO_qualified_output_tokens_per_second',
                'Joint_SLO_attainment','completed_requests_per_second')
            lines.append('| '+arm+' | '+str(result['valid_starts'])+' | '+' | '.join(number(m.get(k)) for k in fields)+' |')
        lines += ['','### Actual Work와 Runtime','',
            '| Arm | 실제 Output Tokens/Start | P/D 최대 Requests | P/D 최대 Query | P/D 최대 Padded Query | Source Credit | D 정책 | Compile Family |',
            '|---|---|---|---|---|---|---|---|']
        for arm,result in cell['arm_results'].items():
            def maximum(role,field):
                return max((g.get(role,{}).get(field,0) for g in result['actual_geometry']),default=0)
            roles=('replica0','replica1') if arm=='Agg' else ('prefill','decode')
            runtime=result['runtime_contract']
            geometry=['/'.join(str(maximum(r,f)) for r in roles) for f in
                ('maximum_requests','maximum_queries','maximum_padded_queries')]
            credits=sorted({r['Source_credit'] for r in runtime})
            policies=sorted({json.dumps(r['policy'],separators=(',',':')) for r in runtime})
            families=sorted({r['execution_family'] for r in runtime})
            lines.append('| '+' | '.join([arm,str(result['per_start_output_tokens']),*geometry,
                str(credits),'<br>'.join(policies),'<br>'.join(families)])+' |')
        lines += ['','D 정책 순서는 Compute Deferral / READ Deferral / Whole-request Quiescence / Running Cap / Completion Guard다.',
            'Running Cap은 Actual Batch 보장이 아니다. Agg의 두 값은 Replica0/1이다.',
            'Query와 Padded Query는 해당 Role에서 관측한 최대값이다. 모든 Step의 Batch가 이 최대값인 것은 아니다.','',
            '### Paired 비교','',
            '| Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | SLO-goodput 비율 [95% CI] | Decode 5% Gate | Absolute Decode≥95% |',
            '|---|---|---|---|---|---|---|']
        for arm,refs in cell['comparisons'].items():
            for ref,test in refs.items():
                fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')
                gate=test['decode_5percent_CI_pass']
                safety='N1, NOT_CONFIRMED' if gate is None else 'PASS' if gate else 'FAIL'
                lines.append('| '+arm+' / '+ref+' | '+' | '.join(interval(test['ratios'][f]) for f in fields)+
                    ' | '+safety+' | '+str(test['absolute_decode_attainment_95percent'])+' |')
        lines += ['','TTFT 우위, Goodput 우위와 Decode Protection은 별도 Gate다.',
            'NativeStrong과 같은 D 정책의 Native를 모두 검사한다.',
            '안전하지 않은 Arm을 빠르다는 이유로 적격 승리로 처리하지 않는다.','']
    lines += ['## 모든 개별 Start','',
        '| Workload | QPS | Start | Seed | GPU Pair | Arm | Raw Name | 상태 |',
        '|---|---:|---:|---:|---|---|---|---|']
    for cell in source['cells']:
        for index,block in enumerate(cell['starts']):
            for arm,row in block['arms'].items():
                lines.append('| '+' | '.join(map(str,[cell['workload'],cell['QPS'],index,block['seed'],
                    block['GPU_pair'],arm,row['name'],row['status']]))+' |')
    lines += ['','Raw Name의 `raw/<name>/configuration.json`, `requests.jsonl`과 `actual_queries_*.json`을 함께 사용한다.',
        '누락되거나 실패한 Start를 보간하지 않는다. Actual Natural Output 수가 다르면 그 차이도 보존한다.','']
    target=ROOT/'analysis/final_continuous_report_tables_v1.md'
    target.write_text('\n'.join(lines),encoding='utf-8')
    print('FINAL_NUMERIC_TABLES',target,flush=True)


if __name__=='__main__':
    main()
