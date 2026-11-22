"""Report all iso-work cells and finite-grid static controls without promotion."""
import json
from .common import ROOT,REPO


def ci(row):
    if row is None:return 'NOT_AVAILABLE'
    return f"{row['estimate']:.6f} [{row['CI95'][0]:.6f},{row['CI95'][1]:.6f}]"


def main():
    measured=json.loads((ROOT/'analysis/iso8_hybrid_heldout_confirmation_v1.json').read_text())
    grid=json.loads((ROOT/'configs/hybrid_static_search_frozen_v1.json').read_text())
    lines=['# Genuine 2D Hybrid: 성능과 강한 대조군','',
        '이 문서는 C0, C3 Controlled Output과 C3 Natural400을 구분한다.',
        '실측, Structural Accounting과 Offline Oracle을 같은 결과로 합산하지 않는다.','',
        '## C0 Exact Iso-work Confirmation','',
        'Layer8, Token8과 Hybrid4/Suffix 약 L/8의 실제 P/D Partition Rows를 검사했다.',
        'Prompt KV와 Query/Layer Coverage Gate도 통과해야 유효한 Cell로 처리한다.',
        'Actual Batch는 1/1/1이다. Source Credit은 4다.',
        '모든 Worker는 BI0/BF16/B8192/S1024/KV42GiB/Block128이다.',
        'Cell당 원본 Prompt는 16개다. 각 Cell은 독립 Server Start 3회를 예정했다.',
        '비율은 Start별 Metric의 평균을 비교한다. CI는 Paired Start Bootstrap 27개 재표본으로 계산했다.',
        'Pointwise Empirical 95% CI다. 다중 비교 보정을 하지 않았다.',
        'Native는 D에서 마지막 Prompt Query를 다시 계산한다. Native를 같은 Offloaded Rows의 Arm이라고 부르지 않는다.','',
        '| Input Bucket | Output | QPS | 완료 Paired Start | Hybrid/Layer TTFT p95 [95% CI] | Hybrid/Token TTFT p95 [95% CI] | Hybrid/Native Goodput [95% CI] | 상태 |',
        '|---|---:|---:|---:|---|---|---|---|']
    for row in measured['cells']:
        test=row.get('comparisons',{})
        fields=[test.get('Hybrid_over_Layer',{}).get('TTFT_p95_ms'),
            test.get('Hybrid_over_Token',{}).get('TTFT_p95_ms'),
            test.get('Hybrid_over_Native',{}).get('SLO_qualified_output_tokens_per_second')]
        lines.append('| '+f"I{row['input_bucket']} | {row['controlled_output']} | {row['QPS']} | {row['paired_complete_starts']} | "+
            ' | '.join(ci(v) for v in fields)+' | '+row['status']+' |')
    lines += ['','Hybrid의 TTFT CI가 두 Pure Axis보다 낮아도 Best-tuned Axis 전체를 이긴 것은 아니다.',
        '해당 비교는 같은 Iso-work Frontier다. 다른 Cut의 최적 Static은 별도 대조군이다.',
        '같은 Rows도 Attention, LM Head, Kernel Launch와 Transfer 비용을 같게 만들지 않는다.','',
        '### 모든 완료 Cell의 Decode Gate','',
        '| Input / Output | Axis | Native-relative ITL/TPOT 5% Gate | Absolute Decode≥95% | Native-relative TTFT CI Win |',
        '|---|---|---|---|---|']
    for row in measured['cells']:
        if 'safety' not in row:continue
        for axis,value in row['safety'].items():
            test=row['comparisons'].get(axis+'_over_Native',{}).get('TTFT_p95_ms')
            win=test is not None and test['CI95'][1]<1
            lines.append(f"| I{row['input_bucket']}/O{row['controlled_output']} | {axis} | {value['native_relative_decode_5percent']} | {value['absolute_decode_attainment_95percent']} | {win} |")
    lines += ['','## C0의 더 강한 Static Discovery','',
        '다음 표는 8개 Workload Cell에서 실행한 200-start, 25-action Discovery를 요약한다.',
        '각 Cell은 Native와 24개 Non-native 2D Action을 포함한다.',
        'Layer-only, Token-only Boundary와 Interior Hybrid를 모두 유지했다.',
        '각 Family의 최소 TTFT p95를 Decode Point Gate 통과 후보 안에서 골랐다.',
        '이 표는 독립 Start 1회다. Confidence-interval Winner나 Held-out Best-static Serving Gain이 아니다.',
        '서로 다른 Action은 작업량이 다르다. 이 표를 Exact Iso-work로 표시하지 않는다.','',
        '| Input / Output | Family | 적격/전체 후보 | Discovery 최소 TTFT p95(ms) | Goodput(output tokens/s) | m / Suffix Fraction | Raw Name |',
        '|---|---|---:|---:|---:|---|---|']
    grid_rows=[]
    for cell in grid['cells']:
        native_path=ROOT/'analysis'/(cell['Native']+'_metrics.json')
        if not native_path.exists():continue
        native=json.loads(native_path.read_text())
        entries=[dict(name=cell['Native'],family='Native',m=0,fraction=0)]
        entries += [dict(a,family={'LAYER_ONLY':'Layer','TOKEN_ONLY':'Token','INTERIOR_HYBRID':'Hybrid'}[a['boundary']]) for a in cell['actions']]
        results=[]
        for entry in entries:
            path=ROOT/'analysis'/(entry['name']+'_metrics.json')
            raw=ROOT/'raw'/entry['name']/'dispatcher_result.json'
            if not path.exists() or not raw.exists() or not json.loads(raw.read_text())['complete']:
                results.append(dict(entry,status='NOT_RUN_OR_FAILED'));continue
            metric=json.loads(path.read_text())
            assert metric['runtime_source_archive_SHA256']==grid['source_capsule_SHA256']
            eligible=metric['decode_SLO_attainment']>=.95 and all(metric[k]<=1.05*native[k] for k in ('ITL_p99_ms','TPOT_p95_ms'))
            results.append(dict(entry,status='MEASURED_ONLINE',eligible=eligible,metrics=metric))
        for family in ('Native','Layer','Token','Hybrid'):
            group=[r for r in results if r['family']==family]
            eligible=[r for r in group if r.get('eligible')]
            best=min(eligible,key=lambda r:r['metrics']['TTFT_p95_ms']) if eligible else None
            if best:
                m=best['metrics']
                lines.append(f"| I{cell['input_bucket']}/O{cell['output']} | {family} | {len(eligible)}/{len(group)} | {m['TTFT_p95_ms']:.3f} | {m['SLO_qualified_output_tokens_per_second']:.3f} | {best['m']} / {best['fraction']:.6f} | {best['name']} |")
            else:lines.append(f"| I{cell['input_bucket']}/O{cell['output']} | {family} | 0/{len(group)} | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |")
        grid_rows.append(dict(input_bucket=cell['input_bucket'],controlled_output=cell['output'],QPS=cell['QPS'],results=results))
    lines += ['','Discovery의 Token-only Boundary는 Suffix Fraction 1/16, 1/8, 1/4, 1/3을 포함한다.',
        '이 Boundary를 Last2/4/8/12의 Exact Rows Control과 같은 Partition 크기로 간주하지 않는다.',
        '이 Static Grid의 QPS는 기존 Serial Service Reference에 기반한다.',
        '따라서 QPS를 측정된 Sustainable Capacity의 고정 비율로 표현하지 않는다.','',
        '## Output만 바꾼 반증','',
        'I3의 별도 검사는 QPS1.253873, Source, Pair, Prompt와 Arrival을 고정했다.',
        'Output16의 Hybrid/Token TTFT p95는 1.070249 [1.054657,1.081369]였다.',
        'Output64에서는 0.980561 [0.971238,0.990277]였다.',
        '각각 Paired Independent Start 3회다. 이 Controlled Output 역전은 C0 범위다.',
        '이를 Natural400 또는 C3 Serving의 역전으로 옮기지 않는다.','',
        '## 실제 Continuous Bridge와 Natural400','',
        'C3 Controlled Bridge는 Output16/64와 D Cap1/2를 사전에 동결했다.',
        '강한 Compiled Stock Token과 진단용 Mode0 Token을 모두 포함한다.',
        'C0의 Hybrid 후보가 느린 Token Control만 이긴다면 Strong Serving GO로 처리하지 않는다.',
        'Actual Continuous Batch, Padding, Compile Family와 Credit도 별도로 기록한다.',
        '해당 결과는 [Controlled C3 Bridge](dense_ar_longbench400_controlled_continuous_bridge.md)에 기록한다.',
        'Main Natural400은 별도의 Discovery-frozen Static 선택과 독립 Arrival Seed를 사용한다.',
        '최종 Serving Gate는 NativeStrong과 같은 D 정책의 Native 대비 두 Decode CI를 모두 요구한다.',
        'C3 성능이 아직 미완료이면 C0 성능만으로 그 Gate를 통과 처리하지 않는다.','',
        '## 근거','',
        '- [Exact Iso-work Confirmation](dense_ar_longbench400_2d_20261121/analysis/iso8_hybrid_heldout_confirmation_v1.json)',
        '- [전체 Static Discovery 설계](dense_ar_longbench400_2d_20261121/configs/hybrid_static_search_frozen_v1.json)',
        '- [Static Discovery 전체 결과](dense_ar_longbench400_2d_20261121/analysis/c0_hybrid_static_discovery_full_ledger_v1.json)',
        '- [Exact Output Counterfactual](dense_ar_longbench400_2d_20261121/analysis/iso8_I3_exact_output_counterfactual_v1.json)',
        '- [Final C3 설계](dense_ar_longbench400_2d_20261121/configs/final_continuous_execution_plan_v1.json)','']
    from .common import write
    write(ROOT/'analysis/c0_hybrid_static_discovery_full_ledger_v1.json',dict(cells=grid_rows,
        evidence='MEASURED_ONLINE_DISCOVERY',scope='Every measured and failed/unsafe action remains. Point eligibility is not an independent-start safety CI.'))
    path=REPO/'poc_flashvep/reports/dense_ar_layer_token_2d_hybrid_performance.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print('HYBRID_PERFORMANCE_REPORT_UPDATED',measured['completed_cells'],'iso-work cells',flush=True)


if __name__=='__main__':
    main()
