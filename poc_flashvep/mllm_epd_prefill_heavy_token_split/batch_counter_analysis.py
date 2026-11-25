"""Report counter-enabled C3 geometry without backfilling unobserved clean runs."""
import json
from pathlib import Path
import time

from .common import ROOT,digest,preserve,write
from .batch_counter_control import TAG,METHODS,qualify
from .campaign_metric_audit import cohort_contract
from .token_discovery_analysis import table


def summarize_worker(value,role):
    rows=value['prepared_query_histogram'];calls=sum(row['calls'] for row in rows)
    for row in rows:
        assert row['requests']==row['prompt_only_requests']+row['decode_only_requests']+row['mixed_requests']
        assert row['query_rows']==row['prompt_query_rows']+row['decode_query_rows']
        assert row['calls']>0 and row['requests']>0
    valid=value['valid'] and value['original_methods_restored'] and not value['errors']
    if role!='encoder':valid=valid and calls>0
    result=dict(role=role,prepared_calls=calls,worker_calls=sum(r['calls'] for r in value['worker_schedule_histogram']),
        empty_worker_control_calls=sum(r['calls'] for r in value['worker_schedule_histogram'] if r['scheduled_query_rows']==0),
        encoder_output_calls=sum(r['calls'] for r in value['encoder_output_histogram']),counter_valid=bool(valid),
        maximum_prepared_requests=max((r['requests'] for r in rows),default=None),
        maximum_prompt_requests=max((r['prompt_only_requests']+r['mixed_requests'] for r in rows),default=None),
        maximum_decode_requests=max((r['decode_only_requests']+r['mixed_requests'] for r in rows),default=None),
        maximum_query_rows=max((r['query_rows'] for r in rows),default=None),
        mean_prepared_requests_per_call=sum(r['requests']*r['calls'] for r in rows)/calls if calls else None,
        total_prompt_query_rows=sum(r['prompt_query_rows']*r['calls'] for r in rows),
        total_decode_query_rows=sum(r['decode_query_rows']*r['calls'] for r in rows),
        scope='Logical prepared query geometry. Call-weighted, not time-weighted or CUDA-graph padded physical GEMM shapes.')
    return result


def main():
    preserve();source=ROOT/'analysis/batch_COUNTER_C46R1_summary.json';summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE_COLLECTION' and len(summary['results'])==24
    assert (ROOT/'state/batch_COUNTER_C46R1_controller_complete.json').exists()
    prior=ROOT/'state/batch_COUNTER_C46_failure.json'
    assert digest(prior)==summary['previous_failure_SHA256']
    target=ROOT/'analysis'/f'{TAG}_audit_v0.json';assert not target.exists()
    node_rows=[];histograms=[];encoder_histograms=[];gates=[];metric_checks=[];provenance=[]
    for method in METHODS:
        tag=summary['method_tags'][method];qualification=ROOT/'analysis'/f'{tag}_counter_qualification.json'
        info=json.loads(qualification.read_text());replies=json.loads((ROOT/'state'/f'{tag}_counter_RPC.json').read_text())
        assert info['independent_initializations']==1 and info['task_quality']=='PENDING' and not info['full_GO']
        expected=qualify(info['stages'],info['collector_valid'])
        assert expected['comparisons']==info['comparisons'] and expected['observer_gate_passed']==info['observer_gate_passed']
        all_geometry=True
        leases=[json.loads(p.read_text()) for p in (ROOT/'state').glob(f'server_{tag}_*.json') if not p.name.endswith('.source_snapshot.json')]
        assert len(leases)==3 and all(r['finished'] for r in leases)
        by_port={str(r['port']):r for r in leases};assert len(by_port)==3
        for port,reply in replies.items():
            assert len(reply['begin'])==len(reply['end'])==1
            end=reply['end'][0];folder=Path(end['folder'])
            assert ROOT in folder.parents and folder==Path(reply['begin'][0]['folder'])
            raw=folder/'histograms.json';value=json.loads(raw.read_text());lease=by_port[port]
            stats=summarize_worker(value,lease['role']);all_geometry=all_geometry and stats['counter_valid']
            node_rows.append(dict(method=method,GPU=lease['gpu'],GPU_info=lease['GPU_info'].strip(),**stats,
                qualification_scope='See method-level observer gate. Full quality pending. Old clean N3 geometry remains unobserved.'))
            for item in value['prepared_query_histogram']:
                histograms.append(dict(method=method,GPU=lease['gpu'],role=lease['role'],**item,
                    evidence='Actual prepared-step histogram, not a request-level probability distribution.'))
            for item in value['encoder_output_histogram']:
                encoder_histograms.append(dict(method=method,GPU=lease['gpu'],role=lease['role'],**item,
                    scope='Function output images/visual rows. Only original encoder or aggregated roles count as actual vision work.'))
            provenance.append(dict(method=method,GPU=lease['gpu'],GPU_info=lease['GPU_info'].strip(),raw_SHA256=digest(raw),
                source_archive_SHA256=lease['source_archive_SHA256'],counter_code_SHA256=lease['source_SHA256']['batch_counter.py']))
        for cell in [r for r in summary['results'] if r['method']==method]:
            run=cell['run'];file=ROOT/'raw'/f'{run}_requests.jsonl';rows=[json.loads(line) for line in file.read_text().splitlines()]
            done=json.loads((ROOT/'state'/f'{run}_complete.json').read_text());metric=cell['metrics']
            assert digest(file)==done['raw_SHA256']==metric['raw_SHA256']
            metric_checks.append(dict(method=method,run=run,**cohort_contract(rows,done,metric)))
        gates.append(dict(method=method,independent_initializations=1,observer_gate_passed=info['observer_gate_passed'],
            complete_geometry=all_geometry,encoder_output_conservation=info['encoder_output_conservation']['passed'],
            evidence='TRACE_ANALYSIS_QUALIFIED_COUNTER' if info['observer_gate_passed'] and all_geometry else 'DIAGNOSTIC',
            task_quality='PENDING',full_GO=False,clean_N3_batch_backfill=False))
    for name,rows in [('worker_geometry',node_rows),('prepared_histograms',histograms),('encoder_histograms',encoder_histograms),
            ('gates',gates),('metric_recalculation',metric_checks),('provenance',provenance)]:
        table(ROOT/'analysis'/f'{TAG}_{name}_v0.csv',rows)
    write(target,dict(time=time.time(),source_SHA256=digest(source),gates=gates,provenance=provenance,
        methods=len(METHODS),independent_initializations_per_method=1,model_quality='PENDING',full_GO=False,
        metric_recalculation='PASS',old_clean_N3_actual_batch='NOT_MEASURED',
        scope='Representative counter-enabled Replay CPC only. No new scientific serving winner, GPU kernel time, SM/HBM counter or future safe-window proof.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(13,4.8));figure_rows=[]
    for axis,key,label in zip(axes,['maximum_prompt_requests','maximum_decode_requests'],['Max prepared prompt requests','Max prepared generation requests']):
        chosen=[r for r in node_rows if r[key] is not None]
        labels=[r['method']+' GPU'+str(r['GPU']) for r in chosen]
        axis.bar(range(len(chosen)),[r[key] for r in chosen]);axis.set_xticks(range(len(chosen)),labels,rotation=70,ha='right')
        axis.set_ylabel(label);axis.grid(axis='y',alpha=.2)
        figure_rows.extend(dict(method=r['method'],GPU=r['GPU'],role=r['role'],metric=key,value=r[key],
            evidence=next(g['evidence'] for g in gates if g['method']==r['method']),independent_initializations=1) for r in chosen)
    figure.suptitle('Counter-enabled C3 Replay | N1 CPC per method | Gate labels in CSV | Not old clean N3 batch shapes')
    figure.tight_layout();name=TAG+'_actual_batch_v0'
    for extension in ['png','pdf']:figure.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
    plt.close(figure);table(ROOT/'figures'/f'{name}.csv',figure_rows)
    target=ROOT.parent/'mllm_epd_actual_batch_counter_validation_v1.md';assert not target.exists()
    lines=['# 실제 Prepared Batch: Counter-only C3 검사','',
        'C46은 Agg DP3의 V2 Runner Counter API에서 실패했다. 실패 Raw와 종료 상태는 보존했다.',
        '완료된 NativeStrong·NativeCommon·Static10·Idle10은 반복하지 않았다.',
        'C46R1은 V2 Counter만 수정하고 Agg DP3·E+PD DP2를 새 초기화로 검사했다.',
        'V2는 반환된 InputBatch와 중첩 EncoderRunner 출력을 읽었다. V1 Hook과 모델 실행은 유지했다.',
        '각 Method는 같은 GPU4/5/6에서 새 초기화 1회로 실행했다. 독립 N3가 아니다.',
        '동일 Main100·QPS2·요청별 Output 수를 사용했다. Natural EOS 결과와 합쳐 평균내지 않았다.',
        '측정 순서는 Warm-up→Clean Before→Counter→Clean After다.',
        'CUPTI·CUDA Event·추가 Tensor Copy·GPU 동기화는 사용하지 않았다.','',
        '## 계측 Gate','',
        '| Method | Observer≤5% | Complete geometry | Actual E outputs | Evidence |','|---|---|---|---|---|']
    for row in gates:lines.append(f'| {row["method"]} | {row["observer_gate_passed"]} | {row["complete_geometry"]} | {row["encoder_output_conservation"]} | {row["evidence"]} |')
    lines+=['','전체 다섯 Metric의 Profile 차이와 Clean Spread가 각각 5% 이하여야 적격이다.',
        '하나라도 실패하면 해당 Method의 계측은 DIAGNOSTIC이다. 유리한 Metric만 선택하지 않았다.',
        '실제 Encoder 또는 Aggregated 역할의 이미지 수·Visual Row 합계도 원래 Main100과 일치해야 한다.',
        '이 Gate를 통과해도 전체 Task Quality는 PENDING이다.','',
        '## Actual logical batch','',
        '| Method | GPU | Role | Prepared calls | Max requests | Max prompt requests | Max generation requests | Max query rows | Mean requests/call |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in node_rows:lines.append(f'| {row["method"]} | {row["GPU"]} | {row["role"]} | {row["prepared_calls"]} | '+
        ' | '.join(str(row[key]) for key in ['maximum_prepared_requests','maximum_prompt_requests','maximum_decode_requests','maximum_query_rows','mean_prepared_requests_per_call'])+' |')
    lines+=['','Prompt와 Generation Row는 실제 입력 준비 직전의 CPU Computed/Prompt 경계로 나눴다.',
        'Scheduler가 먼저 Computed Count를 증가시켰다고 마지막 Prompt Chunk를 Decode로 바꾸지 않았다.',
        '한 요청이 두 종류의 Query를 함께 포함하면 Mixed Request로 보존했다.',
        'Native D의 첫 요청도 실제 준비한 Query 위치를 기준으로 분류했다. Stage 이름만으로 모든 Query를 Decode라 부르지 않았다.',
        'E-only의 Scheduled Token 수를 LLM Forward 작업량에 더하지 않았다.',
        'CUDA Graph의 Padding된 Kernel Shape와 Logical Query Geometry는 다르다.',
        '빈 Worker Cleanup은 별도 Count다. GPU Prefill Service나 실제 요청 Batch1로 바꾸지 않았다.',
        '평균은 Prepared Call 가중이다. 시간 가중 평균이나 요청 가중 평균이 아니다.',
        '이 검사는 과거 Clean N3의 실제 Batch를 소급 측정하지 않는다.',
        '모델·Compiler·Admission·Dataset·SLO는 원래 Family와 같다. Hook은 각 Counter 구간 후 원래 Method로 복원했다.',
        'Exclusive Kernel Time·Queue Timeline·SM/HBM Counter·미래 Safe Window는 이 Counter의 측정 범위가 아니다.','',
        '## 재현 자료','',
        f'Config는 configs/{TAG}.json이다. Worker별 원시 Histogram은 raw/batch_counter/{TAG}_B0_*/에 있다.',
        f'원시 Raw Metric 재계산·Histogram CSV·Lease SHA는 analysis/{TAG}_*_v0에 있다.',
        f'PNG/PDF와 대응 CSV는 figures/{TAG}_actual_batch_v0에 있다.',
        'Original BF16 Qwen2.5-VL-7B Revision cc594898137f460bfe9f0759e9844b3ce807cfb5를 사용했다.',
        'P/D·Replica B8192/S1024, KV42GiB, Block128, Prefix Cache Off와 원래 Vision·Processor를 유지했다.',
        '이 결과는 새로운 Method의 성능 개선이나 독립 Serving GO가 아니다.','']
    target.write_text('\n'.join(lines));preserve();print('ACTUAL_BATCH_COUNTER_CPC_RECALCULATED',flush=True)


if __name__=='__main__':main()
