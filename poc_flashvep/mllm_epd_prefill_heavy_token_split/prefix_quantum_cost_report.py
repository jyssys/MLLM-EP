"""Report all diagnostic E/D costs and original embedding import granularity."""
import csv
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write


def main():
    preserve();summaries=[];joined=[];provenance={}
    for arm,tag,worker,gpu in [('E','prefix_QUANTA_C26R1','LE4',4),('D','prefix_QUANTA_C26R2','LD6',6)]:
        name=f'{tag}_{arm}_CPC';path=ROOT/'analysis'/f'{name}_cost.json'
        cost=json.loads(path.read_text());assert not cost['observer_gate_passed']
        lease=ROOT/'state'/f'server_{tag}_{worker}.json';server=json.loads(lease.read_text())
        assert server['finished']
        transport=ROOT/'raw'/f'ec_cuda_ipc_pd_gpu{gpu}.jsonl'
        imports=[json.loads(line) for line in transport.read_text().splitlines()]
        with (ROOT/'analysis'/f'{name}_requests.csv').open() as stream:requests=list(csv.DictReader(stream))
        assert len(requests)==140
        for request in requests:
            lo,hi=float(request['begin']),float(request['end'])
            matches=[r for r in imports if lo<=r['begin']<=r['end']<=hi]
            assert len(matches)==1,(request['run'],len(matches))
            item=matches[0];assert item['images']==1 and item['clone_complete_before_ACK']
            joined.append(dict(arm=arm,run=request['run'],stage=request['stage'],quantum=int(request['quantum']),
                consumer_pid=item['consumer_pid'],producer_pid=item['producer_pid'],images_imported=item['images'],
                logical_embedding_bytes=item['logical_bytes'],
                import_host_ms=(item['end']-item['begin'])*1000,
                export_RPC_ms=item['export_RPC_ms'],import_clone_sync_host_ms=item['import_clone_sync_host_ms'],
                ACK_host_ms=item['completion_ACK_host_ms'],clone_complete_before_ACK=True,
                evidence='DIAGNOSTIC host enclosure; not GPU-exclusive transfer'))
        selected=[r for r in joined if r['arm']==arm]
        assert len({r['consumer_pid'] for r in selected})==1
        for quantum in [128,256,512,1024,2048]:
            clean=[float(r['HTTP_wall_ms']) for r in requests if int(r['quantum'])==quantum and r['stage'].startswith('clean')]
            events=[r['GPU_enclosure_ms'] for r in cost['events'] if r['quantum']==quantum]
            imports_for_q=[r for r in selected if r['quantum']==quantum and r['stage']=='profile']
            assert len(clean)==18 and len(events)==len(imports_for_q)==9
            checks=[r for r in cost['comparisons'] if r['quantum']==quantum]
            assert len(checks)==2
            summaries.append(dict(arm=arm,GPU=gpu,quantum=quantum,independent_starts=1,
                actual_batch=1,clean_HTTP_samples=18,profile_samples=9,
                clean_HTTP_p50_ms=float(np.median(clean)),
                profile_execute_enclosure_p50_ms=float(np.median(events)),
                profile_execute_enclosure_p95_ms=float(np.percentile(events,95)),
                profile_import_host_p50_ms=float(np.median([r['import_host_ms'] for r in imports_for_q])),
                logical_embedding_bytes=imports_for_q[0]['logical_embedding_bytes'],
                cell_both_HTTP_gates_pass=all(r['observer_gate_passed'] for r in checks),
                whole_arm_gate_pass=False,evidence='DIAGNOSTIC',source_archive_SHA256=server['source_archive_SHA256']))
        provenance[arm]=dict(cost_SHA256=digest(path),lease_SHA256=digest(lease),
            source_archive_SHA256=server['source_archive_SHA256'],GPU_info=server['GPU_info'],
            original_shared_import_log_SHA256=digest(transport),matched_imports=140,
            consumer_pid=selected[0]['consumer_pid'],empty_control_events=len(cost['empty_schedule_control_events']),
            empty_control_total_execute_enclosure_ms=cost['empty_schedule_control_total_enclosure_ms'])
    assert {r['logical_embedding_bytes'] for r in joined}=={19289088}
    for stem,records in [('summary',summaries),('embedding_imports',joined)]:
        with (ROOT/'analysis'/f'prefix_quanta_E_D_C26_{stem}_v0.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(ROOT/'analysis/prefix_quanta_E_D_C26_provenance_v0.json',dict(time=time.time(),arms=provenance,
        scope='Different independent initializations. Not a paired E/D CI or actual online safety experiment.',
        evidence='DIAGNOSTIC',model_revision='cc594898137f460bfe9f0759e9844b3ce807cfb5',
        full_input_tokens=2961,original_images_encoded=2,first_image_span=[15,2706],
        prefix_has_no_imported_KV=True,task_completion=False,safe_quantum='NOT_MEASURED'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in ['E','D']:
        items=[r for r in summaries if r['arm']==arm];x=[r['quantum'] for r in items]
        axes[0].plot(x,[r['profile_execute_enclosure_p95_ms'] for r in items],marker='o',label=arm+' helper')
        axes[1].plot(x,[r['clean_HTTP_p50_ms'] for r in items],marker='o',label=arm+' helper')
    axes[0].set_ylabel('Execute enclosure p95 (ms), not exclusive kernels')
    axes[1].set_ylabel('Clean HTTP median (ms), includes original E')
    for axis in axes:axis.set_xlabel('Actual prefix query tokens');axis.grid(alpha=.2);axis.legend()
    figure.suptitle('DIAGNOSTIC | CPC arm gates failed | No safe-window or serving-gain claim');figure.tight_layout()
    for extension in ['png','pdf']:figure.savefig(ROOT/'figures'/f'prefix_quanta_E_D_C26_cost_v0.{extension}',dpi=180)
    plt.close(figure)
    report=ROOT.parent/'mllm_epd_encoder_decoder_prefix_quantum_cost_v1.md';assert not report.exists()
    lines=['# E/D Prefix Quantum 비용 — 진단용 측정','',
        '두 Arm의 전체 Clean/Profile/Clean Gate는 실패했다. 비용은 모두 `DIAGNOSTIC`으로 보존했다.',
        'E는 주로 비계측 실행 간 변동 때문에 실패했다. D도 일부 Quantum의 Tail/변동 기준을 통과하지 못했다.',
        '이 결과로 Safe Quantum, Online Speedup 또는 Encoder/Decode 비열등성을 주장하지 않는다.','',
        '## 계산과 측정 계약','',
        'Original Qwen2.5-VL-7B BF16, BI0, Full28과 원본 Processor를 사용했다.',
        '모델과 Processor Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다.',
        '동일 Calibration Request6의 Input2961 Tokens와 이미지 2개를 유지했다.',
        'E는 매 요청에서 원본 이미지 2개를 실제 Encoding했다. Helper는 명시한 Prefix만 처리했다.',
        'GPU4는 E와 추가 원본 Full-model Helper를 함께 실행했다. GPU6은 같은 Helper를 독립적으로 실행했다.','',
        'B8192/S1024, Helper KV8GiB와 실제 Batch1을 사용했다. Prefix에는 가져온 KV가 없다.',
        'Output1 Token은 폐기했다. 완성된 Task Answer와 Natural EOS Serving이 아니다.',
        '조건별 독립 초기화는 1회다. 같은 초기화 안의 반복 9개를 독립 Start9회로 계산하지 않았다.',
        'E/D 초기화도 서로 다르다. E/D Paired CI 또는 물리적 GPU 우위로 해석하지 않는다.','',
        '## 전체 Cell 결과','',
        'HTTP Median은 비계측 표본 18개다. Execute p50/p95는 계측 표본 9개다.',
        'Execute Event에는 Host Gap, Dependency와 Embedding Import가 포함될 수 있다.',
        'Import Host 시간은 Execute와 중첩될 수 있다. 두 값을 더해 Total GPU Time으로 만들지 않는다.','',
        '| Helper | Prefix | Clean HTTP p50 ms | Execute p50 ms | Execute p95 ms | Import Host p50 ms | Local HTTP Gates | Arm Gate |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for row in summaries:
        lines.append(f"| {row['arm']} | {row['quantum']} | {row['clean_HTTP_p50_ms']:.3f} | "
            f"{row['profile_execute_enclosure_p50_ms']:.3f} | {row['profile_execute_enclosure_p95_ms']:.3f} | "
            f"{row['profile_import_host_p50_ms']:.3f} | {'PASS' if row['cell_both_HTTP_gates_pass'] else 'FAIL'} | FAIL |")
    lines+=['','일부 D Cell의 두 HTTP 검사가 통과해도 전체 Arm을 PASS로 바꾸지 않았다.',
        '실패 Cell과 Outlier는 모두 CSV에 남겼다. 유리한 표본만 제거하지 않았다.','',
        '## 실제 Embedding 전달 단위','',
        '모든 Quantum은 첫 이미지 Span15:2706 안에서 끝난다. 해당 이미지의 Visual Embedding은 2691×3584 BF16이다.',
        '실제 Import Log는 모든 표본에서 이미지 1개와 19,289,088 Bytes를 기록했다.',
        'E/D 각 140개 HTTP 구간과 같은 Monotonic Clock의 Import 구간을 일대일로 연결했다.',
        '각 Arm의 Consumer PID가 하나인지 확인했다. Clone 완료 ACK도 모든 행에서 유지했다.',
        '따라서 Prefix128을 처리해도 이 Runtime은 첫 이미지 Embedding 전체를 Import했다.','',
        '이 관측은 Runtime의 Image-level 전달 단위에 관한 것이다. Visual Token의 계산이 본질적으로 더 비싸다는 증거가 아니다.',
        '이미지 내부 Exact Split의 수학적 정합성과 전달 효율은 다른 문제다.',
        '전달 단위를 변경하는 Modality-aware Method는 구현하지 않았다. Novelty도 주장하지 않는다.','',
        '## 정리 단계와 실패 보존','',
        'E/D 각 계측에는 요청 실행 45개와 요청이 없는 Engine 정리 실행 90개가 있었다.',
        '정리 단계의 Event는 별도 항목으로 보존했다. Prefill GPU 비용에 임의로 배분하지 않았다.',
        'E의 최초 분석은 이 정리 단계를 예상하지 못해 종료했다. 원시 측정은 이미 완료된 상태였다.',
        '실제 Event 패턴의 CPU Red/Green 검사 뒤, E 데이터는 GPU 재실행 없이 복구했다.',
        'D 비용만 별도 실행했다. E의 실패 Gate와 기존 Raw를 덮어쓰지 않았다.','',
        '## 한계와 재현 파일','',
        'Suffix Continuation은 Prefix Length와 가져온 KV에 따라 비용이 달라진다. 이 Prefix 비용으로 대체하지 않는다.',
        '실제 Encode/Decode Queue가 있는 Online Intervention은 `NOT_RUN`이다.',
        'Future Idle Window에 맞춘 안전한 Quantum 또는 Encoder/Decode SLO 보호도 미검증이다.',
        '요청·Gate 원시는 `analysis/prefix_QUANTA_C26R{1_E,2_D}_CPC_{requests,qualification}.csv`에 있다.',
        '비용과 정리 Event는 대응 `_cost.json`에 있다.',
        '결합 CSV는 `analysis/prefix_quanta_E_D_C26_{summary,embedding_imports}_v0.csv`다.',
        'Figure는 `figures/prefix_quanta_E_D_C26_cost_v0.{png,pdf}`다. Source SHA와 GPU 정보는 Provenance JSON에 있다.',
        '모든 경로는 `mllm_epd_prefill_heavy_token_split_20261124` Namespace 기준이다.','']
    report.write_text('\n'.join(lines));preserve()
    print('PREFIX_QUANTA_DIAGNOSTIC_COST_REPORTED',len(summaries),len(joined),flush=True)


if __name__=='__main__':main()
