"""Recompute disjoint host stages and preserve negative EC transport results."""
import csv
import json
import statistics
import time

from .common import ROOT,digest,preserve,write


def save_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def stages(run,proxy):
    clients=[json.loads(s) for s in (ROOT/'raw'/f'{run}_requests.jsonl').read_text().splitlines()]
    traces={r['runtime_request_id']:r for r in map(json.loads,(ROOT/'raw'/f'proxy_{proxy}.jsonl').read_text().splitlines())
            if r['runtime_request_id'].startswith(run+'_')}
    rows=[]
    for row in clients:
        trace=traces[row['runtime_request_id']]
        assert row['status']==trace['status']=='MEASURED_ONLINE'
        points=[row['scheduled_arrival'],trace['E_submit'],trace['E_complete'],trace['P_submit'],
                trace['P_ready'],trace['D_submit'],row['token_times'][0]]
        names=['arrival_to_E_submit','E_HTTP_stage','E_to_P_dispatch','P_HTTP_stage','P_to_D_dispatch','D_submit_to_client_first_token']
        item=dict(run=run,request_id=row['request_id'],TTFT_ms=row['TTFT_ms'])
        item.update({key:(b-a)*1000 for key,a,b in zip(names,points,points[1:])})
        assert all(item[key]>=0 for key in names)
        assert abs(sum(item[key] for key in names)-row['TTFT_ms'])<1e-6
        rows.append(item)
    assert len(rows)==100
    return rows,{key:statistics.mean(row[key] for row in rows) for key in names}


def main():
    preserve();rows=[];stage_rows=[];means=[]
    for tag,proxy,method in [('ec_CPU_C6','ec_CPU_C6_proxy','CPU staging'),('ec_IPC_C8','ec_IPC_C8_proxy','On-demand CUDA IPC')]:
        for mode in ['replay_q2','replay_q3','natural_q2']:
            run=f'{tag}_{mode}_v0'
            value=json.loads((ROOT/'analysis'/f'{run}_metrics.json').read_text())
            rows.append(dict(method=method,run=run,contract=mode,QPS=3 if mode.endswith('q3') else 2,starts=1,
                requests=value['completed_count'],output_tokens=value['total_output_tokens'],
                TTFT_p50_ms=value['TTFT_p50_ms'],TTFT_p95_ms=value['TTFT_p95_ms'],
                ITL_p99_ms=value['ITL_p99_ms'],ITL_max_ms=value['ITL_max_ms'],TPOT_p95_ms=value['TPOT_p95_ms'],
                SLO_goodput=value['SLO_goodput'],all_ITL_pass=value['gate_attainment']['ITL_PASS'],
                joint_pass=value['gate_attainment']['JOINT_PASS'],raw_SHA256=value['raw_SHA256']))
            details,mean=stages(run,proxy);stage_rows.extend(details)
            means.append(dict(method=method,run=run,contract=mode,TTFT_mean_ms=value['TTFT_mean_ms'],**mean))
    save_csv(ROOT/'analysis/native_EC_transport_controls_v1.csv',rows)
    save_csv(ROOT/'analysis/native_EC_disjoint_request_host_stages_v1.csv',stage_rows)
    save_csv(ROOT/'analysis/native_EC_disjoint_mean_host_stages_v1.csv',means)
    loads=[json.loads(s) for s in (ROOT/'raw/ec_cuda_ipc_prefill_gpu5.jsonl').read_text().splitlines()]
    ack=[]
    for label,mode in [('A','rpc'),('B','file'),('C','rpc')]:
        run=f'ec_ACK_C9_{label}_{mode}_replay100_q2_v0'
        done=json.loads((ROOT/'state'/f'{run}_complete.json').read_text());lo=done['monotonic_start'];hi=lo+done['duration_seconds']
        selected=[r for r in loads if lo<=r['begin'] and r['end']<=hi]
        assert selected
        value=json.loads((ROOT/'analysis'/f'{run}_metrics.json').read_text())
        ack.append(dict(arm=label,mode=mode,run=run,loads=len(selected),requests=100,output_tokens=value['total_output_tokens'],
            TTFT_p95_ms=value['TTFT_p95_ms'],ITL_p99_ms=value['ITL_p99_ms'],TPOT_p95_ms=value['TPOT_p95_ms'],
            all_ITL_pass=value['gate_attainment']['ITL_PASS'],SLO_goodput=value['SLO_goodput'],
            **{key:statistics.mean(r[key] for r in selected) for key in
                ['export_RPC_ms','import_clone_sync_host_ms','completion_ACK_host_ms']}))
    save_csv(ROOT/'analysis/native_EC_ACK_controls_v1.csv',ack)
    write(ROOT/'analysis/native_EC_transport_and_ACK_v1.json',dict(time=time.time(),transport_rows=rows,mean_host_stages=means,
        ACK_rows=ack,host_additive_identity_requests=len(stage_rows),host_additive_identity='PASS',
        evidence='DIAGNOSTIC_C3',independent_starts_per_transport_arm=1,ACK_independent_starts=1,
        native_quality='PENDING',GPU_exclusive_operator_times='NOT_MEASURED',generic_token_split='NOT_RUN',
        scope='No per-request GPU-exclusive attribution. Host stages sum to mean TTFT, not to TTFTp95. '
            'E HTTP completion is not a common GPU-ready boundary across synchronous CPU and asynchronous GPU transports.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    labels=['Replay q2','Replay q3','Natural q2']
    for ax,key,y in zip(axes,['TTFT_p95_ms','SLO_goodput'],['TTFT p95 (ms)','SLO output tokens/s']):
        for i,method in enumerate(['CPU staging','On-demand CUDA IPC']):
            values=[r[key] for r in rows if r['method']==method]
            ax.bar([j+(i-.5)*.34 for j in range(3)],values,width=.34,label=method)
        ax.set_xticks(range(3),labels);ax.set_ylabel(y);ax.legend()
    fig.suptitle('DIAGNOSTIC | Native EC transport | Main100 | One start per arm | No CI')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'native_EC_transport_controls_v1.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].bar(['RPC A','File B','RPC C'],[r['TTFT_p95_ms'] for r in ack]);axes[0].set_ylabel('TTFT p95 (ms)')
    bottom=[0.,0.,0.]
    for key,label in [('export_RPC_ms','Export RPC'),('import_clone_sync_host_ms','Import/clone/sync'),('completion_ACK_host_ms','Completion ACK')]:
        values=[r[key] for r in ack];axes[1].bar(['RPC A','File B','RPC C'],values,bottom=bottom,label=label)
        bottom=[a+b for a,b in zip(bottom,values)]
    axes[1].set_ylabel('Mean host enclosure per load (ms)');axes[1].legend()
    fig.suptitle('DIAGNOSTIC | One-init ACK intervention | Not independent N3 or GPU-exclusive cost')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'native_EC_ACK_controls_v1.{ext}',dpi=160)
    plt.close(fig)
    lines=['# MLLM Native EPD EC Transport and ACK Diagnostic','',
        '증거 수준은 DIAGNOSTIC_C3다. Generic Token Split이나 추가 Prefill 삽입은 실행하지 않았다.',
        'GPU4/5/6에서 실제 E Encoding, 전체28-Layer Prefill, NIXL KV 전달과 Decode를 실행했다.',
        'Model Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다. Original BF16과 BI0를 유지했다.',
        'Main100, Processor, 입력, 이미지 순서, KV42GiB/Block128, P/D B8192/S1024를 유지했다.',
        'E는 Cache0/Renderer2/B114688/S1024와 같은 원본 Vision Encoder를 사용했다.',
        'Source Credit이나 인위적인 D Cap은 추가하지 않았다. 영상·Pruning·Layer/Hybrid는 사용하지 않았다.','',
        '## 실제 실행 상태','',
        '첫 CUDA IPC 후보 C7은 실행에 실패했다. 모델 출력 오류로 분류하지 않는다.',
        '코덱이 메모리 Handle을 64바이트로 가정했다. 설치된 PyTorch가 반환한 Handle은 66바이트였다.',
        'Opaque 바이트를 자르지 않고 보존하도록 수정했다. 범위·형상·BF16·Storage 경계 검사는 유지했다.',
        '실패 Raw와 Source Archive는 보존했다. C7의 N8과 성능 측정은 NOT_RUN이다.',
        '수정한 C8은 N1과 N8 Natural EOS를 기존 같은 Geometry 기준선과 일치시켰다.',
        '실제 Mantis 요청1개의 MRoPE·입력 Embedding·Hidden과 전체28-Layer KV도 CPU 전달과 일치했다.',
        '이는 전체 Cross-geometry 품질 Gate를 통과했다는 뜻이 아니다. 기존 품질 한계는 유지한다.',
        '두 소비자와 비동기 Producer의 별도 전달8개 검사도 통과했다. 이는 Serving 속도 향상이 아니다.','',
        '## CPU 전달 대 On-demand GPU 전달','',
        '각 Arm은 독립 초기화1회다. 두 초기화를 독립 N3로 취급하지 않는다.',
        'Replay는 매 Arm 5,269 Tokens다. Natural의 실제 출력량은 별도로 기록했다.',
        '같은 도착·순서 Seed를 사용했다. 모든100개 요청을 Drain까지 측정했다.','',
        '| Method | Contract | Output Tokens | TTFT p95 ms | ITL p99 ms | TPOT p95 ms/token | All-ITL PASS % | SLO output tok/s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["contract"]} | {r["output_tokens"]} | {r["TTFT_p95_ms"]:.3f} | '
            f'{r["ITL_p99_ms"]:.3f} | {r["TPOT_p95_ms"]:.3f} | {100*r["all_ITL_pass"]:.2f} | {r["SLO_goodput"]:.3f} |')
    lines+=['','GPU 전달은 값 정합성을 유지했지만 이 구현에서는 TTFT와 Goodput을 악화시켰다.',
        'CPU 기준선을 약하게 바꾸거나 이 부정적 결과를 제거하지 않았다.',
        'GPU 전달에는 E GPU Tensor 보유와 최대12GiB의 유효 Backing Storage 상한이 있다.',
        'C8의 QPS3까지 관측한 Peak는 약4.024GB였다. 상한12GiB를 채워서 Admission이 막힌 결과는 아니다.',
        '단위 GB와 GiB는 다르다. 전체 Driver Memory나 Activation Peak와 이 보유량도 다르다.','',
        '## ACK의 한 요소 인과 검사','',
        '복사 완료 뒤에도 P가 E의 ACK RPC 응답을 기다리는 경로를 확인했다.',
        'File Arm은 GPU Clone 완료 동기화를 유지했다. 완료 Marker만 먼저 기록하고 P가 계산을 이어갔다.',
        'E는 성공한 D 완료 시점에 Marker를 검사한 뒤 Tensor를 해제했다. 조기 Credit 반환은 하지 않았다.',
        '동일 초기화에서 RPC→File→RPC를 실행했다. 이는 독립 Start3회가 아니다.','',
        '| Arm | TTFT p95 ms | Export RPC mean ms/load | Import/Clone/Sync mean ms/load | ACK mean ms/load | All-ITL PASS % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in ack:
        lines.append(f'| {r["arm"]}: {r["mode"]} | {r["TTFT_p95_ms"]:.3f} | {r["export_RPC_ms"]:.3f} | '
            f'{r["import_clone_sync_host_ms"]:.3f} | {r["completion_ACK_host_ms"]:.3f} | {100*r["all_ITL_pass"]:.2f} |')
    lines+=['','File ACK에서 TTFT p95가 줄었다. 다시 RPC로 바꾸자 지연이 증가했다.',
        '이 한 초기화는 동기 ACK 경로가 P Critical Path 지연에 기여함을 지지한다.',
        '개별 요청이나 Load149개를 독립 Server Start로 취급하지 않는다. 인과 기여율의 일반적인 CI는 없다.',
        'File ACK에도 CPU 기준선보다 높은 TTFT가 남았다. Export RPC와 Import·Clone·Sync 비용은 남아 있다.',
        'Import·Clone·Sync Host Enclosure에는 필요한 E GPU 준비 대기가 들어갈 수 있다.',
        '이를 순수 GPU 복사시간이나 제거 가능한 낭비 전부로 해석하지 않는다.','',
        '## Host 경계의 의미','',
        '600개 요청에서 서로 겹치지 않는6개 Host 구간의 합이 요청 TTFT와 일치했다.',
        '이는 평균 TTFT의 가법 분해다. TTFT p95의 가법 분해나 요청별 GPU-exclusive 비용이 아니다.',
        'E HTTP 완료는 두 Transport에서 같은 GPU-ready 경계가 아니다.',
        'CPU 저장은 GPU 완료를 기다린다. GPU Handle 발행은 CUDA Event를 통해 실제 준비 완료를 기다릴 수 있다.',
        '따라서 E HTTP 구간 감소를 Encoder GPU 계산 절감으로 해석하지 않는다.',
        'P HTTP 구간에도 GPU·Scheduler·큐·Transfer·Control 비용이 함께 들어간다.',
        '요청별6개 구간과 모든 실패 범주는 대응 CSV와 기존 Metrics JSON에 보존했다.','',
        '## 다음 Gate','',
        'Native full-P에서 Descriptor를 E 단계에 미리 발행하는 별도 한 요소 검사를 진행한다.',
        '이 최적화는 Token-prefix에 그대로 적용할 수 없다. 각 이미지의 실제 Consumer Query Coverage 계획이 필요하다.',
        '실제 Generic Token Split, Decode-safe 추가 작업, 독립 N3 및 전체 Task Quality는 아직 미완료다.',
        'Prefill-heavy와 실제 E/D-safe Headroom도 이 검사만으로 확인하지 않는다.',
        'CSV/PNG/PDF는 `mllm_epd_prefill_heavy_token_split_20261124/analysis/native_EC_*v1.*`와 대응 Figure에 있다.', '']
    target=ROOT.parent/'mllm_epd_ec_transport_and_ack_diagnostic_v1.md'
    assert not target.exists(),'Do not overwrite an earlier diagnostic report'
    target.write_text('\n'.join(lines))
    preserve();print('NATIVE_EC_REPORT_V1',len(rows),'cells',len(stage_rows),'exact host decompositions')


if __name__=='__main__':main()
