"""Keep isolated head-offload cost,transport correctness and serving claims separate."""
import json
import os
from pathlib import Path
import time
import numpy as np
from .common import ROOT,START,write,digest,bind
from .critical_analysis import csv_save
from .attention_poc import POC


def analyze():
    bind()
    from ..dense_ar_output_aware_v2.statistics import ratio
    def process_ratio(left,right):
        result=ratio(left,right)
        result['method']='Paired independent benchmark-process empirical bootstrap; pointwise,unadjusted'
        return result
    active=POC/'state/active_protocol.json'
    protocol=json.loads(active.read_text()) if active.exists() else dict(config='attention_poc_v1.json')
    config=POC/'configs'/protocol['config'];cfg=json.loads(config.read_text());starts=[];states=[];checks=[]
    manifest=POC/'raw'/cfg['capture_name']/'capture_manifest.json'
    captures=json.loads(manifest.read_text())['captures'] if manifest.exists() else []
    shapes={(row['prompt_id'],row['layer'],row['query_start']):row for row in captures}
    for index in range(3):
        folder=POC/'raw'/f'{cfg.get("benchmark_prefix","Benchmark")}_s{index}';receipt=folder/'dispatcher_result.json'
        if not receipt.exists():states.append(dict(start=index,status='NOT_RUN_OR_RUNNING'));continue
        if not json.loads(receipt.read_text())['complete']:states.append(dict(start=index,status='FAILED'));continue
        value=json.loads((folder/'result.json').read_text());states.append(dict(start=index,status='COMPLETE',
            source_SHA256=digest(folder/'result.json'),GPU_UUIDs=value['GPU_UUIDs']))
        for row in value['rows']:starts.append(row)
        for row in value['checks']:
            shape=shapes.get((row['prompt_id'],row['layer'],row['query_start']))
            assert shape,'Every correctness observation must map to its real captured query'
            checks.append(dict(start=index,input_tokens=shape['input_tokens'],query_tokens=shape['query_tokens'],
                key_tokens=shape['key_tokens'],configured_num_splits=shape['num_splits'],**row))
    cells=[]
    keys=sorted({(row['prompt_id'],row['input_tokens'],row['selected_layers'],row['remote_KV_heads']) for row in starts})
    for prompt,length,layers,heads in keys:
        by_start={index:{row['mode']:row for row in starts if row['start']==index and row['prompt_id']==prompt
            and row['selected_layers']==layers and row['remote_KV_heads']==heads} for index in range(3)}
        remote_modes=cfg.get('remote_modes',['RemoteSplit'])
        matching=[(index,values) for index,values in by_start.items() if all(mode in values for mode in ['LocalFull','LocalSplit',*remote_modes])]
        if not matching:continue
        observed=[row for row in checks if row['prompt_id']==prompt and row['remote_KV_heads']==heads and row['layer']<layers
            and row['start'] in {index for index,values in matching}]
        assert observed,'Never treat an absent correctness comparison as an exact pass'
        for mode in remote_modes:
            transfer_check='cached_vs_local_split' if mode=='RemoteCachedSplit' else 'remote_vs_local_split'
            exact={field:all(row[field]['exact'] for row in observed) for field in ('full_vs_captured','split_vs_full',transfer_check)}
            rr=process_ratio([rows[mode]['p50_ms'] for index,rows in matching],[rows['LocalFull']['p50_ms'] for index,rows in matching])
            cells.append(dict(prompt_id=prompt,input_tokens=length,selected_layers=layers,remote_KV_heads=heads,remote_query_heads=heads*4,
                remote_mode=mode,independent_benchmark_processes=len(matching),strict=exact,
                remote_over_local_full_p50=rr,
                local_split_over_local_full_p50=process_ratio([rows['LocalSplit']['p50_ms'] for index,rows in matching],[rows['LocalFull']['p50_ms'] for index,rows in matching]),
                remote_over_local_split_p50=process_ratio([rows[mode]['p50_ms'] for index,rows in matching],[rows['LocalSplit']['p50_ms'] for index,rows in matching]),
                local_full_p50_ms=float(np.mean([rows['LocalFull']['p50_ms'] for index,rows in matching])),
                remote_p50_ms=float(np.mean([rows[mode]['p50_ms'] for index,rows in matching])),
                transfer_payload_bytes=matching[0][1][mode]['transfer_payload_bytes'],
                transfer_payload_scope='Logical requested tensor bytes. Actual DMA/wire bytes are not measured.',
                evidence=('ISOLATED_ATTENTION_MEASUREMENT' if all(exact.values()) else
                    'DIAGNOSTIC_REPLAY_REFERENCE_MISMATCH' if not exact['full_vs_captured'] else
                    'DIAGNOSTIC_REMOTE_SPLIT_MISMATCH' if not exact[transfer_check] else
                    'DIAGNOSTIC_HEAD_SHAPE_NUMERICAL_DIFFERENCE'),
                point_win=rr['estimate']<1,confirmed_isolated_cost_win=len(matching)==3 and all(exact.values()) and rr['CI95'][1]<1,
                full_model_accuracy='NOT_RUN',decode_safety='NOT_RUN',online_serving='NOT_RUN'))
    protocols=[]
    for path in sorted((POC/'configs').glob('attention_poc_v*.json')):
        version=int(path.stem.rsplit('v',1)[1]);completion=POC/'state'/f'completion_v{version}.json'
        state=json.loads(completion.read_text()) if completion.exists() else {}
        if version==1 and (POC/'state/prelaunch_protocol_v1_revision.json').exists():
            state=json.loads((POC/'state/prelaunch_protocol_v1_revision.json').read_text())
        protocols.append(dict(version=version,config_SHA256=digest(path),
            source_archive_SHA256=json.loads(path.read_text())['source_archive_SHA256'],status=state.get('status','NOT_RUN_OR_RUNNING'),
            error=state.get('error'),reason=state.get('reason')))
    correctness_summary={field:dict(observations=len([row for row in checks if field in row]),
        nonexact_observations=sum(not row[field]['exact'] for row in checks if field in row),
        maximum_absolute_difference=max([row[field]['max_abs'] for row in checks if field in row],default=None))
        for field in ('full_vs_captured','split_vs_full','remote_vs_local_split','cached_vs_local_split')}
    query_groups=[]
    for query in sorted({row['query_tokens'] for row in checks}):
        selected=[row for row in checks if row['query_tokens']==query]
        query_groups.append(dict(query_tokens=query,observations=len(selected),
            split_full_nonexact=sum(not row['split_vs_full']['exact'] for row in selected),
            remote_split_nonexact=sum(not row['remote_vs_local_split']['exact'] for row in selected),
            cached_split_nonexact=sum(not row['cached_vs_local_split']['exact'] for row in selected),
            scope='Observed query-shape association. The actual scheduler split/reduction choices are not instrumented.'))
    value=dict(time=time.time(),active_config=config.name,config_SHA256=digest(config),protocols=protocols,
        correctness_summary=correctness_summary,correctness_by_query_shape=query_groups,
        states=states,starts=starts,cells=cells,checks=checks,
        scope='Wall timing for serial sequences of captured attention calls only. All heads remain computed. '
            'Remote-output exactness compares remote split with the same local split shape. Full-shape comparison is separate. '
            'A local-split win is not a win over the stronger full-head kernel. Use LocalFull as primary isolated baseline. '
            'CI uses independent benchmark processes,not iteration samples or independent server starts. '
            'P/D engine queues,full36 model outputs,task quality and decode protection are not tested by this PoC. '
            'No claim of exclusive GPU compute,DMA duration,NIXL/NCCL cost or online serving speedup.')
    write(POC/'analysis/attention_head_poc_analysis.json',value)
    csv_save(POC/'figures/attention_head_poc_per_start.csv',starts)
    csv_save(POC/'figures/attention_head_poc_cells.csv',cells)
    csv_save(POC/'figures/attention_head_poc_correctness.csv',checks)
    csv_save(POC/'figures/attention_head_poc_query_shape_correctness.csv',query_groups)
    if cells:figures(value)
    report(value);print('ATTENTION_POC_ANALYSIS',len(cells),'cells',flush=True);return value


def figures(value):
    os.environ.setdefault('SOURCE_DATE_EPOCH',str(START))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=value['cells'];modes=list(dict.fromkeys(row['remote_mode'] for row in rows))
    fig,axes=plt.subplots(1,len(modes),figsize=(7*len(modes),5),squeeze=False)
    for ax,mode in zip(axes[0],modes):
        for layers in (1,2,4):
            for heads in (2,4):
                selected=sorted([row for row in rows if row['remote_mode']==mode and row['selected_layers']==layers
                    and row['remote_KV_heads']==heads],key=lambda row:row['input_tokens'])
                if not selected:continue
                x=[row['input_tokens'] for row in selected];y=[row['remote_over_local_full_p50']['estimate'] for row in selected]
                errors=np.array([[value-row['remote_over_local_full_p50']['CI95'][0],row['remote_over_local_full_p50']['CI95'][1]-value]
                    if row['remote_over_local_full_p50']['CI95'] is not None else [0,0] for value,row in zip(y,selected)]).T
                ax.errorbar(x,y,yerr=errors,marker='o',capsize=2,label=f'{layers} layers,{heads}/8 KV groups')
                failed=[row for row in selected if not all(row['strict'].values())]
                if failed:ax.scatter([row['input_tokens'] for row in failed],
                    [row['remote_over_local_full_p50']['estimate'] for row in failed],color='crimson',marker='x',s=75,zorder=8)
        ax.axhline(1,color='black',ls='--');ax.set_title(mode);ax.set_ylabel('Remote / local-full wall p50 start-mean ratio')
        ax.set_xlabel('Actual input tokens');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Isolated real-input attention-head PoC | D idle | not full-layer or online serving')
    fig.text(.01,.015,'Raw measured points only. CSV contains independent-process N,95% CI and strict gates. '
        'Red×=strict tensor mismatch. Packing,peer-copy,launch,waiting and joining enter wall time. No active-decode safety claim.',fontsize=8)
    fig.tight_layout(rect=(0,.09,1,.94))
    for extension in ('png','pdf'):fig.savefig(POC/'figures'/('attention_head_poc_cost.'+extension),dpi=180)
    csv_save(POC/'figures/attention_head_poc_cost.csv',rows);plt.close(fig)


def report(value):
    def link(path):return '['+path.name+']('+os.path.relpath(path,ROOT.parent)+')'
    lines=['# Attention-head Offloading 소규모 PoC','',
        '상태: 실측 완료 여부와 정합성 Gate는 아래 표를 따릅니다. 기존 P-critical-path 캠페인과 별도입니다.','',
        'Qwen3-8B, Original BF16, BI0를 사용합니다. 원문 Prompt와 LongBench400-v2는 변경하지 않습니다.',
        '입력 길이만으로8K·16K·32K에 가장 가까운 원본 Request를 사전 선택합니다.',
        '길이별 원본 Prompt는1개입니다. 길이 효과를 Task·Prompt 차이와 분리하지 않았습니다.',
        'Original Full36 vLLM의 Eager Capture에서 실제 Q/K/V와 Paged KV, RoPE 이후 Query를 저장합니다.',
        'Capture는 B8192/S1024와 KV42GiB를 사용합니다. CPU Tensor 복사는 침습적이므로 Capture 시간은 성능 결과가 아닙니다.','',
        '재생 실험은 LocalFull, LocalSplit과 원격 실행 경로를 비교합니다.',
        'RemoteSplit은 Chunk마다 전체 Prefix KV를 전송합니다.',
        'RemoteCachedSplit은 Layer별 원격 KV를 보존합니다. 새 Page와 갱신한 경계 Page만 전송합니다.',
        '두 원격 경로는 같은 LocalFull 기준과 비교합니다. KV 재사용이 없는 경로만으로 전체 방법을 판정하지 않습니다.',
        'KV Head2/8과4/8을 옮깁니다. Query Head는 GQA 비율4:1을 유지합니다.',
        '모든 Head를 계산합니다. Head를 버리는 근사 기법이 아닙니다.',
        '1·2·4 Layer는 선택한 Layer들의 Attention 호출 열을 뜻합니다. 전체 Transformer Layer 실행시간이 아닙니다.',
        'QKV Projection, FFN, 다음 Layer Hidden의 준비 비용은 포함하지 않습니다.',
        '왕복 시간에는 Packing, Torch Peer Copy, 두 Device의 Launch·Wait와 결과 병합이 포함됩니다.',
        'Tensor·Event·Scheduler Metadata의 초기 준비는 Timing 전에 수행합니다. 동적 Metadata 생성 비용은 이 수치에 포함하지 않습니다.',
        '실제 NIXL/NCCL 경로와 별도입니다. 이 시간으로 Production 통신 비용을 확정하지 않습니다.',
        '재생 중 전체 Model Weight와42GiB Serving KV Cache는 상주하지 않습니다.',
        'D는 이 실험 외에는 계산하지 않습니다. Native Decode 보호와 실제 Safe Window는 NOT_RUN입니다.','',
        '| Input Tokens | Layers | 원격 경로 | 원격 KV Heads | 독립 Process | LocalFull ms | Remote ms | 논리 Payload MiB | Remote/LocalFull p50 [95% CI] | Strict Gates |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---|---|']
    for row in value['cells']:
        r=row['remote_over_local_full_p50'];lines.append(f'| {row["input_tokens"]} | {row["selected_layers"]} | {row["remote_mode"]} | {row["remote_KV_heads"]}/8 | '
            f'{row["independent_benchmark_processes"]} | {row["local_full_p50_ms"]:.4f} | {row["remote_p50_ms"]:.4f} | '
            f'{row["transfer_payload_bytes"]/2**20:.3f} | {r["estimate"]:.6f} {r["CI95"]} | {row["strict"]} |')
    lines+=['','CI는 독립 Benchmark Process를 사용합니다. 독립 Serving Start CI가 아닙니다.',
        '동일 Capture를3개 Process가 재생합니다. 새 원본 Prompt3×3개를 생성한 실험은 아닙니다.',
        'ms 열은 Process별 p50의 평균입니다. 모든 Iteration을 합친 p50이 아닙니다.',
        '논리 Payload는 요청한 Tensor Bytes입니다. 실제 DMA 또는 Wire Bytes는 측정하지 않았습니다.',
        'Remote와 LocalSplit의 Exact 비교는 같은 Head 형상의 최종 Attention 결과를 검사합니다.',
        '이 비교만으로 Copy와 원격 Kernel의 개별 정확도를 각각 분해하지는 않습니다.',
        'LocalSplit과 LocalFull의 차이는 Head Geometry 변경을 검사합니다.',
        '전체 모델 Greedy Output·Task 품질·Online TTFT·SLO-goodput은 NOT_RUN입니다.',
        'Isolated 비용 이득만으로 Layer-specific Serving Gain이나 Dynamic Scheduler 성공을 주장하지 않습니다.','',
        '## 정합성 결과의 구분','',
        '| 비교 | 관측 수 | Exact 불일치 수 | 최대 절대 오차 |','|---|---:|---:|---:|']
    for field,row in value['correctness_summary'].items():
        lines.append(f'| {field} | {row["observations"]} | {row["nonexact_observations"]} | {row["maximum_absolute_difference"]} |')
    lines+=['','원격 결과는 같은 Head 형상의 LocalSplit과 비교합니다.',
        'Head를 분할한 결과와 전체 Head Kernel의 결과는 별도로 비교합니다.',
        '불일치한 Cell은 Strict 통과 비용 이득으로 확정하지 않습니다.',
        '이 차이를 전체 모델 품질 저하 또는 품질 통과로 바꾸어 해석하지 않습니다.','',
        '| 실제 Query Tokens | 관측 수 | Split/Full 불일치 | Remote/LocalSplit 불일치 | Cached/LocalSplit 불일치 |',
        '|---:|---:|---:|---:|---:|']
    for row in value['correctness_by_query_shape']:
        lines.append(f'| {row["query_tokens"]} | {row["observations"]} | {row["split_full_nonexact"]} | '
            f'{row["remote_split_nonexact"]} | {row["cached_split_nonexact"]} |')
    lines+=['','이 표는 Query Shape별 관측입니다. 실제 Split-K 선택이나 Reduction Tree는 계측하지 않았습니다.',
        '216개 관측은36개 Captured Attention 호출×2개 Head 분할×3개 재생 Process입니다.',
        '이를216개 독립 Request로 취급하지 않습니다.',
        'Kernel의 내부 선택을 불일치의 확정 원인으로 표현하지 않습니다.','',
        '## Protocol 이력','',
        '| Version | 상태 | Source Archive SHA256 |','|---:|---|---|']
    for row in value['protocols']:
        lines.append(f'| {row["version"]} | {row["status"]} | `{row["source_archive_SHA256"]}` |')
    lines+=['','활성 Protocol: `'+value['active_config']+'`. 이전 Raw와 실패는 보존합니다.',
        '이전 Protocol의 Process를 새 Protocol과 합쳐 독립 N을 늘리지 않습니다.','',
        link(POC/'analysis/attention_head_poc_analysis.json'),'',
        '관련 선행연구: [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509), '
        '[AFD 설계 공간 분석](https://arxiv.org/abs/2605.28302).',
        '학습용 Sequence Parallelism과 대규모 MoE의 AFD 결과를 이 Dense8B PoC의 성능 근거로 사용하지 않습니다.','']
    (ROOT.parent/'dense_ar_attention_head_offloading_poc.md').write_text('\n'.join(lines))


if __name__=='__main__':analyze()
