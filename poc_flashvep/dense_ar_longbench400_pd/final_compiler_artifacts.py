"""Reconstruct the final compiler repair without changing raw evidence."""
import argparse
import bisect
import json
import time
import numpy as np
from .common import ROOT,REPO,write,digest,preserve


QPLAN='eager_prefill_native_decode_correctness_frozen_v2.json'
PPLAN='eager_prefill_native_decode_confirmation_frozen_v2.json'


def post_emission_step_indices(steps,groups,first_emissions):
    """Select an offline host-step subset, not exclusive GPU work."""
    return [i for i,queries in groups.items() if queries and all(
        q['logical_phase']=='DECODE' and q['count']==1 and
        q['start']>=q['original_input_tokens'] and q['request_id'] in first_emissions and
        steps[i]['start']>=first_emissions[q['request_id']] for q in queries)]


def teacher_scope():
    plan=json.loads((ROOT/'configs'/QPLAN).read_text())
    rows=[]
    for name in plan['teachers']:
        folder=ROOT/'raw'/name
        cfg=json.loads((folder/'configuration.json').read_text())
        receipt=json.loads((folder/'dispatcher_result.json').read_text())
        paths=json.loads((folder/'independent_teacher_execution_path.json').read_text())
        assert receipt['complete'] and cfg['teacher_eager_all_prompt']
        assert cfg['source_capsule_SHA256']==plan['source_capsule_SHA256']
        assert paths['rows'] and all(r['execution']=='ORIGINAL_FULL36_EAGER_PROMPT' for r in paths['rows'])
        assert all(paths[f] for f in ('original_inner_model_unchanged','original_layers_unchanged',
            'no_partition_adapter','no_NIXL','no_tensor_correction'))
        rows.append(dict(name=name,eager_prompt_calls=len(paths['rows']),
            source_capsule_SHA256=receipt['source_SHA256'],
            raw_hashes={f:digest(folder/f) for f in ('configuration.json','dispatcher_result.json',
                'independent_teacher_execution_path.json','actual_queries_single.json','actual_runtime_single.json')},
            preserved_raw_scope_text=paths['scope']))
    value=dict(time=time.time(),passed=True,teachers=rows,evidence='CORRECTNESS_ONLY_SOURCE_AND_EVENT_AUDIT',
        scope='The frozen V2 teacher config and actual events select eager/CUDA for ALL prompt stages. '
              'Decode uses the original native compiled model. The generic raw scope text inherited from '
              'the prefix-only teacher is inaccurate for V2. Preserve that text and its hash. '
              'This derived audit clarifies the effective path without rewriting raw evidence or rerunning a GPU job. '
              'The eager-call log is not a GPU kernel timeline or a decode performance measurement.')
    write(ROOT/'analysis/eager_teacher_effective_scope_v2.json',value)
    print('FINAL_TEACHER_SCOPE_AUDIT',len(rows),'PASS',flush=True)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .confirmed_figures import mean_ci
    from .figures import save
    data=json.loads((ROOT/'analysis/eager_compile_confirmation_v2_results_v1.json').read_text())
    assert data['complete'],'Wait for every declared terminal outcome'
    arms=[a for a,r in data['arms'].items() if r['starts']==3]
    if not arms:
        print('FINAL_COMPILER_FIGURES_NOT_RUN_BLOCKED_CORRECTNESS',flush=True)
        return
    caption=('MEASURED ONLINE | Adaptive final compiler repair | I3 original16 prompts | Output16,ignore_eos | QPS1.253873.\n'
        'N3 independent paired starts/arm,all0/1 | BI0,B8192/S1024,KV42GiB/Block128 | Common/offload D cap1; NativeStrong unchanged; offload credit8.\n'
        'Pointwise start-level empirical95% CI,27 resamples. No multiplicity correction. Failed old arms remain.\n'
        'Not Natural400 performance,exclusive GPU time,sustainable capacity,or a dynamic-window scheduler.')
    colors={'NativeStrong':'#777777','NativeCommon':'#aaaaaa','Layer':'#0072b2','StockToken':'#009e73',
        'Hybrid':'#d55e00','LayerScoped':'#56b4e9','HybridScoped':'#e69f00'}
    for field,label,stem in (('TTFT_p95_ms','TTFT p95 (ms)','ttft'),
            ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','goodput')):
        fig,ax=plt.subplots(figsize=(10,5.5));points=[]
        for index,arm in enumerate(arms):
            records=data['arms'][arm]['records'];mean,ci=mean_ci([r[field] for r in records])
            ax.bar(index,mean,color=colors[arm],alpha=.85)
            ax.errorbar(index,mean,yerr=[[max(0,mean-ci[0])],[max(0,ci[1]-mean)]],
                fmt='none',ecolor='black',capsize=3)
            points.append(dict(arm=arm,mean=mean,CI95=ci,starts=3))
        ax.set(xticks=range(len(arms)),xticklabels=arms,ylabel=label,
            title='Same-condition D-path ablation | Legacy and new candidates\nNew candidates: eager CUDA prefills / native compiled decode')
        ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
        save(fig,'eager_prefill_native_decode_'+stem,caption,dict(evidence='MEASURED_ONLINE_ADAPTIVE_COMPILER_REPAIR',
            metric=field,points=points,source_capsule_SHA256=data['source_capsule_SHA256']))
    fig,axes=plt.subplots(1,2,figsize=(12,5.8));points=[]
    for ax,ref in zip(axes,('NativeStrong','NativeCommon')):
        candidates=[a for a in ('Layer','Hybrid','LayerScoped','HybridScoped','StockToken')
            if ref in data['comparisons'].get(a,{})]
        for index,arm in enumerate(candidates):
            test=data['comparisons'][arm][ref]
            for offset,field,marker in ((-.12,'ITL_p99_ms','o'),(.12,'TPOT_p95_ms','s')):
                value=test['ratios'][field];mean=value['estimate'];ci=value['CI95']
                ax.errorbar(index+offset,mean,yerr=[[max(0,mean-ci[0])],[max(0,ci[1]-mean)]],
                    fmt=marker,color=colors[arm],capsize=3,
                    label=field if index==0 else None)
                points.append(dict(arm=arm,reference=ref,metric=field,**value,
                    absolute_decode_pass=test['absolute_decode_pass']))
        ax.axhline(1.05,color='red',linestyle='--',label='5% CI-upper limit')
        ax.axhline(1,color='gray',linewidth=.7)
        ax.set(xticks=range(len(candidates)),xticklabels=candidates,ylabel='Candidate / reference ratio',title=ref)
        ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    save(fig,'eager_prefill_native_decode_noninferiority',caption+'\n'
        'PASS requires BOTH metric CI upper bounds<=1.05 against BOTH Native references. Absolute request-decode attainment>=95% in EACH start is separate.',
        dict(evidence='MEASURED_ONLINE_PAIRED_DECODE_GATE',points=points,
            source_capsule_SHA256=data['source_capsule_SHA256']))
    print('FINAL_COMPILER_FIGURES',3,'PNG/PDF pairs',flush=True)


def host_paths():
    from .confirmed_figures import mean_ci
    from .confirmation_analysis import ratio
    plan=json.loads((ROOT/'configs'/PPLAN).read_text())
    data=json.loads((ROOT/'analysis/eager_compile_confirmation_v2_results_v1.json').read_text())
    assert data['complete'];groups={};rows=[]
    fields=('client_ITL_p99_ms','worker_emission_p99_ms','delivery_p99_ms',
        'pure_decode_step_wall_p50_ms','pure_decode_step_wall_p99_ms','pure_step_host_gap_p99_ms')
    for arm,item in data['arms'].items():
        if item['starts']!=3:continue
        values=[]
        for cell in plan['cells']:
            name=cell['jobs'][arm];folder=ROOT/'raw'/name
            requests=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
            clients=[];workers=[];deliveries=[]
            for request in requests:
                events=request['token_events']
                deliveries += [1000*(e['time']-e['worker_time']) for e in events]
                clients += [1000*(b['time']-a['time']) for a,b in zip(events,events[1:])]
                workers += [1000*(b['worker_time']-a['worker_time']) for a,b in zip(events,events[1:])]
            steps=json.loads((folder/'engine_steps_decode.json').read_text())['rows']
            queries=json.loads((folder/'actual_queries_decode.json').read_text())['rows']
            starts=[r['start'] for r in steps];matched={}
            for query in queries:
                index=bisect.bisect_right(starts,query['time'])-1
                if index>=0 and query['time']<=steps[index]['end']:
                    matched.setdefault(index,[]).append(query)
            pure=[i for i,q in matched.items() if all(r['logical_phase']=='DECODE' for r in q)]
            walls=[steps[i]['wall_ms'] for i in pure]
            gaps=[1000*(steps[i]['start']-steps[i-1]['end']) for i in pure if i-1 in pure]
            emissions={'q'+str(r['sample_id']):r['token_events'][0]['worker_time'] for r in requests}
            post=post_emission_step_indices(steps,matched,emissions)
            post_walls=[steps[i]['wall_ms'] for i in post]
            post_gaps=[1000*(steps[i]['start']-steps[i-1]['end']) for i in post if i-1 in post]
            assert clients and workers and deliveries and walls and gaps,(name,'Missing diagnostic boundary')
            assert post_walls and post_gaps,(name,'Missing post-emission diagnostic subset')
            percentile=lambda a,q:float(np.percentile(a,q))
            values.append(dict(name=name,client_ITL_p99_ms=percentile(clients,99),
                worker_emission_p99_ms=percentile(workers,99),delivery_p99_ms=percentile(deliveries,99),
                pure_decode_step_wall_p50_ms=percentile(walls,50),pure_decode_step_wall_p99_ms=percentile(walls,99),
                pure_step_host_gap_p99_ms=percentile(gaps,99),pure_decode_host_steps=len(pure),
                post_emission_step_wall_p50_ms=percentile(post_walls,50),
                post_emission_step_wall_p99_ms=percentile(post_walls,99),
                post_emission_host_gap_p99_ms=percentile(post_gaps,99),
                post_emission_host_steps=len(post),decode_label_steps_before_emission=len(pure)-len(post)))
        means={f:mean_ci([r[f] for r in values]) for f in fields}
        means.update({f:mean_ci([r[f] for r in values]) for f in (
            'post_emission_step_wall_p50_ms','post_emission_step_wall_p99_ms','post_emission_host_gap_p99_ms')})
        groups[arm]=dict(starts=3,rows=values,means=means);rows.extend(values)
    comparisons={}
    for candidate,old in (('LayerScoped','Layer'),('HybridScoped','Hybrid')):
        if candidate in groups and old in groups:
            comparisons[candidate+'/'+old]={f:ratio([r[f] for r in groups[candidate]['rows']],
                [r[f] for r in groups[old]['rows']]) for f in fields}
    write(ROOT/'analysis/eager_compile_host_boundaries_v2.json',dict(time=time.time(),groups=groups,
        comparisons=comparisons,evidence='TRACE_ANALYSIS_OF_MEASURED_ONLINE',
        scope='Same-condition new source and paired seeds. Each value derives from actual client/worker/engine/query events. '
              'Legacy pure_decode field names mean DECODE-labeled query rows. They may begin BEFORE first worker output. '
              'The separate post_emission subset requires one query/request,position>=prompt length,and host start>=first worker output. '
              'Keep both subsets. Neither is exclusive GPU operation or an idle proof. '
              'The source intervention also changes numerical dispatch,subsequent batching and admission feedback. '
              'Do not subtract these percentiles as a causal cost decomposition. Every original interval remains in primary metrics.'))
    target=REPO/'poc_flashvep/reports/dense_ar_longbench400_eager_prefill_native_decode_results.md'
    marker='## 같은 실행의 Host 경계 진단'
    prefix=target.read_text().split(marker)[0].rstrip()
    lines=['',marker,'','아래 값은 Start별 통계량의 평균이다. GPU-exclusive 시간은 아니다.',
        '첫 표는 기록한 Query가 모두 DECODE인 Host Step이다. 첫 Worker 출력 전의 Step도 포함할 수 있다.',
        '비동기 Engine의 첫 출력 대기와 후속 Decode를 같은 구간으로 해석하지 않는다.',
        '이 Step의 Wall Time도 Kernel 시간이나 Physical Idle과 같지 않다.','',
        '| Arm | Client ITL p99(ms) | Worker Emission p99(ms) | Delivery p99(ms) | Decode-labeled Host Step p50(ms) | Decode-labeled Host Step p99(ms) | 연속 Labeled Step 사이 Host Gap p99(ms) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,item in groups.items():
        lines.append('| '+arm+' | '+' | '.join(f'{item["means"][f][0]:.6f}' for f in fields)+' |')
    lines += ['', '### 첫 Worker 출력 이후의 별도 Host Step 부분집합','',
        '각 Request의 첫 Worker 출력 이후 시작한 Step만 아래 부분집합에 넣었다.',
        'Query는 Request당1개이며 Position은 원래 Prompt 길이 이상이어야 한다.',
        '이것은 후속 Streaming Host 경계다. 모든15개 GPU Decode Step의 완전한 비용 합은 아니다.',
        '기존 첫 표와 모든 Client Interval은 그대로 보존한다. Primary Safety와 Throughput은 바뀌지 않는다.','',
        '| Arm | 선택한 Host Steps/3 Starts | Labeled 중 부분집합 밖 Steps/3 Starts | 이후 Step p50(ms) | 이후 Step p99(ms) | 연속 이후 Step 사이 Gap p99(ms) |',
        '|---|---:|---:|---:|---:|---:|']
    post_fields=('post_emission_step_wall_p50_ms','post_emission_step_wall_p99_ms','post_emission_host_gap_p99_ms')
    for arm,item in groups.items():
        n=sum(r['post_emission_host_steps'] for r in item['rows'])
        outside=sum(r['decode_label_steps_before_emission'] for r in item['rows'])
        lines.append(f'| {arm} | {n} | {outside} | '+' | '.join(f'{item["means"][f][0]:.6f}' for f in post_fields)+' |')
    lines += ['', '각 Percentile의 차이를 비용의 합으로 해석하지 않는다.',
        '비교에는 Compiler·Kernel 정책과 이후 Admission·Batch Feedback이 포함된다.',
        '한 Operator의 GPU 비용 감소가 원인이라고 단정하지 않는다.',
        '실제 GPU-exclusive Operator Breakdown과 SM/HBM은 이 Source에서 NOT_MEASURED다.','',
        '- [Start별 값과 Paired CI](dense_ar_longbench400_2d_20261121/analysis/eager_compile_host_boundaries_v2.json)','']
    target.write_text(prefix+'\n'+'\n'.join(lines),encoding='utf-8')
    print('FINAL_COMPILER_HOST_DIAGNOSTIC',len(rows),'paired-start records',flush=True)


def residual_appendix():
    path=REPO/'poc_flashvep/reports/dense_ar_longbench400_decode_compile_residual.md'
    prefix=path.read_text().split('## 후속 IR Provider 개입과 최종 Scope')[0].rstrip()
    lines=['','## 후속 IR Provider 개입과 최종 Scope','',
        '위 표의 V1–V3 실패는 그대로 유지한다. 아래 결과는 이후의 별도 개입이다.',
        '위 Host 표의 Pure Step은 Query의 DECODE Label 기준이다. 첫 출력 전 대기를 포함할 수 있다.',
        '별도 최종 보고서는 첫 Worker 출력 이후의1-query Host Step을 추가 분리한다.',
        '어느 표도 GPU-exclusive Decode Kernel 시간으로 사용하지 않는다.',
        '설치 Source의 `vllm/platforms/cuda.py`는 Compile 여부에 따라 IR Norm Provider 우선순위를 바꾼다.',
        'Mode0는 `vllm_c,native`, 기본 Inductor Compile은 `native`를 사용했다.',
        '`custom_ops=all`은 이 IR 우선순위를 복원하지 않았다.',
        'D의 RMSNorm과 Fused-add RMSNorm 우선순위만 함께 원래 값으로 바꿨다.',
        '이 Global IR 개입에서는 Layer와 Hybrid의 Strict N1/N8 KV·Hidden·Greedy가 모두 일치했다.',
        '이후 실제 Mixed32, Guarded Natural32와 Natural400 State·Quality Gate도 모두 통과했다.',
        '이 결과는 적어도 검사한 엄격한 불일치에서 Norm Provider Bundle의 인과적 근거다.',
        '두 Norm Operator의 개별 기여율이나 모든 BI0 수치 차이의 원인을 확정한 것은 아니다.',
        'Global IR 경로의 적격 Serving 속도 향상은 측정하지 않았다.','',
        'Partial-prefill만 CUDA Norm으로 제한한 다음 Source에서는 Layer가 전체 Gate를 통과했다.',
        'Hybrid는 Strict N1의 KV·Hidden Gate를 실패했다. Greedy16은 일치했다.',
        '이 Source의 예정 성능21회는 첫 실행 전에 대체했다. 모든21회를 NOT_RUN으로 보존한다.',
        '실패한 State를 사후 Tolerance 또는 Tensor Correction으로 통과시키지 않았다.','',
        '마지막 제한적 수정은 모든 Prompt Stage를 원래 Eager/CUDA Norm으로 유지한다.',
        'Decode-only Full36은 원래 Native Compiled Model을 사용한다.',
        'Cold Dispatch의 CPU Assertion 실패는 GPU 시작 전에 고쳤다. V1의 예정38회는 모두 미실행이다.',
        'V2는 독립 Full36 Teacher와 같은 Numerical Stage 계약을 사용한다.',
        'P의 계산과 Token·Timestamp 및 Frozen SLO는 바꾸지 않았다.',
        '이 최종 Source의 정합성과 성능 판정은 별도 보고서에 기록한다.','',
        '- [Global IR 정합성 설계](dense_ar_longbench400_2d_20261121/configs/decode_IR_priority_correctness_frozen_v1.json)',
        '- [Partial-only Source의 미실행 성능21회](dense_ar_longbench400_scoped_decode_compile_results.md)',
        '- [최종 Eager Prefill / Native Decode 결과](dense_ar_longbench400_eager_prefill_native_decode_results.md)',
        '- [Teacher의 실제 Scope와 Raw Hash](dense_ar_longbench400_2d_20261121/analysis/eager_teacher_effective_scope_v2.json)','']
    path.write_text(prefix+'\n'+'\n'.join(lines),encoding='utf-8')


def decisions():
    from .decision_log import record
    path=ROOT/'analysis/research_decisions.json'
    existing={r['id'] for r in json.loads(path.read_text())['rows']}
    entries=[
        ('R_D_IR_NORMALIZATION_PROVIDER_INTERVENTION',
         'D-only compile changes strict KV/hidden. Disabled controls pass. Actual Mode0 versus Mode3 logs select different IR Norm providers.',
         ['The IR Norm provider bundle causes the observed strict divergence.',
          'Custom-op all alone restores the state.', 'KV ownership differs independently of the numerical path.'],
         'Keep the D compiler and every other option. Set both D IR Norm priorities to the Mode0 vllm_c,native order.',
         'Layer and Hybrid both pass strict N1/N8,real mixed32,guarded32 and Natural400. Custom-op all alone failed. '
         'This supports the provider bundle for the tested divergence. Individual norm contributions are not separated.',
         'Do not infer serving speedup. Test a stage-scoped path with an independent full36 teacher.',
         'CORRECTNESS_ONLY_CONTROLLED_KERNEL_PROVIDER_INTERVENTION'),
        ('R_PARTIAL_ONLY_IR_HYBRID_STRICT_FAILURE',
         'Scoping CUDA normalization only to partial prefills passes Layer proofs but fails Hybrid strict N1 KV/hidden. Greedy16 agrees.',
         ['Hybrid full36 suffix still uses a different numerical path.',
          'The compiled suffix or adapter boundary has another state-sensitive difference.'],
         'Use an original full36 legal-boundary teacher with the declared eager-prefix/compiled-suffix path. Preserve the strict mismatch.',
         'Layer passes all proof stages. Hybrid fails strict N1. All21 old serving starts stay NOT_RUN before any timing.',
         'One final bounded repair keeps ALL prompt work eager/CUDA and preserves only native compiled decode. No new SLO or tolerance.',
         'CORRECTNESS_ONLY_AND_EXPLICIT_NOT_RUN'),
        ('R_FINAL_EAGER_COLD_DISPATCH_CPU_PREFLIGHT',
         'The actual nested dispatch from the V1 capsule reads dispatch_schedule before cold initialization creates it.',
         ['A cold-start attribute guard is missing.'],
         'Execute the immutable nested dispatch AST on a cold CPU stub before any GPU launch.',
         'The V1 harness raises AttributeError. A getattr-based pending-prompt helper passes. All38 V1 starts remain NOT_RUN.',
         'Freeze corrected V2,then require N1,N8,mixed32,Natural32 and Natural400 before the unchanged21 paired serving starts.',
         'CPU_MINIMAL_RED_GREEN_NO_GPU_PERFORMANCE'),
        ('R_REGISTRATION_CONTROLLED_BRIDGE54_COMPLETE',
         'All54 original fixed-condition bridge starts have their common-registration repaired counterparts.',
         ['Registration/admission feedback explains every residual tail.',
          'Compiler,adapter or client-delivery paths leave condition-specific tails.'],
         'Keep all three original output/cap conditions,strong Token,IDs,arrival and pair4/5. Apply only the common registration source change.',
         'O16/Cap1 Layer/Hybrid fail the fast NativeStrong gate. O64/Cap1 both fail an ITL gate. '
         'O64/Cap2 Layer passes both Native references but L/T TTFT CI includes1. Hybrid still fails. Strong Token passes all three.',
         'Keep every failure. Do not transfer Natural400 safety to the fixed-output bridge or weaken the strong Token baseline.',
         'MEASURED_ONLINE_ADAPTIVE_COMMON_RUNTIME_REPAIR'),
        ('R_FINAL_FIRST_EMISSION_HOST_SCOPE_RED_GREEN',
         'A real StockToken step has DECODE query rows but starts before the first worker output. Its host wall is321.881566ms.',
         ['An asynchronous pre-first-output wait belongs to that host enclosure.',
          'Every subsequent decode step is actually similarly slow.',
          'Query labels alone do not identify the client streaming phase.'],
         'The actual raw-step assertion start>=first worker output fails. Add an offline one-query,post-prompt,post-first-output subset and preserve the old labeled-step table.',
         'The new CPU regression passes. All seven arms have672 selected host steps/three starts and48 labeled steps outside that subset. '
         'Post-emission host p99 is8.247261ms NativeStrong,8.336188ms StockToken and8.523289ms new Hybrid. '
         'These remain host walls,not exclusive GPU times. Primary client intervals,CI and safety verdicts are unchanged.',
         'Bind both diagnostic denominators. Do not classify a query-labeled pre-first step as an exclusive subsequent decode kernel.',
         'TRACE_ANALYSIS_AND_CPU_ACTUAL_RAW_REGRESSION')]
    for args in entries:
        if args[0] not in existing:record(*args)
    target=ROOT/'analysis/eager_compile_confirmation_v2_results_v1.json'
    identifier='R_FINAL_EAGER_PREFILL_NATIVE_DECODE_N3_COMPLETE'
    if target.exists() and json.loads(target.read_text())['complete'] and identifier not in existing:
        record(identifier,
            'All21 frozen fixed-condition starts completed. Both new axes passed strict,mixed32 and Natural400 first.',
            ['D full-model compiler/numerical dispatch explains the residual tail.',
             'Adapter,host or scheduling feedback leaves an additional condition-specific tail.'],
            'Same new source,pair0/1,three paired seeds,I3/Output16/Cap1. Retain legacy Layer/Hybrid,strong compiled Token and BOTH Native references.',
            'New Hybrid passes both Native5% and every-start absolute decode gates. H/legacyH ITL ratio0.979221,TPOT0.980672. '
            'New Layer passes NativeStrong but fails Common Native ITL CIupper1.065921. '
            'New H/T TTFT1.121429 CI[1.050122,1.179215],goodput0.965898 CI[0.949395,0.988077]. '
            'Thus partial tail repair does not establish a strong-Token serving win.',
            'End this bounded GPU campaign. Preserve failures,unmeasured C3 operator shares and unverified safe windows. Finish review,resource restoration,archive and publication.',
            'MEASURED_ONLINE_ADAPTIVE_COMPILER_PATH_ABLATION')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figures',action='store_true');args=parser.parse_args()
    preserve();teacher_scope();residual_appendix();decisions()
    if args.figures:figures();host_paths()
    preserve()


if __name__=='__main__':main()
