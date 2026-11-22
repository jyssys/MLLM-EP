"""Locate paired mean-TTFT differences without inventing GPU operator shares."""
from functools import cache
import itertools
import json
import time
import numpy as np
from .common import ROOT,write
from .service_accounting import analyze
from .final_continuous_selection import policy


FRONT=('Arrival_to_P_submit_ms','P_submit_to_accept_ms','P_accept_to_prepare_ms')
STAGES=('Arrival_to_P_prepare_ms','P_prepare_to_ready_ms','Ready_to_D_submit_ms',
    'D_submit_to_prepare_ms','D_prepare_to_first_worker_ms','Worker_to_client_ms')


@cache
def load(name):
    path=ROOT/'analysis'/(name+'_service_accounting.json')
    value=json.loads(path.read_text()) if path.exists() else analyze(name)
    cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
    return value,cfg


def pair(left,right):
    a,ca=load(left);b,cb=load(right)
    for field in ('source_capsule_SHA256','dataset_SHA256','arrival_offsets','seed','gpus','output_cap','ignore_eos'):
        assert ca[field]==cb[field],(left,right,field)
    aa={r['request_id']:r for r in a['rows']};bb={r['request_id']:r for r in b['rows']}
    assert set(aa)==set(bb) and all(aa[r]['prompt_id']==bb[r]['prompt_id'] for r in aa)
    delta={k:a['means'][k]-b['means'][k] for k in a['means']}
    stages=dict(Arrival_to_P_prepare_ms=sum(delta[k] for k in FRONT),
        P_prepare_to_ready_ms=delta['P_prepare_to_ready_ms'],Ready_to_D_submit_ms=delta['Ready_to_D_submit_ms'],
        D_submit_to_prepare_ms=delta['D_submit_to_accept_ms']+delta['D_accept_to_prepare_ms'],
        D_prepare_to_first_worker_ms=delta['D_prepare_to_first_worker_ms'],Worker_to_client_ms=delta['Worker_to_client_ms'])
    assert abs(sum(stages.values())-delta['TTFT_ms'])<1e-3
    return dict(left=left,right=right,seed=ca['seed'],GPU_pair=ca['gpus'],
        stages_ms=stages,mean_TTFT_delta_ms=delta['TTFT_ms'],
        post_first_output_residency_delta_ms=delta['D_decode_residency_ms'],
        actual_output_tokens=dict(left=sum(r['output_tokens'] for r in aa.values()),
            right=sum(r['output_tokens'] for r in bb.values())),
        same_request_output_lengths=all(aa[r]['output_tokens']==bb[r]['output_tokens'] for r in aa),
        contracts={label:dict(credit=c.get('continuous_source_credit_limit',0),axis=c.get('axis_kind','Native'),
            cut=c.get('axis_cut',0),D_policy=list(policy(c)),execution_family=c['execution_family'])
            for label,c in (('left',ca),('right',cb))})


def absolute_interval(values):
    a=np.asarray(values,float);n=len(a)
    if n<3:return dict(mean=float(a.mean()),CI95_ms=None,independent_starts=n)
    index=np.asarray(list(itertools.product(range(n),repeat=n)))
    return dict(mean=float(a.mean()),CI95_ms=np.percentile(a[index].mean(axis=1),[2.5,97.5]).tolist(),
        independent_starts=n,resamples=len(index))


def main():
    assert all((ROOT/'state'/f'final_performance_lane{i}_complete_v1.json').exists() for i in range(4))
    final=json.loads((ROOT/'analysis/final_continuous_strong_confirmation_v1.json').read_text())
    groups=[];deflections=[]
    for cell in final['cells']:
        names={a:[start['arms'][a] for start in cell['starts']] for a in cell['arm_results']}
        pairs=[('BestLayer','BestToken'),('BestHybrid','BestToken'),('BestHybrid','BestLayer')]
        if 'TokenCap2Control' in names:pairs.append(('TokenCap2Control','BestToken'))
        for arm in ('BestLayer','BestToken','BestHybrid','BestWholeDeflection','TokenCap2Control'):
            if arm not in names:continue
            pairs.append((arm,'NativeStrong'))
            cfg=json.loads((ROOT/'configs'/(names[arm][0]['name']+'.json')).read_text())
            common=[a for a in names if a.startswith('NativeCommon') and policy(json.loads(
                (ROOT/'configs'/(names[a][0]['name']+'.json')).read_text()))==policy(cfg)]
            assert len(common)==1,(arm,common)
            pairs.append((arm,common[0]))
        for arm,reference in pairs:
            if arm not in names or reference not in names:continue
            if arm=='BestWholeDeflection':
                from .deflection_service_accounting import paired
                rows=[paired(left['name'],right['name']) for left,right in zip(names[arm],names[reference])
                      if left['status']=='MEASURED_ONLINE' and right['status']=='MEASURED_ONLINE']
                deflections.append(dict(workload=cell['workload'],QPS=cell['QPS'],arm=arm,reference=reference,
                    required_starts=cell['required_starts'],paired_starts=len(rows),rows=rows,
                    status='COMPLETE' if len(rows)==cell['required_starts'] else 'PARTIAL_OR_INVALID'))
                continue
            results=[];invalid=[]
            for left,right in zip(names[arm],names[reference]):
                if left['status']!='MEASURED_ONLINE' or right['status']!='MEASURED_ONLINE':
                    invalid.append(dict(left=left,right=right));continue
                results.append(pair(left['name'],right['name']))
            row=dict(workload=cell['workload'],QPS=cell['QPS'],arm=arm,reference=reference,
                required_starts=cell['required_starts'],paired_starts=len(results),results=results,
                invalid_or_missing=invalid,status='COMPLETE' if len(results)==cell['required_starts'] else 'PARTIAL_OR_INVALID')
            if results:
                row['stage_mean_differences']={k:absolute_interval([r['stages_ms'][k] for r in results]) for k in STAGES}
                row['mean_TTFT_difference']=absolute_interval([r['mean_TTFT_delta_ms'] for r in results])
                row['post_first_output_residency_difference']=absolute_interval([r['post_first_output_residency_delta_ms'] for r in results])
            groups.append(row)
    write(ROOT/'analysis/final_C3_paired_host_service_decomposition_v1.json',dict(time=time.time(),comparisons=groups,
        deflection_route_comparisons=deflections,
        evidence='TRACE_ANALYSIS_OF_MEASURED_ONLINE',
        scope='Exact same-ID/prompt/arrival/source/pair comparisons within each final start block. '
            'Six observed host-path differences add to mean client-TTFT difference. They do not decompose TTFTp95. '
            'Prepare is a host boundary,not a GPU start. C3 prepare-to-ready includes chunk packing and other requests. '
            'Post-first-output residency is separate and is not added to TTFT. Natural actual output lengths can differ. '
            'Pointwise absolute-difference CIs resample paired server starts. Stage CIs do not add to a TTFT CI. '
            'Configured credits,compile families and D policies remain explicit. This does not isolate GPU operator causality.'))
    lines=['# Final C3: 같은 Request의 Host 경계 차이','',
        '단위는 ms다. 값은 왼쪽 Arm에서 Reference를 뺀 평균이다.',
        '각 행은 같은 Request ID, Prompt, Source, Arrival과 Pair를 비교한다.',
        '6개 구간의 평균 차이는 Mean TTFT 차이에 합산된다.',
        '이를 TTFT p95 또는 GPU-exclusive Operator의 분해라고 부르지 않는다.','',
        '| Workload / QPS | Arm − Reference | Paired Starts | Arrival→P Prepare | P Prepare→Ready | Ready→D Submit | D Submit→Prepare | D Prepare→First Worker | Worker→Client | Mean TTFT 차이 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in groups:
        if not row['paired_starts']:continue
        values=[row['stage_mean_differences'][k]['mean'] for k in STAGES]
        lines.append('| '+f"{row['workload']}/q{row['QPS']} | {row['arm']} − {row['reference']} | {row['paired_starts']} | "+
            ' | '.join(f'{v:.3f}' for v in values+[row['mean_TTFT_difference']['mean']])+' |')
    lines += ['','## Whole-prefill Deflection의 다른 경로','',
        'D-local Request에는 P 실행과 P→D Handoff가 없다. 이 단계를 0ms로 만들어 넣지 않는다.',
        '다음 값은 선택된 Route 안의 조건부 Mean TTFT 차이다. Reference도 같은 Request ID만 사용한다.',
        'Route 선택은 무작위가 아니다. 서로 다른 Route의 값을 하나의 P Service 인과 효과로 해석하지 않는다.','',
        '| Workload / QPS | Reference | Seed | Route | Requests | Mean TTFT 차이(ms) |',
        '|---|---|---:|---|---:|---:|']
    for comparison in deflections:
        for result in comparison['rows']:
            for route,group in result['route_groups'].items():
                if group['n']:
                    lines.append(f'| {comparison["workload"]}/q{comparison["QPS"]} | {comparison["reference"]} | '
                        f'{result["seed"]} | {route} | {group["n"]} | {group["paired_mean_TTFT_delta_ms"]:.3f} |')
    lines += ['','각 Stage의 절대 차이 CI, 실제 Output 수, Credit와 Compile Family는 전체 JSON에 있다.',
        'N1에는 독립 Start CI가 없다. N3 CI는 Pointwise Empirical Bootstrap 27개 재표본이다.',
        '첫 Output 이후 Decode Residency는 TTFT와 분리해 기록한다.',
        'Host 구간의 증가를 실제 GPU Attention 비용의 증가로 자동 해석하지 않는다.','']
    (ROOT/'analysis/final_C3_paired_host_service_decomposition_v1.md').write_text('\n'.join(lines),encoding='utf-8')
    print('FINAL_PAIRED_HOST_DECOMPOSITION',len(groups),'comparison cells',flush=True)


if __name__=='__main__':
    main()
