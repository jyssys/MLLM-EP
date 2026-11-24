"""Explain observed N3 host paths and metric failures without causal GPU claims."""
import csv
import itertools
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .token_confirmation_analysis import mean_interval
from .token_discovery_analysis import STAGES,table

GATES=('TTFT_PASS','ITL_PASS','TPOT_PASS')
METHODS=('NativeStrong','NativeCommon','Static10','Idle10')


def failure_accounting(rows,duration):
    """Exact metric-level attribution. It is not a runtime intervention."""
    assert rows and duration>0
    total=sum(r['output_tokens'] for r in rows)
    joint=[r for r in rows if all(r[k] for k in GATES)]
    counterfactual=[]
    for count in range(4):
        for removed in itertools.combinations(GATES,count):
            passed=[r for r in rows if all(r[k] for k in GATES if k not in removed)]
            tokens=sum(r['output_tokens'] for r in passed)
            counterfactual.append(dict(removed_gates='+'.join(removed) or 'NONE',
                request_count=len(passed),request_denominator=len(rows),
                qualified_tokens=tokens,total_tokens=total,
                SLO_goodput=tokens/duration,scope='OFFLINE_METRIC_COUNTERFACTUAL. No service or queue change.'))
    attributed={k:dict(lost_requests=0.,lost_output_tokens=0.) for k in GATES}
    for row in rows:
        failed=[k for k in GATES if not row[k]]
        for key in failed:
            attributed[key]['lost_requests']+=1/len(failed)
            attributed[key]['lost_output_tokens']+=row['output_tokens']/len(failed)
    assert abs(sum(v['lost_requests'] for v in attributed.values())-(len(rows)-len(joint)))<1e-8
    assert abs(sum(v['lost_output_tokens'] for v in attributed.values())-
        (total-sum(r['output_tokens'] for r in joint)))<1e-8
    return counterfactual,[dict(gate=k,**v,
        scope='Shapley-style metric allocation: divide each failed request across its failed gates. Not runtime causality.')
        for k,v in attributed.items()]


def zero_gate_oracle(rows,duration):
    """Remove only observed coordinator D wait, with no feedback or safety model."""
    original=[];ideal=[];qualified=0;original_qualified=0
    for row in rows:
        wait=row['D_coordinator_gate_wait'];ttft=row['TTFT_ms']
        assert -1e-5<=wait<=ttft+1e-5
        value=max(0.,ttft-wait);original.append(ttft);ideal.append(value)
        decode=row['ITL_PASS'] and row['TPOT_PASS']
        if decode and value<=5000:qualified+=row['output_tokens']
        if decode and ttft<=5000:original_qualified+=row['output_tokens']
    return dict(observed_TTFT_p95_ms=float(np.percentile(original,95)),
        zero_D_coordinator_wait_TTFT_p95_ms=float(np.percentile(ideal,95)),
        observed_SLO_goodput=original_qualified/duration,
        zero_D_coordinator_wait_SLO_goodput=qualified/duration,
        evidence='OFFLINE_ORACLE',
        assumptions='Remove observed wait only. Keep other request times, ITL and output work fixed. '
            'No queue feedback, new arrival, nonpreemption or decode interference simulation. Not achievable speedup.')


def read_csv(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def main():
    preserve();prefix='token_confirmation_C37'
    sources={key:ROOT/'analysis'/f'{prefix}_{key}_v0.csv' for key in
        ['request_host_paths','request_SLO','starts','decisions']}
    paths=read_csv(sources['request_host_paths']);requests=read_csv(sources['request_SLO'])
    starts=read_csv(sources['starts']);decisions=read_csv(sources['decisions'])
    identity=lambda r:(int(r['block']),r['method'],r['subset'],r['mode'],r['request_id'])
    by_id={identity(r):r for r in requests};assert len(by_id)==len(requests)==len(paths)==3600
    records=[]
    for path in paths:
        row=by_id[identity(path)]
        item={**row,**{key:float(path[key]) for key in STAGES},
            **{key:float(row[key]) for key in ['arrival','TTFT_ms','max_ITL_ms','TPOT_ms']},
            **{key:int(row[key]) for key in ['block','input_tokens','output_tokens']},
            **{key:row[key]=='True' for key in GATES+('JOINT_PASS',)}}
        assert abs(sum(item[k] for k in STAGES)-item['TTFT_ms'])<1e-5
        assert item['JOINT_PASS']==all(item[k] for k in GATES)
        records.append(item)
    grouped={}
    for row in records:grouped.setdefault((row['block'],row['method'],row['subset'],row['mode']),[]).append(row)
    means=[];counterfactual=[];attribution=[];oracles=[];paired=[];source_failures=[]
    for start in starts:
        key=(int(start['block']),start['method'],start['subset'],start['mode'])
        values=grouped[key];span=float(start['duration_seconds'])
        fields=dict(block=key[0],method=key[1],subset=key[2],mode=key[3])
        means.append(dict(**fields,requests=len(values),mean_TTFT_ms=float(np.mean([r['TTFT_ms'] for r in values])),
            **{s:float(np.mean([r[s] for r in values])) for s in STAGES},
            scope='Disjoint host path means. Enclosures contain host, queue, transfer and GPU work. Not p95 or exclusive GPU costs.'))
        c,a=failure_accounting(values,span)
        counterfactual.extend(dict(**fields,**r) for r in c);attribution.extend(dict(**fields,**r) for r in a)
        oracles.append(dict(**fields,**zero_gate_oracle(values,span)))
        for source in sorted({r['source'] for r in values}):
            selected=[r for r in values if r['source']==source]
            for passed in [True,False]:
                chosen=[r for r in selected if r['JOINT_PASS']==passed]
                source_failures.append(dict(**fields,source=source,joint_pass=passed,
                    request_count=len(chosen),source_requests=len(selected),
                    output_tokens=sum(r['output_tokens'] for r in chosen),
                    source_output_tokens=sum(r['output_tokens'] for r in selected),
                    mean_input_tokens=float(np.mean([r['input_tokens'] for r in chosen])) if chosen else None,
                    mean_output_tokens=float(np.mean([r['output_tokens'] for r in chosen])) if chosen else None))
    for subset in ['Main100','Heldout50']:
        for mode in ['natural','replay']:
            for method in ['Static10','Idle10']:
                for baseline in ['NativeStrong','NativeCommon']:
                    for block in range(3):
                        actual={r['request_id']:r for r in grouped[(block,method,subset,mode)]}
                        control={r['request_id']:r for r in grouped[(block,baseline,subset,mode)]}
                        assert actual.keys()==control.keys()
                        if mode=='replay':assert all(actual[k]['output_tokens']==control[k]['output_tokens'] for k in actual)
                        differences={s:float(np.mean([actual[k][s]-control[k][s] for k in actual])) for s in STAGES}
                        delta=float(np.mean([actual[k]['TTFT_ms']-control[k]['TTFT_ms'] for k in actual]))
                        assert abs(sum(differences.values())-delta)<1e-5
                        paired.append(dict(block=block,subset=subset,mode=mode,method=method,baseline=baseline,
                            matched_request_count=len(actual),mean_TTFT_delta_ms=delta,**differences,
                            scope='Matched request host differences. Geometry, E output and scheduling can differ. Not one-factor causal costs.'))
    paired_summary=[]
    for subset in ['Main100','Heldout50']:
        for mode in ['natural','replay']:
            for method in ['Static10','Idle10']:
                for baseline in ['NativeStrong','NativeCommon']:
                    selected=[r for r in paired if (r['subset'],r['mode'],r['method'],r['baseline'])==(subset,mode,method,baseline)]
                    assert len(selected)==3
                    for stage in STAGES+['mean_TTFT_delta_ms']:
                        result=mean_interval([r[stage] for r in selected])
                        paired_summary.append(dict(subset=subset,mode=mode,method=method,baseline=baseline,
                            stage=stage,delta_ms=result['mean'],CI_low=result['CI'][0],CI_high=result['CI'][1],
                            independent_start_pairs=3,scope='N3 paired start mean differences. Limited bootstrap resolution.'))
    decision_summary=[]
    for subset in ['Main100','Heldout50']:
        for mode in ['natural','replay']:
            for method in ['Static10','Idle10']:
                for block in range(3):
                    chosen=[r for r in decisions if (r['subset'],r['mode'],r['method'],int(r['block']))==(subset,mode,method,block)]
                    enabled=[r for r in chosen if r['selected']=='True']
                    decision_summary.append(dict(subset=subset,mode=mode,method=method,block=block,
                        selected=len(enabled),requests=len(chosen),selected_fraction=len(enabled)/len(chosen),
                        offloaded_prompt_tokens=sum(int(r['offloaded_prompt_tokens']) for r in chosen),
                        total_D_gate_wait_ms=sum(float(r['coordinator_D_wait_ms']) for r in chosen),
                        scope='HTTP-state decision. HTTP pressure is not GPU queue or a future safe window.'))
    outputs={}
    for name,rows in [('stage_means',means),('paired_stage_differences',paired),('paired_stage_summary',paired_summary),
            ('metric_counterfactual',counterfactual),('metric_shapley',attribution),
            ('zero_D_gate_oracle',oracles),('source_failures',source_failures),('decisions',decision_summary)]:
        path=ROOT/'analysis'/f'token_root_cause_C37_{name}_v0.csv';table(path,rows);outputs[name]=str(path.relative_to(ROOT))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for subset in ['Main100','Heldout50']:
        fig,axes=plt.subplots(1,2,figsize=(12,4.6));figrows=[]
        for axis,mode in zip(axes,['natural','replay']):
            bottom=np.zeros(4)
            for stage in STAGES:
                values=[float(np.mean([r[stage] for r in means if r['subset']==subset and r['mode']==mode and r['method']==m])) for m in METHODS]
                axis.bar(range(4),values,bottom=bottom,label=stage);bottom+=values
                figrows.extend(dict(subset=subset,mode=mode,method=m,stage=stage,start_mean_ms=value,
                    independent_starts=3,scope='Disjoint HTTP host means, not exclusive GPU or p95 decomposition.') for m,value in zip(METHODS,values))
            axis.set_xticks(range(4),METHODS,rotation=15);axis.set_ylabel('Mean host path (ms)');axis.set_title(mode);axis.grid(axis='y',alpha=.2)
        axes[1].legend(fontsize=6,loc='upper left',bbox_to_anchor=(1.01,1))
        fig.suptitle(f'{subset} | QPS2 | Independent N3 | Quality pending')
        fig.tight_layout();name=f'token_root_cause_C37_{subset}_host_stages_v0'
        for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=180)
        plt.close(fig);table(ROOT/'figures'/f'{name}.csv',figrows)
    report=ROOT.parent/'mllm_epd_protected_token_critical_path_v1.md';assert not report.exists()
    lines=['# Protected Token Split — Request 경로와 SLO 실패 원인','',
        '증거는 실제 E/P/D Online 결과의 `TRACE_ANALYSIS`다. 전체 Task Quality는 PENDING이다.',
        '동일 Request ID와 고정 Arrival의 독립 Start3회를 연결했다. Natural과 Replay는 별도다.',
        'Host 경계를 중복 없이 합산했다. 이 합계는 Mean TTFT를 설명하며 p95의 가법 분해가 아니다.',
        'P/E HTTP 구간에는 GPU Compute, Queue, Scheduler와 Transfer가 함께 들어 있다.',
        '이 보고서는 그 구간을 GPU-exclusive 계산시간으로 바꾸지 않는다.','',
        '## 같은 요청의 평균 경로 차이','',
        '아래는 Equal-output Replay다. 양수는 후보가 Native보다 느리다는 뜻이다.',
        'NativeStrong은 Stock Compiler다. NativeCommon은 후보와 공통 LLM Compiler 설정을 사용했다.',
        '두 Native 모두 원래 Admission을 유지했다. Candidate의 추가 Suffix 보호 정책은 같은 정책이 아니다.','',
        '| Subset | Candidate | Native | E HTTP Δms | P HTTP Δms | D Coordinator Wait Δms | D Submit→First Δms | Mean TTFT Δms [95% CI] |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for subset in ['Main100','Heldout50']:
        for method in ['Static10','Idle10']:
            for baseline in ['NativeStrong','NativeCommon']:
                selected=[r for r in paired_summary if (r['subset'],r['mode'],r['method'],r['baseline'])==(subset,'replay',method,baseline)]
                values={r['stage']:r for r in selected};total=values['mean_TTFT_delta_ms']
                lines.append(f'| {subset} | {method} | {baseline} | '+
                    ' | '.join(f'{values[s]["delta_ms"]:.3f}' for s in ['E_HTTP_enclosure','P_HTTP_enclosure','D_coordinator_gate_wait','D_submit_to_client_first'])+
                    f' | {total["delta_ms"]:.3f} [{total["CI_low"]:.3f}, {total["CI_high"]:.3f}] |')
    lines+=['','## 확인한 범위','',
        '정적 Token은 기존 D Stream이 끝난 뒤 Suffix를 시작했다. 이를 위해 실제 Coordinator 대기가 발생했다.',
        'D에서 Prefill을 실행하지 않는 Native는 이 추가 Suffix 보호 대기가 없다.',
        '이 차이는 Partition과 Admission의 결합 효과다. 개별 원인의 인과적 비율은 이 표만으로 확정하지 않는다.',
        '낮은 ITL은 긴 Suffix Admission 대기와 동시에 나타날 수 있다. 낮은 ITL만으로 Serving 이득을 판단하지 않는다.',
        '현재 상태 Idle-fallback도 계획 후 실제 D 시작 시점까지 상태가 바뀔 수 있다.',
        '계획 시 HTTP Idle은 GPU의 미래 Decode-safe Budget이 아니다.','',
        '## SLO와 Offline 진단','',
        'TTFT≤5000ms, 모든 Client ITL≤34ms, Request TPOT≤33ms/token을 유지했다.',
        '8개 실패 조합은 원래 N3 분석에 있다. 이번 분석은 Gate 제거 Counterfactual도 계산했다.',
        'Shapley-style Attribution은 실패 요청의 Token 수를 실패한 Gate에 균등 배분한다.',
        '이는 Metric 분해다. 실제 GPU 또는 Host의 인과적 기여율이 아니다.',
        'Zero-D-Gate Oracle은 관측된 Coordinator 대기만 제거한다. 다른 Queue와 Decode 경쟁은 다시 계산하지 않는다.',
        '따라서 Oracle Goodput과 TTFT는 실제 Online Speedup이 아니다.',
        '같은 Output 수의 Replay도 Answer String이 같다는 보장은 없다. Natural Output 변화도 별도로 보존했다.','',
        '## 재현 자료','',
        'Main100 SHA256은 `910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1`다.',
        '모델 Revision은 `cc594898137f460bfe9f0759e9844b3ce807cfb5`다. GPU4/5/6을 E/P/D로 사용했다.',
        'B8192/S1024, KV42GiB, Block128, Original BF16, BI0, 원본 Vision과 Processor를 유지했다.',
        '각 Stage 차이와 CI는 `analysis/token_root_cause_C37_*_v0.csv`에 있다.',
        'PNG/PDF와 대응 CSV는 `figures/token_root_cause_C37_*_v0`에 있다.',
        '모든 Raw/Arrival 검사는 앞선 N3 Recalculation을 통과했다. 기존 실패나 Frozen Manifest를 변경하지 않았다.','']
    report.write_text('\n'.join(lines))
    write(ROOT/'analysis/token_root_cause_C37_audit_v0.json',dict(time=time.time(),
        source_SHA256={key:digest(path) for key,path in sources.items()},outputs=outputs,
        matched_requests=len(records),independent_starts=3,task_quality='PENDING',
        causal_scope='Disjoint host path accounting and metric counterfactuals only. No kernel-exclusive or one-factor causal fraction.'))
    preserve();print('TOKEN_N3_HOST_PATHS_AND_METRIC_FAILURES_RECALCULATED',flush=True)


if __name__=='__main__':main()
