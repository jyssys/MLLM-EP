"""Analyze one-factor Native transport periods without inventing new starts."""
import itertools
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .metrics import gates
from .native_common_pilot_report import arrival_contract_SHA256
from .token_confirmation_analysis import paired_interval,mean_interval,CATEGORIES
from .token_discovery_analysis import table
from .transport_policy_control import TAG,policy_order
from .whole_deflection_analysis import host_path,COMMON_STAGES

KEYS=['TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','ITL_mean_ms','ITL_p99_ms','ITL_max_ms',
    'TPOT_p95_ms','raw_output_tokens_per_second','total_input_output_tokens_per_second',
    'completed_requests_per_second','SLO_goodput','joint_percent','decode_pass_percent']
IMPORT_KEYS=['export_RPC_ms','descriptor_prepare_host_ms','import_clone_sync_host_ms','completion_ACK_host_ms']


def start_policy_means(rows,keys):
    result=[]
    for block in range(3):
        for mode in ['natural','replay']:
            for inline in [True,False]:
                selected=[r for r in rows if (r['block'],r['mode'],r['inline'])==(block,mode,inline)]
                expected={index for index,value in enumerate(policy_order(block)) if value==inline}
                assert len(selected)==2 and {r['period'] for r in selected}==expected
                result.append(dict(block=block,mode=mode,inline=inline,periods=sorted(expected),
                    **{key:float(np.mean([r[key] for r in selected])) for key in keys},
                    independent_initializations=1,periods_are_not_independent_starts=True))
    return result


def paired_delta(actual,reference):
    values=np.asarray(actual,dtype=float)-np.asarray(reference,dtype=float)
    assert values.shape==(3,)
    samples=[float(values[list(indices)].mean()) for indices in itertools.product(range(3),repeat=3)]
    return dict(mean_delta=float(values.mean()),paired_start_bootstrap95_CI=np.percentile(samples,[2.5,97.5]).tolist(),
        independent_start_pairs=3,empirical_resamples=27)


def import_groups(records,lo,hi,inline):
    assert hi>lo
    overlapping=[r for r in records if r['begin']<hi and r['end']>lo]
    assert overlapping and all(lo<=r['begin']<=r['end']<=hi for r in overlapping), 'Import crosses the drained cohort boundary'
    assert len({r['consumer_pid'] for r in overlapping})==len({r['producer_pid'] for r in overlapping})==1
    for row in overlapping:
        assert row['clone_complete_before_ACK'] and row['ack_mode']=='file'
        assert row['inline_images']==(row['images'] if inline else 0)
        if inline:assert row['export_RPC_ms']==0
    return overlapping


def main():
    preserve();source=ROOT/'analysis'/f'{TAG}_summary.json';summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE' and summary['independent_initializations']==3 and len(summary['results'])==24
    assert (ROOT/'state'/f'{TAG}_controller_complete.json').exists()
    plan=ROOT/'configs'/f'{TAG}.json';frozen=json.loads(plan.read_text())
    assert frozen['periods']==[policy_order(index) for index in range(3)]
    manifest=ROOT/'data/main100_manifest_template_v2.json';data=json.loads(manifest.read_text())
    assert digest(manifest)==frozen['manifest_SHA256']
    lookup={r['request_id']:r for r in data['requests']}
    counts={r['request_id']:r['output_tokens'] for r in map(json.loads,
        (ROOT/'raw/reference_main400_natural_template_v2_v0_requests.jsonl').read_text().splitlines())}
    # Read once. These append-only raw records are not rewritten or attributed
    # to individual requests. A scheduler import can contain several images.
    transport=ROOT/'raw/ec_cuda_ipc_prefill_gpu5.jsonl'
    imports=[json.loads(line) for line in transport.open()]
    periods=[];paths=[];failures=[];groups=[];import_periods=[];arrivals=set();provenance=[]
    for cell in summary['results']:
        run=cell['run'];block=cell['block'];period=cell['period'];inline=cell['inline'];mode=cell['mode'];metric=cell['metrics']
        assert inline==policy_order(block)[period]
        raw=ROOT/'raw'/f'{run}_requests.jsonl';requests=[json.loads(line) for line in raw.read_text().splitlines()]
        assert len(requests)==100 and {r['request_id'] for r in requests}==set(lookup)
        assert all(r['status']=='MEASURED_ONLINE' and len(r['token_ids'])==r['output_tokens']==r['usage']['completion_tokens'] for r in requests)
        assert all(r['prompt_SHA256']==lookup[r['request_id']]['prompt_SHA256'] for r in requests)
        assert digest(raw)==metric['raw_SHA256']
        assert sum(r['output_tokens'] for r in requests)==metric['total_output_tokens']
        assert np.percentile([r['TTFT_ms'] for r in requests],95)==metric['TTFT_p95_ms']
        assert np.percentile([v for r in requests for v in r['ITL_ms']],99)==metric['ITL_p99_ms']
        assert np.percentile([r['TPOT_ms'] for r in requests],95)==metric['TPOT_p95_ms']
        assert sum(r['output_tokens'] for r in requests if gates(r)['JOINT_PASS'])/metric['duration_seconds']==metric['SLO_goodput']
        if mode=='replay':assert all(r['output_tokens']==counts[r['request_id']] for r in requests)
        arrivals.add(arrival_contract_SHA256(ROOT/'configs'/f'{run}_arrival.json'))
        done=json.loads((ROOT/'state'/f'{run}_complete.json').read_text())
        assert done['passed'] and not done['serial'] and done['raw_SHA256']==digest(raw)
        assert done['duration_seconds']==metric['duration_seconds']
        row=dict(block=block,period=period,inline=inline,mode=mode,run=run,requests=100,
            output_tokens=metric['total_output_tokens'],duration_seconds=metric['duration_seconds'],
            **{key:metric[key] for key in KEYS if key in metric},
            joint_percent=100*metric['gate_attainment']['JOINT_PASS'],
            decode_pass_percent=sum(gates(r)['ITL_PASS'] and gates(r)['TPOT_PASS'] for r in requests),
            scope='One repeated period. Not an independent server start.')
        periods.append(row)
        proxy=ROOT/'raw'/f'proxy_{TAG}_B{block}_policy_proxy.jsonl'
        traces={r['runtime_request_id']:r for r in map(json.loads,proxy.read_text().splitlines())}
        for request in requests:
            trace=traces[request['runtime_request_id']]
            assert trace['status']=='MEASURED_ONLINE' and not trace['deflected'] and not trace['decision']['selected']
            paths.append(dict(block=block,period=period,inline=inline,mode=mode,request_id=request['request_id'],
                TTFT_ms=request['TTFT_ms'],output_tokens=request['output_tokens'],**host_path(request,trace)))
        for category in CATEGORIES:
            selected=[r for r in requests if gates(r)['category']==category]
            failures.append(dict(block=block,period=period,inline=inline,mode=mode,category=category,
                requests=len(selected),request_denominator=100,output_tokens=sum(r['output_tokens'] for r in selected),
                total_output_tokens=metric['total_output_tokens']))
        lo=done['monotonic_start'];hi=lo+done['duration_seconds']
        selected=import_groups(imports,lo,hi,inline)
        groups.extend(dict(block=block,period=period,inline=inline,mode=mode,run=run,**r) for r in selected)
        imported=dict(block=block,period=period,inline=inline,mode=mode,run=run,
            import_groups=len(selected),images=sum(r['images'] for r in selected),
            logical_embedding_bytes=sum(r['logical_bytes'] for r in selected),
            scope='Host import groups. Not individual requests, independent starts, exclusive GPU kernels or DMA.')
        for key in IMPORT_KEYS:
            imported[key+'_mean']=float(np.mean([r[key] for r in selected]))
            imported[key+'_p95']=float(np.percentile([r[key] for r in selected],95))
        import_periods.append(imported)
        lease=ROOT/'state'/f'server_{TAG}_B{block}_P5.json';worker=json.loads(lease.read_text());assert worker['finished']
        provenance.append(dict(run=run,raw_SHA256=digest(raw),lease_SHA256=digest(lease),GPU_info=worker['GPU_info'],
            source_archive_SHA256=worker['source_archive_SHA256'],producer_pid=selected[0]['producer_pid'],
            consumer_pid=selected[0]['consumer_pid'],monotonic_begin=lo,monotonic_end=hi))
    assert len(arrivals)==1
    means=start_policy_means(periods,KEYS)
    path_periods=[]
    for cell in periods:
        selected=[r for r in paths if (r['block'],r['period'],r['mode'])==(cell['block'],cell['period'],cell['mode'])]
        assert len(selected)==100
        path_periods.append(dict(block=cell['block'],period=cell['period'],inline=cell['inline'],mode=cell['mode'],
            **{key:float(np.mean([r[key] for r in selected])) for key in COMMON_STAGES}))
    path_means=start_policy_means(path_periods,COMMON_STAGES)
    import_means=start_policy_means(import_periods,[key+suffix for key in IMPORT_KEYS for suffix in ['_mean','_p95']])
    ratios=[];statistics=[];deltas=[]
    for mode in ['natural','replay']:
        ordered=lambda rows,inline:sorted([r for r in rows if r['mode']==mode and r['inline']==inline],key=lambda r:r['block'])
        for inline in [True,False]:
            selected=ordered(means,inline)
            statistics.extend(dict(mode=mode,inline=inline,metric=key,**mean_interval([r[key] for r in selected])) for key in KEYS)
        for key in KEYS:
            actual=[r[key] for r in ordered(means,True)];reference=[r[key] for r in ordered(means,False)]
            ratios.append(dict(mode=mode,metric=key,ratio_direction='Inline / On-demand',**paired_interval(actual,reference)))
        for scope,values,keys in [('Disjoint mean HTTP stage',path_means,COMMON_STAGES),
                ('Mean of period-level import group summaries',import_means,[k+s for k in IMPORT_KEYS for s in ['_mean','_p95']])]:
            for key in keys:deltas.append(dict(mode=mode,metric=key,scope=scope,delta_direction='Inline minus On-demand (ms)',
                **paired_delta([r[key] for r in ordered(values,True)],[r[key] for r in ordered(values,False)])))
    for label,rows in [('periods',periods),('start_policy_means',means),('request_host_paths',paths),('failure_categories',failures),
            ('import_groups',groups),('import_periods',import_periods),('import_start_means',import_means),
            ('paired_ratios',ratios),('mean_CI',statistics),('paired_host_deltas',deltas)]:
        table(ROOT/'analysis'/f'{TAG}_{label}_v0.csv',rows)
    write(ROOT/'analysis'/f'{TAG}_audit_v0.json',dict(time=time.time(),source_SHA256=digest(source),config_SHA256=digest(plan),
        manifest_SHA256=digest(manifest),arrival_contract_SHA256=next(iter(arrivals)),provenance=provenance,
        cohorts=24,independent_initializations=3,within_start_averaging='Two periods per policy before independent-start inference',
        metric_recalculation='PASS',single_factor='Original E inline descriptor policy only',
        already_inline_controls='NativeStrong and Static Token are not repaired or weakened by this control.',
        task_quality='PENDING',full_GO=False,GPU_exclusive_import_cost='NOT_MEASURED',
        scope='Native transport root-cause experiment. No Token, no Whole-request Deflection and no measured safe-window claim.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,label in [('TTFT_p95_ms','TTFT p95 (ms)'),('ITL_p99_ms','Client ITL p99 (ms)'),
            ('TPOT_p95_ms','Request TPOT p95 (ms/token)'),('SLO_goodput','SLO output tokens/s')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4));figure_rows=[]
        for axis,mode in zip(axes,['natural','replay']):
            for index,inline in enumerate([True,False]):
                item=next(r for r in statistics if (r['mode'],r['inline'],r['metric'])==(mode,inline,key));low,high=item['CI']
                axis.errorbar(index,item['mean'],yerr=[[item['mean']-low],[high-item['mean']]],fmt='o',capsize=4)
                figure_rows.append(dict(mode=mode,inline=inline,metric=key,start_mean=item['mean'],CI_low=low,CI_high=high,
                    independent_starts=3,periods_per_policy_within_start=2,evidence='MEASURED_ONLINE Native one-factor control; quality pending'))
            axis.set_xticks([0,1],['Inline','On-demand']);axis.set_title(mode);axis.set_ylabel(label);axis.grid(alpha=.2)
        fig.suptitle('Native EC descriptor policy | ABBA within independent N3 | Not a new serving method');fig.tight_layout()
        name=f'{TAG}_{key}_v0'
        for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
        plt.close(fig);table(ROOT/'figures'/f'{name}.csv',figure_rows)
    target=ROOT.parent/'mllm_epd_native_ec_descriptor_causal_control_v1.md';assert not target.exists()
    lines=['# Native EC Descriptor Policy — 한 요소 인과 검사','',
        'Native Full28 E/P/D에서 E Descriptor Policy 하나만 변경했다. Token Split과 Whole-request Deflection은 없다.',
        '각 독립 초기화에서 ABBA 또는 BAAB를 실행했다. 각 Policy의 두 Period 평균을 먼저 계산했다.',
        '독립 Start는 3회다. 24개 Cohort를 24회 독립 Start로 취급하지 않았다.',
        'Natural EOS와 Equal-output Replay를 별도로 평가했다. 전체 Task Quality는 PENDING이다.','',
        '| Output | Metric | Inline/On-demand | Paired 95% CI |','|---|---|---:|---|']
    for row in ratios:lines.append(f'| {row["mode"]} | {row["metric"]} | {row["ratio_of_start_means"]:.6f} | {row["paired_start_bootstrap95_CI"]} |')
    lines+=['','## 해석 범위','',
        '두 Policy 모두 원본 E·Full P·NIXL D와 같은 공통 Compiler·Coordinator를 사용했다.',
        '원래 Lifetime Guard가 비어 있는 Image/Descriptor 상태를 확인한 후에만 Policy를 변경했다.',
        'Clone 완료와 모든 ACK를 유지했다. Tensor나 출력값을 수정하지 않았다.',
        'Inline Arm의 모든 실제 P Import에서 Export RPC가 0인지 검사했다. On-demand Import도 원시 기록으로 보존했다.',
        'Import Group에는 여러 이미지나 요청이 포함될 수 있다. Group 시간을 요청별 GPU 비용으로 배분하지 않았다.',
        'Clone/Sync Host Enclosure에는 E Ready Event 대기가 포함될 수 있다. 이를 순수 DMA나 제거 가능한 낭비로 단정하지 않았다.',
        '요청별 다섯 공통 Host 구간의 합은 TTFT와 일치한다. 세부 P 구간을 다시 더하지 않았다.',
        'NativeStrong과 기존 Static Token은 이미 필요한 Descriptor를 미리 발행했다.',
        '이 인과 검사의 개선율을 두 기존 Method의 개선으로 옮기지 않았다.',
        'Independent N3의 27개 Paired Bootstrap 조합을 사용했다. CI 해상도는 제한적이다.',
        '이 결과는 Native 전송 경로의 인과 검사다. Prefill-heavy·Safe Headroom·Online Token Split 성공의 증거는 아니다.','',
        '## 재현 정보','',
        'Qwen2.5-VL-7B BF16 Revision cc594898137f460bfe9f0759e9844b3ce807cfb5, GPU4/5/6을 사용했다.',
        'Main100, QPS2, B8192/S1024, KV42GiB, Block128, Prefix Cache Off와 원래 SLO를 유지했다.',
        f'Config는 `configs/{TAG}.json`이다. 모든 CSV·Audit·Figure는 `{TAG}_*_v0`에 있다.',
        '실제 Clean GPU Batch와 GPU-exclusive Transfer는 NOT_MEASURED다. Finite-cohort 처리량을 Sustainable Capacity로 부르지 않았다.','']
    target.write_text('\n'.join(lines));preserve();print('NATIVE_EC_ONE_FACTOR_N3_RECALCULATED',flush=True)


if __name__=='__main__':main()
