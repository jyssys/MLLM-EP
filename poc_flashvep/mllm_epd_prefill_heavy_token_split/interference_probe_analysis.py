"""Qualify discarded real-prefill probes without calling them serving speedup."""
import argparse
import json
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .metrics import gates
from .native_common_pilot_report import arrival_contract_SHA256
from .token_confirmation_analysis import paired_interval,mean_interval,CATEGORIES
from .token_discovery_analysis import table

KEYS=['TTFT_p50_ms','TTFT_p95_ms','ITL_p99_ms','ITL_max_ms','TPOT_p95_ms','E_HTTP_p95_ms',
    'raw_output_tokens_per_second','SLO_goodput','joint_percent','decode_pass_percent']


def protection_gate(pairs,starts,arm,mode):
    chosen=[r for r in starts if r['arm']==arm and r['mode']==mode]
    assert len(chosen)==3
    def entry(reference,metric):
        return next(r for r in pairs if r['arm']==arm and r['mode']==mode and r['reference']==reference and r['metric']==metric)
    checks={}
    for reference in ['NativeStrong','NativeCommon','ResidentNoProbeMean']:
        checks[reference]={metric:entry(reference,metric)['paired_start_bootstrap95_CI'][1]<=1.05
            for metric in ['E_HTTP_p95_ms','ITL_p99_ms','TPOT_p95_ms','TTFT_p50_ms','TTFT_p95_ms']}
    absolute=all(r['decode_pass_percent']>=95 for r in chosen)
    additional=all(r['accepted_probes']>0 for r in chosen)
    passed=absolute and additional and all(ok for values in checks.values() for ok in values.values())
    return dict(arm=arm,mode=mode,relative_checks=checks,absolute_decode_all_starts=absolute,
        actual_additional_work_all_starts=additional,observed_HTTP_protection_gate=passed,
        decision='PROBE_HTTP_NONINFERIOR_ON_TESTED_TRACES' if passed else 'PROBE_PROTECTION_NOT_CONFIRMED',
        encoder_exclusive_GPU_service_gate='NOT_MEASURED',
        serving_speedup=False,real_request_prefill_offload=False,full_task_quality='PENDING',
        scope='Additional unrelated prefix work only. HTTP service contains queue and host costs. '
            'No safe future-window, prefill-heavy, active-decode insertion or actual request throughput gain claim.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='interference_PROBE_C40R2')
    args=parser.parse_args();tag=args.tag;preserve()
    source=ROOT/'analysis'/f'{tag}_summary.json';summary=json.loads(source.read_text())
    assert summary['status']=='COMPLETE' and len(summary['starts'])==3
    assert (ROOT/'state'/f'{tag}_controller_complete.json').exists()
    starts=[];failures=[];probe_rows=[];memory=[];arrivals=set()
    for block,group in enumerate(summary['starts']):
        groups=[('NativeStrong',group['NativeStrong']),('common',group['common_and_probes'])]
        cached={}
        for family,cells in groups:
            proxy_name=f'{tag}_B{block}_STOCK_proxy' if family=='NativeStrong' else f'{tag}_B{block}_proxy'
            proxy_path=ROOT/'raw'/f'proxy_{proxy_name}.jsonl'
            traces={r['runtime_request_id']:r for r in map(json.loads,proxy_path.read_text().splitlines())}
            for cell in cells:
                label=cell['label']
                if 'warmup' in label:continue
                mode='natural' if label.endswith('_natural') else 'replay'
                assert label.endswith('_'+mode)
                arm='NativeStrong' if family=='NativeStrong' else label.removesuffix('_'+mode)
                if arm=='plain':arm='NativeCommon'
                metric=cell['metrics'];run=metric['run'];raw_path=ROOT/'raw'/f'{run}_requests.jsonl'
                requests=[json.loads(line) for line in raw_path.read_text().splitlines()]
                assert len(requests)==100 and all(r['status']=='MEASURED_ONLINE' for r in requests)
                assert digest(raw_path)==metric['raw_SHA256']
                assert sum(r['output_tokens'] for r in requests)==metric['total_output_tokens']
                assert np.percentile([r['TTFT_ms'] for r in requests],95)==metric['TTFT_p95_ms']
                assert sum(r['output_tokens'] for r in requests if gates(r)['JOINT_PASS'])/metric['duration_seconds']==metric['SLO_goodput']
                arrivals.add(arrival_contract_SHA256(ROOT/'configs'/f'{run}_arrival.json'))
                selected=[traces[r['runtime_request_id']] for r in requests]
                assert all(r['status']=='MEASURED_ONLINE' and 'E_embedding_release' in r for r in selected)
                encoder=[1000*(r['E_complete']-r['E_submit']) for r in selected]
                gate_rows=[gates(r) for r in requests]
                value=dict(block=block,arm=arm,mode=mode,run=run,requests=100,
                    output_tokens=metric['total_output_tokens'],duration_seconds=metric['duration_seconds'],
                    **{k:metric[k] for k in KEYS if k in metric},E_HTTP_p95_ms=float(np.percentile(encoder,95)),
                    joint_percent=100*metric['gate_attainment']['JOINT_PASS'],
                    decode_pass_percent=100*sum(g['ITL_PASS'] and g['TPOT_PASS'] for g in gate_rows)/len(gate_rows),
                    accepted_probes=0,rejected_probes=0,additional_prefix_tokens=0,
                    scope='MEASURED_ONLINE Main100. Probe work excluded from client throughput and SLO numerator.')
                path=ROOT/'analysis'/f'{run}_probes.json'
                if arm in ['E128','E512','D128','D512']:
                    probe=json.loads(path.read_text());accepted=probe['accepted'];quantum=int(arm[1:])
                    assert accepted==len(probe['rows'])<=32 and probe['task_completion'] is False
                    for row in probe['rows']:
                        assert row['sample']['task_completion'] is False and row['sample']['quantum']==quantum
                        assert row['sample']['output']['usage']['completion_tokens']==1
                        assert row['release']['released'] and not row['release']['failed']
                        before=row['lease']['before']['native_state']
                        assert before['prefill_pending']>0
                        if arm.startswith('E'):assert before['encoder_pending']==0
                        else:
                            admission=row['lease']['admission']['admission']['before']
                            assert admission['active_http_requests']==admission['admission_waiting']==0
                            assert admission['prefill_owner'] is None
                        probe_rows.append(dict(block=block,mode=mode,arm=arm,run=run,quantum=quantum,
                            begin=row['sample']['begin'],end=row['sample']['end'],HTTP_wall_ms=row['sample']['HTTP_wall_ms'],
                            discarded_output_tokens=1,additional_prefix_tokens=quantum,
                            additional_token_layer_rows=28*quantum,task_completion=False,
                            evidence='INTERFERENCE_PROBE. Actual full28 prefix with primed original E embeddings. Not imported-KV suffix.'))
                    value.update(accepted_probes=accepted,rejected_probes=probe['rejected'],additional_prefix_tokens=accepted*quantum)
                starts.append(value);cached[(arm,mode)]=value
                for category in CATEGORIES:
                    subset=[r for r in requests if gates(r)['category']==category]
                    failures.append(dict(block=block,arm=arm,mode=mode,category=category,requests=len(subset),
                        request_denominator=100,output_tokens=sum(r['output_tokens'] for r in subset),
                        total_output_tokens=metric['total_output_tokens']))
        for mode in ['natural','replay']:
            before=cached['clean_before',mode];after=cached['clean_after',mode]
            starts.append(dict(block=block,arm='ResidentNoProbeMean',mode=mode,run='DERIVED_BEFORE_AFTER_MEAN',requests=100,
                output_tokens=(before['output_tokens']+after['output_tokens'])/2,
                duration_seconds=(before['duration_seconds']+after['duration_seconds'])/2,
                **{key:(before[key]+after[key])/2 for key in KEYS},accepted_probes=0,rejected_probes=0,
                additional_prefix_tokens=0,scope='Mean of two measured no-probe controls within one initialization. Not another independent start or cohort.'))
        block_data=json.loads((ROOT/'analysis'/f'{tag}_B{block}_summary.json').read_text())
        for phase,ports in block_data['memory'].items():
            for port,value in ports.items():
                memory.append(dict(block=block,phase=phase,port=port,
                    device_total_bytes=value['device_total_bytes'],device_used_bytes=value['device_total_bytes']-value['device_free_bytes'],
                    process_allocated_bytes=value['allocated_bytes'],process_reserved_bytes=value['reserved_bytes'],
                    process_peak_allocated_bytes=value['maximum_allocated_bytes'],
                    scope='Worker memory sample outside timed cohorts. Device use includes co-resident processes.'))
    assert len(arrivals)==1
    pairs=[]
    for mode in ['natural','replay']:
        for arm in ['clean_before','clean_after','E128','E512','D128','D512']:
            actual=sorted([r for r in starts if r['arm']==arm and r['mode']==mode],key=lambda r:r['block'])
            assert len(actual)==3
            for reference in ['NativeStrong','NativeCommon','ResidentNoProbeMean']:
                control=sorted([r for r in starts if r['arm']==reference and r['mode']==mode],key=lambda r:r['block'])
                assert len(control)==3
                for metric in KEYS:
                    pairs.append(dict(arm=arm,mode=mode,reference=reference,metric=metric,
                        **paired_interval([r[metric] for r in actual],[r[metric] for r in control])))
    decisions=[protection_gate(pairs,starts,arm,mode) for arm in ['E128','E512','D128','D512'] for mode in ['natural','replay']]
    for name,rows in [('starts',starts),('failure_categories',failures),('probes',probe_rows),('memory',memory),('paired_ratios',pairs)]:
        table(ROOT/'analysis'/f'{tag}_{name}_v0.csv',rows)
    write(ROOT/'analysis'/f'{tag}_audit_v0.json',dict(time=time.time(),summary_SHA256=digest(source),decisions=decisions,
        E_HTTP_scope='Queue, host and original encode enclosure. Not exclusive encoder GPU service.',
        actual_batch='NOT_MEASURED during clean cohorts',prefill_heavy='NOT_CONFIRMED',task_quality='PENDING',
        metric_recalculation='PASS',arrival_contract_SHA256=next(iter(arrivals)),additional_work_is_not_client_completion=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=['NativeStrong','NativeCommon','clean_before','E128','E512','D128','D512','clean_after']
    for metric,label in [('E_HTTP_p95_ms','E HTTP enclosure p95 (ms)'),('TTFT_p95_ms','Client TTFT p95 (ms)'),
            ('ITL_p99_ms','Client ITL p99 (ms)'),('TPOT_p95_ms','Request TPOT p95 (ms/token)'),
            ('SLO_goodput','Main100 SLO-qualified output tokens/s')]:
        fig,axes=plt.subplots(1,2,figsize=(13,4.8));records=[]
        for axis,mode in zip(axes,['natural','replay']):
            for index,arm in enumerate(arms):
                values=[r[metric] for r in starts if r['arm']==arm and r['mode']==mode]
                stat=mean_interval(values);low,high=stat['CI'];center=stat['mean']
                axis.errorbar(index,center,yerr=[[center-low],[high-center]],fmt='o',capsize=4)
                records.append(dict(mode=mode,arm=arm,metric=metric,start_mean=center,CI_low=low,CI_high=high,
                    independent_starts=3,scope='Unrelated probe work is excluded from Main100 throughput. Not a serving speedup.'))
            axis.set_xticks(range(len(arms)),arms,rotation=35,ha='right');axis.set_ylabel(label);axis.set_title(mode);axis.grid(alpha=.2)
        fig.suptitle('INTERFERENCE_PROBE | Main100 QPS2 | Independent N3 | Quality pending')
        fig.tight_layout();name=tag+'_'+metric+'_v0'
        for extension in ['png','pdf']:fig.savefig(ROOT/'figures'/f'{name}.{extension}',dpi=180)
        plt.close(fig);table(ROOT/'figures'/f'{name}.csv',records)
    report=ROOT.parent/'mllm_epd_encoder_decoder_interference_probe_v1.md';assert not report.exists()
    lines=['# E/D 실제 Prefill Probe — 독립 N3 간섭 검사','',
        '증거는 `INTERFERENCE_PROBE`다. Probe는 실제 요청의 완료에 기여하지 않는다.',
        'Main100은 매번 실제 Online E/P/D를 실행했다. Probe만 별도 Calibration Embedding을 미리 준비했다.',
        '추가 계산으로 Client Throughput이 개선됐다고 주장하지 않는다.','',
        '## 계약','',
        'GPU4/5/6, Original BF16 Qwen2.5-VL-7B, BI0, Full28, 원본 Vision/Processor를 유지했다.',
        '모델 Revision은 cc594898137f460bfe9f0759e9844b3ce807cfb5다.',
        'B8192/S1024, P/D KV42GiB, Block128, Prefix Cache Off다. E Helper는 추가 Full Model과 KV8GiB를 사용했다.',
        '복제된 Vision Weight 메모리도 포함했다. 최대 Encoder Batch Capacity는 측정하지 않았다.',
        '각 Block은 새 Stock Native와 별도의 Common/Probe 초기화를 사용했다.',
        'N1/N8 출력 검사와 E/D128/512의 Same-GPU Teacher KV 검사를 통과한 뒤 측정했다.',
        'Clean/Profile 계측을 Main에 삽입하지 않았다. E HTTP Enclosure는 GPU-exclusive Encode Time이 아니다.',
        'E는 Pending E HTTP가 없고 P HTTP 작업이 있을 때만 Probe를 허용했다.',
        'D는 기존 Stream과 Pending Admission이 모두 없을 때만 Probe를 허용했다.',
        '새 Encode Arrival은 이미 실행 중인 Helper Kernel을 기다릴 수 있다. Stream Priority를 Preemption으로 가정하지 않았다.',
        '128/512는 원래 Full28 Prefix다. Imported Prefix KV를 사용하는 Suffix 비용이 아니다.','',
        '## 사전 고정한 보호 Gate','',
        '| Mode | Probe | Relative HTTP Protection | Absolute Decode≥95% | Actual Additional Work | Decision |',
        '|---|---|---|---|---|---|']
    for item in decisions:
        relative=all(value for fields in item['relative_checks'].values() for value in fields.values())
        lines.append(f'| {item["mode"]} | {item["arm"]} | {relative} | {item["absolute_decode_all_starts"]} | '+
            f'{item["actual_additional_work_all_starts"]} | {item["decision"]} |')
    lines+=['','NativeStrong·NativeCommon·동일 초기화의 No-probe 평균을 각각 대조군으로 사용했다.',
        'E HTTP p95·ITL p99·TPOT p95·TTFT p50/p95의 Paired CI 상한은 각각 1.05 이하여야 한다.',
        'Absolute Decode는 Request별 All-ITL과 TPOT의 AND다. 각각의 Pass Rate만 검사하지 않았다.',
        'N3의 모든 27개 Paired Resample을 사용했다. CI 해상도는 제한적이다.',
        'HTTP 보호 Gate를 통과해도 미래 GPU Safe Window나 Prefill-heavy를 확인한 것은 아니다.',
        '실제 Encoder-exclusive Service p95의 상대 Gate는 미측정이다. 이를 HTTP Enclosure로 대체하지 않았다.',
        '이 결과만으로 Full ENCODE_SAFE_HEADROOM_CONFIRMED 또는 Online Token Split GO를 부여하지 않는다.',
        '모든 실패 조합의 Request Count와 Token Weight를 보존했다. 전체 Task Quality도 계속 PENDING이다.',
        'Original C40R1의 Port Collision 실패와 그 수정은 별도 보고서에 있다. 실패한 실행은 삭제하지 않았다.','',
        '## 재현 자료','',
        f'Config는 `configs/{tag}.json`이다. Main100 Manifest와 SLO는 변경하지 않았다.',
        f'CSV/JSON은 `analysis/{tag}_*_v0`에 있다. Figure는 `figures/{tag}_*_v0`에 있다.',
        'Probe의 실제 HTTP 비용·수락/거부·추가 Token-layer Rows는 별도 CSV에 있다.',
        '추가 Probe의 1-token Sample은 버렸다. Main Request의 Output Token 수에 더하지 않았다.',
        'Natural EOS와 Equal-output Replay의 결과를 합쳐 평균을 계산하지 않았다.','']
    report.write_text('\n'.join(lines));preserve();print('INTERFERENCE_PROBE_N3_RECALCULATED',flush=True)


if __name__=='__main__':main()
