"""Test request-registration tails against unchanged client-visible metrics."""
import bisect
from functools import cache
import json
import time
import numpy as np
from .common import ROOT, REPO, write
from .confirmation_analysis import ratio
from .final_report_tables import interval
from .final_service_decomposition import pair, absolute_interval, STAGES
from .final_confirmation_analysis import FIELDS, complete_metric


def percentiles(values):
    if not values:
        return None
    return dict(n=len(values), mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), maximum=float(max(values)))


@cache
def transition(name):
    folder = ROOT / 'raw' / name
    requests = [json.loads(line) for line in (folder / 'requests.jsonl').read_text().splitlines()]
    steps = json.loads((folder / 'engine_steps_decode.json').read_text())['rows']
    starts = [s['start'] for s in steps]
    accepts = [json.loads(line) for line in (folder / 'admissions.jsonl').read_text().splitlines()]
    accepts = [r for r in accepts if r['role'] == 'decode']
    rows = []
    for request in requests:
        events = request['token_events']
        if len(events) < 2:
            continue
        first, second = events[:2]
        old = bisect.bisect_right(starts, first['worker_time']) - 1
        new = bisect.bisect_right(starts, second['worker_time']) - 1
        assert old >= 0 and new >= old
        left, right = first['worker_time'], steps[new]['start']
        accepted = [r['request_id'] for r in accepts if left <= r['worker_accepted'] <= right]
        rows.append(dict(request_id='q' + str(request['sample_id']), prompt_id=request['prompt_id'],
            actual_output_tokens=request['output_tokens'],
            first_transition_client_ms=1000 * (second['time'] - first['time']),
            first_transition_worker_ms=1000 * (second['worker_time'] - first['worker_time']),
            next_token_step_host_wall_ms=1000 * (steps[new]['end'] - steps[new]['start']),
            next_token_step_preceding_host_gap_ms=1000 * (steps[new]['start'] - steps[new - 1]['end']) if new else None,
            first_emission_to_next_token_step_start_ms=1000 * (right - left),
            intervening_host_steps=new - old - 1,
            new_D_registrations_before_next_token_step=accepted))
    registration = folder / 'D_request_registration_deferral.json'
    log = json.loads(registration.read_text()) if registration.exists() else None
    queued={};waits=[];events=[]
    valid={'q'+str(r['sample_id']) for r in requests}
    if log:
        for event in log['rows']:
            rid=event.get('request_id')
            if event['kind']!='add' or rid not in valid:
                continue
            if event['event']=='RECEIVE_COMMAND' and event['deferred']:
                assert rid not in queued
                queued[rid]=event['time'];events.append((event['time'],1))
            elif event['event']=='RELEASE_COMMAND':
                begin=queued.pop(rid)
                waits.append(dict(request_id=rid,receive_to_release_ms=1000*(event['time']-begin)))
                events.append((event['time'],-1))
        assert not queued, 'Every deferred request must leave the host-registration queue'
    depth=0;maximum_depth=0
    for _,delta in sorted(events):
        depth+=delta;assert depth>=0;maximum_depth=max(maximum_depth,depth)
    assert depth==0
    window=json.loads((folder/'window.json').read_text())
    host_queue=dict(deferred_requests=len(waits),maximum_requests=maximum_depth,
        mean_requests=sum(r['receive_to_release_ms'] for r in waits)/1000/(window['end']-window['start']),
        conditional_wait_ms=percentiles([r['receive_to_release_ms'] for r in waits]),rows=waits,
        scope='Separate host-registration queue,not the vLLM scheduler or decode-ready queue. '
              'Receive-to-release excludes prior coordinator/IPC waiting. Mean occupancy uses the measured client window.')
    return dict(name=name, rows=rows,
        D_host_registration_queue=host_queue,
        first_transition_client_ms=percentiles([r['first_transition_client_ms'] for r in rows]),
        first_emission_to_next_token_step_start_ms=percentiles([
            r['first_emission_to_next_token_step_start_ms'] for r in rows]),
        next_step_host_wall_ms=percentiles([r['next_token_step_host_wall_ms'] for r in rows]),
        transitions_with_new_D_registration=sum(bool(r['new_D_registrations_before_next_token_step']) for r in rows),
        IPC_receive_ms=percentiles([r['receive_host_ms'] for r in log['rows']
                                   if r['event'] == 'RECEIVE_COMMAND']) if log else None,
        registration_lifetime_pass=log['passed'] if log else None,
        scope='First two worker output notifications. Co-delivery and asynchronous output can affect this diagnostic. '
              'The primary ITL still includes every token interval. Host step/gap is not GPU-exclusive time or physical idle. '
              'A registration completion inside a gap is overlap,not exclusive CPU causal attribution.')


def main(plan_name='request_registration_deferral_frozen_v1.json', suffix=''):
    plan = json.loads((ROOT / 'configs' / plan_name).read_text())
    common_native=plan.get('common_native',dict(LayerRegistration='NativeRegistration',
        TokenRegistration='NativeCompletionRegistration',HybridRegistration='NativeRegistration'))
    groups = []
    for stage in ('OBSERVED_TRACE_DIAGNOSTIC', 'INDEPENDENT_CONFIRMATION'):
        cells = [c for c in plan['cells'] if c['stage'] == stage]
        if not cells:
            continue
        arms = list(cells[0]['jobs'])
        values = {a: [] for a in arms}
        starts = []
        for cell in cells:
            statuses = {}
            for arm, name in cell['jobs'].items():
                folder = ROOT / 'raw' / name
                receipt = folder / 'dispatcher_result.json'
                status = 'NOT_RUN'
                if receipt.exists():
                    dispatch = json.loads(receipt.read_text())
                    assert dispatch['source_SHA256'] == plan['source_capsule_SHA256']
                    status = 'FAILED'
                    if dispatch['complete']:
                        quality = ROOT / 'analysis' / (name + '_quality_gate.json')
                        if quality.exists():
                            status = 'MEASURED_ONLINE' if json.loads(quality.read_text())['passed'] else 'FAILED_TASK_QUALITY'
                            if status == 'MEASURED_ONLINE':
                                metric_path = ROOT / 'analysis' / (name + '_metrics.json')
                                metric = complete_metric(json.loads(metric_path.read_text()) if metric_path.exists() else {}, name)
                                values[arm].append(metric)
                statuses[arm] = dict(name=name, status=status)
            starts.append(dict(seed=cell['seed'], GPU_pair=cell['GPU_pair'], arms=statuses))
        complete = all(len(v) == len(cells) for v in values.values())
        row = dict(stage=stage, starts=starts, required_starts=len(cells), complete=complete,
                   QPS=cells[0]['QPS'],common_native=common_native,
                   arms={a: dict(valid_starts=len(v), means={f:float(np.mean([m[f] for m in v]))
                       for f in FIELDS} if v else {}) for a, v in values.items()}, comparisons={})
        for arm in arms:
            row['arms'][arm]['actual_geometry']=[v['actual_geometry'] for v in values[arm]]
            contracts=[]
            for cell in cells:
                cfg=json.loads((ROOT/'configs'/(cell['jobs'][arm]+'.json')).read_text())
                contracts.append(dict(Source_credit=cfg.get('continuous_source_credit_limit',0),
                    execution_family=cfg['execution_family'],D_cap=cfg.get('D_running_admission_cap'),
                    compute=bool(cfg.get('d_compute_throttle')),READ=bool(cfg.get('defer_D_remote_setup')),
                    completion=bool(cfg.get('D_prefill_completion_guard')),
                    registration=bool(cfg.get('defer_D_request_registration')),
                    dataset_SHA256=cfg['dataset_SHA256'],source_SHA256=cfg['source_capsule_SHA256']))
            row['arms'][arm]['runtime_contracts']=contracts
        if complete:
            tests = [('TokenRegistration', 'TokenOldControl'),
                     ('TokenRegistration', common_native['TokenRegistration'])]
            if stage == 'INDEPENDENT_CONFIRMATION':
                tests += [(a, ref) for a, common in common_native.items()
                          for ref in ('NativeStrong', common)]
                tests += [('HybridRegistration', a) for a in ('LayerRegistration', 'TokenRegistration')]
                tests += [('LayerRegistration', 'TokenRegistration')]
            for arm, ref in tests:
                if arm + '_over_' + ref in row['comparisons']:
                    continue
                paired = []
                for cell in cells:
                    left, right = cell['jobs'][arm], cell['jobs'][ref]
                    paired.append(pair(left, right))
                metrics = {f: ratio([m[f] for m in values[arm]], [m[f] for m in values[ref]])
                    if len(cells) >= 3 else dict(estimate=values[arm][0][f] / values[ref][0][f]
                    if values[ref][0][f] > 0 else None, CI95=None, starts=1) for f in FIELDS}
                row['comparisons'][arm + '_over_' + ref] = dict(ratios=metrics,
                    decode_5percent_CI_pass=None if len(cells) < 3 else all(
                        metrics[f]['CI95'][1] <= 1.05 for f in ('ITL_p99_ms', 'TPOT_p95_ms')),
                    absolute_decode_pass=all(m['decode_SLO_attainment'] >= .95 for m in values[arm]),
                    paired_host_boundaries=paired)
            row['first_transition_diagnostics'] = {
                arm: [transition(c['jobs'][arm]) for c in cells] for arm in arms}
        groups.append(row)
    output = dict(time=time.time(), source_capsule_SHA256=plan['source_capsule_SHA256'],
        groups=groups, complete=all(r['complete'] for r in groups), evidence='MEASURED_ONLINE_ADAPTIVE_REPAIR',
        scope='Adaptive follow-up after observed host gaps. Original main140 and READ-repair results stay unchanged. '
              'Each plan fixes its independent seeds and controls before that plan starts performance measurements. '
              'The q1.8 plan precedes its pilot. The q2.8 extension follows the first q1.8 pilot and precedes all q2.8 measurements. '
              'Confirmation uses same400 requests and new arrivals. '
              'Registration deferral also changes admission timing and subsequent batching. '
              'No timestamp changes,no exclusive GPU claim,no active-decode safe-window predictor or dynamic 2D claim.')
    result_name='request_registration_deferral'+suffix+'_results_v1.json'
    write(ROOT / 'analysis' / result_name, output)
    print('REQUEST_REGISTRATION_ANALYSIS', [(r['stage'], r['complete']) for r in groups], flush=True)
    if not output['complete']:
        return
    row = groups[-1]
    qps=row['QPS']
    seeds=[s['seed']for s in row['starts']]
    pair_description=str(row['starts'][0]['GPU_pair'])
    lines = [f'# LongBench400: D Request 등록과 첫 Decode Tail — QPS{qps}', '',
        'Evidence: `MEASURED_ONLINE_ADAPTIVE_REPAIR` 및 `TRACE_ANALYSIS`.',
        '원래 Main140과 READ Repair는 유지했다. 이 문서는 별도 후속 검사다.',
        '먼저 관측한 Heldout200 Host Gap을 사용해 가설을 세웠다.',
        '새 성능 측정 전에 3개 새 Seed와 7개 대조군을 동결했다.',
        f'Confirmation Seed는 {seeds}다. 모두 GPU Pair {pair_description}를 사용했다.',
        f'Natural400, QPS{qps}, BI0, BF16, B8192/S1024와 KV42GiB/Block128을 유지했다.',
        'Model Revision은 `b968826d9c46dd6066d109eabc6255188de91218`이다.',
        'Dataset SHA256은 `2c67ccb92835ec2ca308ba745d52247e809be1d3012bd100843d954272787f67`이다.',
        'Source Capsule SHA256은 `'+plan['source_capsule_SHA256']+'`이다.',
        'Prefix Cache를 사용하지 않았다. 모든 Decode는 Full36 Layers다.',
        'Primary Latency는 Coordinator Client-visible 값이다. HTTP Server Latency가 아니다.',
        '이 비교의 GPU-exclusive Operator 비중은 `NOT_MEASURED`다.',
        '이는 같은 Request의 새로운 Arrival 검사다. 새로운 Request Population은 아니다.', '',
        'QPS2.8 Extension은 첫 QPS1.8 Pilot 관측 후 동결했다.',
        '그러나 모든 QPS2.8 성능 측정 전에 Seed와 Arm을 고정했다.',
        '이를 원래 Main의 사전 맹검 Test와 같은 설계로 표시하지 않는다.', '',
        '## 변경과 경쟁 가설', '',
        'H1은 새 Request 등록과 Input Processing이 다음 Decode 제출을 지연한다는 가설이다.',
        'H2는 IPC Receive 비용이 대기를 지배한다는 가설이다. 새 정책도 IPC Receive는 계속 수행한다.',
        'H3은 첫 Decode 준비 또는 Batch 변동이 지배한다는 가설이다.',
        '새 정책은 Output이 시작된 Running Request가 있으면 새 `engine.add_request` 호출을 미룬다.',
        '기존 Decode Step과 NIXL ACK 및 Retirement는 계속 실행한다.',
        '받은 Request는 버리지 않는다. Stop 전에 모든 추가 Request를 처리한다.',
        '새 Admission 시점은 이후 Batch도 바꾼다. 단독 Python 비용의 인과 효과로 단정하지 않는다.', '',
        '같은 Source의 등록-only A/B 대조군은 TokenRegistration과 TokenOldControl이다.',
        'Hybrid의 READ-only A/B는 별도 READ Repair에서 수행했다.',
        '이번 Hybrid와 원래 Compute-only Main의 차이를 등록-only 효과라고 부르지 않는다.', '',
        '## 독립 Start 3회의 실제 Metric', '',
        '| Arm | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | Raw output tokens/s | SLO-goodput(output tokens/s) | Mean Joint SLO(0–1) | Output Tokens/Start |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    display = ('TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'SLO_qualified_output_tokens_per_second')
    table_fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','generated_output_tokens_per_second',
                  'SLO_qualified_output_tokens_per_second','Joint_SLO_attainment')
    for arm in arms:
        means = row['arms'][arm]['means']
        lengths = [json.loads((ROOT / 'analysis' / (s['arms'][arm]['name'] + '_metrics.json')).read_text())['output_tokens']
                   for s in row['starts']]
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{means[f]:.6f}' for f in table_fields) + ' | ' + str(lengths) + ' |')
    lines += ['', 'Joint SLO는 TTFT와 Request별 Decode 조건을 함께 만족한 비율이다.',
        '표는 Start별 비율의 평균이다. 평균 Joint≥95%가 모든 Start의 Joint≥95%를 보장하지 않는다.',
        'Decode 보호와 Joint SLO를 구분한다. Raw Throughput과 SLO-goodput도 다른 Metric이다.',
        '처리량은 유한400개 Arrival부터 Drain까지 측정했다. Sustainable Capacity가 아니다.',
        '', '### 관측한 가장 큰 Client Interval', '',
        '| Arm | 각 Start의 Max ITL(ms) | 관측 전체 Max ITL(ms) |',
        '|---|---|---:|']
    for arm in arms:
        maxima=[json.loads((ROOT/'analysis'/(s['arms'][arm]['name']+'_metrics.json')).read_text())['max_ITL_ms']
                for s in row['starts']]
        lines.append('| '+arm+' | '+str([round(v,6) for v in maxima])+f' | {max(maxima):.6f} |')
    lines += ['', '이는 관측 Max다. 미래 지연의 최악 상한이 아니다.',
        'p99 또는 TPOT 비열등성 통과가 모든 Interval의 동일성이나 34ms 이하를 보장하지 않는다.',
        'Request별 모든 ITL과 TPOT를 사용하는 Absolute Decode Attainment도 별도로 검사한다.']
    lines += ['', '## Paired CI와 Decode 보호', '',
        '| Arm / Reference | TTFT 비율 [95% CI] | ITL 비율 [95% CI] | TPOT 비율 [95% CI] | Goodput 비율 [95% CI] | Decode 5% | Absolute Decode≥95% |',
        '|---|---|---|---|---|---|---|']
    for name, test in row['comparisons'].items():
        lines.append('| ' + name.replace('_over_', ' / ') + ' | ' +
            ' | '.join(interval(test['ratios'][f]) for f in display) +
            f" | {test['decode_5percent_CI_pass']} | {test['absolute_decode_pass']} |")
    lines += ['', 'CI는 Paired Server Start 단위다. N3에서 27개 Bootstrap 조합을 완전 열거했다.',
        'Pointwise Empirical 95% CI다. 다중 비교 보정은 하지 않았다.',
        '각 Arm은 NativeStrong과 같은 D 정책의 Native를 모두 통과해야 한다.',
        '공통 Native의 정확한 Mapping은 다음과 같다: '+str(common_native)+'.',
        '실패한 Gate를 TTFT 개선으로 대체하지 않는다.']
    lines += ['', '## 같은 Request의 Host 경계 차이', '',
        '아래 값은 왼쪽 Arm에서 Reference를 뺀 Start별 평균의 Paired CI다.',
        '단위는 ms다. 같은 Request ID, Prompt, Source, Pair와 Arrival을 비교한다.',
        '여섯 구간의 평균 차이는 Mean TTFT 차이에 합산된다. 구간별 CI는 합산하지 않는다.',
        '이는 TTFT p95 또는 GPU-exclusive Operator 비용의 분해가 아니다.', '',
        '| Arm − Reference | Arrival→P Prepare | P Prepare→Ready | Ready→D Submit | D Submit→Prepare | D Prepare→First Worker | Worker→Client | Mean TTFT 차이 | 첫 Output 이후 Residency 차이 |',
        '|---|---|---|---|---|---|---|---|---|']
    for key in ('LayerRegistration_over_TokenRegistration',
                'HybridRegistration_over_TokenRegistration',
                'HybridRegistration_over_LayerRegistration',
                'TokenRegistration_over_TokenOldControl'):
        pairs=row['comparisons'][key]['paired_host_boundaries']
        values=[absolute_interval([p['stages_ms'][stage] for p in pairs]) for stage in STAGES]
        values += [absolute_interval([p[field] for p in pairs]) for field in
                   ('mean_TTFT_delta_ms','post_first_output_residency_delta_ms')]
        def delta(item):
            bounds=item['CI95_ms']
            return f'{item["mean"]:.3f} [{bounds[0]:.3f},{bounds[1]:.3f}]' if bounds else f'{item["mean"]:.3f} [N/A]'
        lines.append('| '+key.replace('_over_',' − ')+' | '+' | '.join(delta(v) for v in values)+' |')
    lines += ['', 'P Prepare는 Host Input 준비 경계다. 실제 GPU Kernel 시작 시각이 아니다.',
        'C3 Prepare→Ready는 Chunk Packing, 다른 Request와 Host 작업도 포함한다.',
        'Arrival→P Prepare 차이를 Credit만의 효과 또는 Attention만의 효과로 단정하지 않는다.',
        '첫 Output 이후 Residency는 TTFT에 더하지 않았다.',
        'Natural Output 수가 Arm별로 다를 수 있다. 같은 Output 길이의 Controlled 실험과 구분한다.',
        'Credit, Compile과 실제 Batch가 다른 Best-versus-best 비교다. 순수한 Axis-only 인과 효과가 아니다.', '',
        '## 실제 Batch와 Runtime', '',
        '| Arm | Source Credit | Compile Family | D Cap | P / D-prefill / D-decode 최대 Request Batch |',
        '|---|---|---|---|---|']
    for arm in arms:
        arm_row=row['arms'][arm]
        pmax=max(g['prefill']['maximum_requests'] for g in arm_row['actual_geometry'])
        hist=[h for g in arm_row['actual_geometry'] for h in g['decode']['prefill_decode_request_histogram']]
        dpre=max(h['prefill'] for h in hist);ddec=max(h['decode'] for h in hist)
        contracts=arm_row['runtime_contracts']
        credits=sorted({c['Source_credit'] for c in contracts})
        families=sorted({c['execution_family'] for c in contracts})
        caps=sorted({str(c['D_cap']) for c in contracts})
        lines.append(f'| {arm} | {credits} | {families} | {caps} | {pmax}/{dpre}/{ddec} |')
    lines += ['', '위 Batch는 3개 Start 전체의 최대값이다. 모든 Step의 고정 Batch가 아니다.',
        'C3 Source Credit0은 별도 Coordinator Outstanding 제한을 끈 값이다. 요청0개를 허용한다는 뜻이 아니다.',
        'Worker B/S, KV Capacity와 Admission 제한은 그대로 적용한다. C0의 별도 Credit 설정과 구분한다.',
        '이 후속 검사는 Discovery에서 동결한 Best-versus-best C3 설정을 유지한다.',
        'Actual Batch1/1/1, 같은 Source Credit 또는 정확한 Iso-work를 강제한 Mechanism Control은 아니다.',
        '같은 B/S만으로 Axis별 실제 Geometry가 같다고 주장하지 않는다.',
        'Native의 D-prefill 열은 Last-prompt Query 재계산도 포함한다. Layer/Token Offload라고 부르지 않는다.',
        '모든 Batch Histogram과 Policy Flag는 전체 JSON에 있다.', '',
        '## 별도 Host 등록 Queue', '',
        '| Arm | Deferred Requests / 1200 Requests | 평균 Pending Requests | 최대 Pending Requests | Start-mean Conditional Wait p95(ms) |',
        '|---|---:|---:|---:|---:|']
    for arm in arms:
        queues=[v['D_host_registration_queue'] for v in row['first_transition_diagnostics'][arm]]
        waits=[q['conditional_wait_ms']['p95'] for q in queues if q['conditional_wait_ms']]
        lines.append(f'| {arm} | '+str(sum(q['deferred_requests'] for q in queues))+' | '+
            f"{np.mean([q['mean_requests'] for q in queues]):.6f} | "+
            str(max(q['maximum_requests'] for q in queues))+' | '+
            (f'{np.mean(waits):.6f}' if waits else 'N/A')+' |')
    lines += ['', 'Queue 값은 이 Host 등록 Queue만 측정한다.',
        '정책을 사용하지 않은 Arm의 0은 vLLM Admission 또는 Remote-KV Queue가 비었다는 뜻이 아니다.',
        'Conditional Wait p95는 실제로 미룬 Request만 사용한 Start별 p95의 평균이다.',
        '이를 모든 Request 또는 모든 Start를 합친 p95로 바꾸어 읽지 않는다.', '',
        '## 관측 범위', '',
        '첫 Transition 진단은 첫 두 Worker Output Notification을 비교한다.',
        '비동기 Output과 Co-delivery가 이 진단에 영향을 줄 수 있다.',
        '미룬 Request는 별도 Host 등록 Queue에 남는다. 이를 빈 vLLM Queue에 숨기지 않는다.',
        'Receive→Release 대기와 Queue 길이는 전체 JSON의 `D_host_registration_queue`에 기록했다.',
        '이 대기는 앞선 Coordinator 또는 IPC 대기를 포함하지 않는다.',
        'Primary ITL은 모든 실제 Client Interval을 유지한다.',
        'Host Step과 Host Gap은 GPU-exclusive 시간 또는 Physical Idle이 아니다.',
        'Natural Output 수와 실제 Continuous Batch는 Arm별로 다를 수 있다.',
        'Normal Completion의 State와 Lifetime을 검사했다. Cancellation과 Remote Failure는 별도 미검증이다.',
        '새 정책은 미래 EOS, Arrival 또는 Decode-safe Window를 예측하지 않는다.', '',
        f'- [동결한 설계](dense_ar_longbench400_2d_20261121/configs/{plan_name})',
        f'- [전체 Metric, 같은 ID 경계와 첫 Transition](dense_ar_longbench400_2d_20261121/analysis/{result_name})', '']
    (REPO / 'poc_flashvep/reports' / ('dense_ar_longbench400_request_registration_tail'+suffix+'.md')).write_text(
        '\n'.join(lines), encoding='utf-8')
    figures(row,suffix)


def figures(row,suffix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .confirmed_figures import mean_ci
    from .figures import save
    arms = list(row['arms'])
    colors = {'NativeStrong':'#777777', 'NativeRegistration':'#009e73',
        'NativeCap2Registration':'#56b4e9',
        'NativeCompletionRegistration':'#56b4e9', 'LayerRegistration':'#0072b2',
        'TokenRegistration':'#cc79a7', 'HybridRegistration':'#e69f00', 'TokenOldControl':'#8c564b'}
    qps=row['QPS'];pair_description=str(row['starts'][0]['GPU_pair'])
    caption = (f'MEASURED ONLINE ADAPTIVE HOST-ADMISSION REPAIR | Natural400 | QPS{qps} | BI0,B8192/S1024,KV42GiB.\n'
        f'Three paired independent starts on{pair_description},new arrival seeds,frozen before this extension. Same400 requests,not new-request generalization.\n'
        'Error bars:pointwise start-mean empirical95% CI,27 resamples. Natural output counts and continuous batch can differ.\n'
        'Original140-start main and READ follow-up remain unchanged. No future safe-window prediction,dynamic 2D gain,or sustainable-capacity claim.')
    for field, ylabel, stem in (
        ('TTFT_p95_ms','TTFT p95 (ms)','ttft'),
        ('ITL_p99_ms','Client-visible ITL p99 (ms)','itl'),
        ('TPOT_p95_ms','Request TPOT p95 (ms/token)','tpot'),
        ('SLO_qualified_output_tokens_per_second','SLO-qualified output tokens/s','goodput')):
        fig, ax = plt.subplots(figsize=(12,5.6))
        points = []
        for index, arm in enumerate(arms):
            values = [json.loads((ROOT/'analysis'/(s['arms'][arm]['name']+'_metrics.json')).read_text())[field]
                      for s in row['starts']]
            mean, ci = mean_ci(values)
            ax.bar(index,mean,color=colors[arm],alpha=.8)
            ax.errorbar(index,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='none',ecolor='black',capsize=3)
            points.append(dict(arm=arm,mean=mean,CI95=ci,independent_starts=3))
        labels=[a.replace('Registration','Reg') for a in arms]
        ax.set(xticks=range(len(arms)),xticklabels=labels,ylabel=ylabel,
               title=f'D request-registration intervention | Natural400 | QPS{qps}')
        ax.tick_params(axis='x',rotation=23)
        ax.grid(axis='y',alpha=.2)
        save(fig,'registration_repair'+suffix+'_confirmed_'+stem,caption,
             dict(evidence='MEASURED_ONLINE_ADAPTIVE_REPAIR',metric=field,points=points))
    fig, ax = plt.subplots(figsize=(11,5.4))
    points=[]
    for index,(arm,common) in enumerate(row['common_native'].items()):
        for offset,ref,style in ((-.18,'NativeStrong','o'),(.18,common,'s')):
            test=row['comparisons'][arm+'_over_'+ref]
            for metric,dx,color in (('ITL_p99_ms',-.055,'#0072b2'),('TPOT_p95_ms',.055,'#d55e00')):
                result=test['ratios'][metric]; value=result['estimate']; low,high=result['CI95']
                ax.errorbar(index+offset+dx,value,yerr=[[value-low],[high-value]],fmt=style,
                            color=color,capsize=3)
                points.append(dict(arm=arm,reference=ref,metric=metric,ratio=result))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],color='#0072b2',marker='o',ls='',label='ITL p99'),
        Line2D([],[],color='#d55e00',marker='o',ls='',label='TPOT p95'),
        Line2D([],[],color='black',marker='o',ls='',label='NativeStrong reference'),
        Line2D([],[],color='black',marker='s',ls='',label='Exact common-policy Native')],fontsize=8,ncol=2)
    ax.axhline(1.05,color='red',ls='--',label='5% limit')
    ax.axhline(1,color='gray',ls=':')
    ax.set(xticks=range(3),xticklabels=['LayerReg','TokenReg','HybridReg'],
           ylabel='Arm / Native ratio, paired95% CI',
           title='Decode qualification requires both metrics and both Native references')
    ax.grid(alpha=.2)
    save(fig,'registration_repair'+suffix+'_confirmed_decode_gates',caption,
         dict(evidence='MEASURED_ONLINE_ADAPTIVE_REPAIR',points=points,CI_upper_limit=1.05))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='request_registration_deferral_frozen_v1.json')
    parser.add_argument('--suffix', default='')
    arguments = parser.parse_args()
    main(arguments.plan, arguments.suffix)
