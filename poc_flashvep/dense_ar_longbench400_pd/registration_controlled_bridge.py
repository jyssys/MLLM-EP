"""Retest every frozen bridge condition after the common host-registration fix."""
import argparse
import hashlib
import json
import time
import numpy as np
from .common import ROOT, REPO, write, digest, preserve
from .final_confirmation_jobs import clone
from .registration_deferral_campaign import registration_signature, run_plan
from .run import save, execute
from .qualification import audit
from .metrics import summarize


PLAN = 'request_registration_controlled_bridge_frozen_v1.json'
PREFIX = 'request_registration_controlled_bridge'


def build():
    preserve()
    path = ROOT / 'configs' / PLAN
    assert not path.exists(), 'Freeze this follow-up before any new measurement'
    assert (ROOT / 'state/controlled_continuous_bridge_complete_v1.json').exists()
    old = json.loads((ROOT / 'configs/controlled_continuous_I3_bridge_frozen_v1.json').read_text())
    origin = json.loads((ROOT / 'configs/request_registration_deferral_frozen_v1.json').read_text())
    later = json.loads((ROOT / 'configs/request_registration_q28_extension_frozen_v1.json').read_text())
    source, sha = origin['source_capsule'], origin['source_capsule_SHA256']
    assert later['source_capsule_SHA256'] == sha
    candidates = list(origin['qualification'].values()) + list(later['qualification'].values())
    proofs, cells, jobs = {}, [], []
    for block in old['cells']:
        arms = {}
        for label, original_name in block['jobs'].items():
            base = json.loads((ROOT / 'configs' / (original_name + '.json')).read_text())
            base['defer_D_request_registration'] = label != 'NativeStrong'
            base.pop('bridge_composition_gate', None)
            gate = None
            if base['defer_D_request_registration']:
                signature = registration_signature(base)
                key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
                if key not in proofs:
                    matches = [p for p in candidates if p['signature'] == signature]
                    if matches:
                        proof = dict(matches[0], reused=True)
                    else:
                        names = []
                        gate = f'RegistrationBridge_{key}_gate_v1.json'
                        for count, dataset in ((32, 'continuous_qualification32_natural.json'),
                                               (400, 'longbench_balanced400_v1.json')):
                            name = f'RegistrationBridgeQ_{label}_{key}_Natural{count}_v1'
                            cfg = clone(base, name, dataset, 2.8, 2026112201, (6,7), source, sha, correctness=True)
                            cfg.pop('registration_deferral_gate', None)
                            save(cfg)
                            names.append(name)
                        proof = dict(jobs=names, gate=gate, signature=signature, reused=False)
                    proofs[key] = proof
                gate = proofs[key]['gate']
            name = f'C3RegistrationBridge_I3_O{block["output"]}_Cap{block["D_cap"]}_s{block["start"]}_{label}_v1'
            cfg = clone(base, name, block['dataset'], block['QPS'], block['seed'], (4,5), source, sha)
            cfg.update(output_cap=block['output'], ignore_eos=True,
                registration_bridge_original=original_name,
                scope='Adaptive common-runtime repair of all three original C3 bridge conditions. '
                      'Same4/5,prompt IDs,arrival,cut,output and strong Token. Original failures remain. '
                      'Only the new registration policy changes runtime semantics. Admission and batch feedback can change. '
                      'This is not an exclusive CPU-cost estimate or a new-request test.')
            for field in ('suffix_by_request','aggregate_work_audit'):
                if field in base:
                    cfg[field] = base[field]
            if gate:
                cfg['registration_deferral_gate'] = gate
            for field in ('dataset_SHA256','arrival_offsets','seed','gpus','output_cap','ignore_eos',
                          'axis_kind','axis_cut','execution_family','continuous_source_credit_limit',
                          'D_running_admission_cap','suffix_by_request'):
                assert cfg.get(field) == base.get(field), (name, field)
            save(cfg)
            arms[label] = name
        ordered = list(arms.values())
        np.random.default_rng(block['seed'] + block['D_cap']*100 + block['output']).shuffle(ordered)
        jobs.extend(ordered)
        cells.append(dict(block, jobs=arms, stage='ADAPTIVE_FIXED_CONDITION_REPAIR'))
    write(path, dict(time=time.time(), source_capsule=source, source_capsule_SHA256=sha,
        original_bridge_plan_SHA256=digest(ROOT / 'configs/controlled_continuous_I3_bridge_frozen_v1.json'),
        original_source_capsule_SHA256=old['source_capsule_SHA256'],
        qualification=proofs, cells=cells, performance_jobs=jobs,
        trigger='All original O64 C3 offload arms fail a Native-relative gate. '
                'The common registration fix must be tested on this exact controlled workload before dismissing its C0 candidate.',
        hypotheses=['H1: Registration/admission feedback causes the residual bridge tail; the common fix reduces it.',
                    'H2: Strong Token forward/packing remains better; Hybrid ranking persists after the tail repair.',
                    'H3: Rare client-delivery delay is independent; removing registration does not remove every large interval.'],
        scope='All54 original bridge conditions/arms/starts retained. No new cut,output,QPS,SLO or favorable seed. '
              'Proofs use6/7 after READ+Layer drain. Performance uses original4/5 after q2.8 registration confirmation drains. '
              'Exact-source/signature proofs may be reused only after they exist. Same-source disabled NativeStrong stays unchanged. '
              'Old versus new comparison is adaptive and separated in time. Unchanged NativeStrong is also a drift control. '
              'No paired C0-to-C3 causal claim,no active-decode insertion,no dynamic selector.'))
    preserve()
    print('REGISTRATION_BRIDGE_FROZEN',len(jobs),'serving starts',
          sum(not p['reused'] for p in proofs.values())*2,'new proof starts',flush=True)


def run():
    plan = json.loads((ROOT / 'configs' / PLAN).read_text())
    while not (ROOT / 'state/hybrid_READ_repair_layer_control_complete_v1.json').exists():
        time.sleep(20)
    for proof in plan['qualification'].values():
        if proof['reused']:
            while not (ROOT / 'state' / proof['gate']).exists():
                time.sleep(20)
    run_plan(dict(plan, performance_jobs=[]), PREFIX+'_qualification')
    while not (ROOT / 'state/request_registration_q28_extension_complete_v1.json').exists():
        time.sleep(20)
    results=[]
    for name in plan['performance_jobs']:
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        gate=cfg.get('registration_deferral_gate')
        if gate and not json.loads((ROOT/'state'/gate).read_text())['passed']:
            result=dict(name=name,status='NOT_RUN',reason='BLOCKED_CORRECTNESS',gate=gate)
        elif execute(name):
            if cfg.get('axis_kind'):
                state=audit(name)
                assert state['passed'] and state['decode_full36_observation_complete']
            metric=summarize(name)
            assert metric['output_tokens']==cfg['count']*cfg['output_cap']
            result=dict(name=name,status='MEASURED_ONLINE',
                **{f:metric[f] for f in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms')})
        else:
            result=dict(name=name,status='FAILED')
        results.append(result)
        write(ROOT/'state'/(PREFIX+'_progress_v1.json'),dict(results=results))
        print('REGISTRATION_BRIDGE_SERVING',result,flush=True)
    write(ROOT/'state'/(PREFIX+'_complete_v1.json'),dict(complete=True,results=results,
        scope='Operational terminal outcomes. Not a scientific GO. Original failures remain unchanged.'))
    analyze()


def analyze():
    from .controlled_bridge_analysis import main as analysis
    from .controlled_bridge_figures import main as figures
    analysis(PLAN,'_registration')
    data=json.loads((ROOT/'analysis/controlled_continuous_I3_bridge_registration_results_v1.json').read_text())
    report(data)
    if data['complete']:
        figures('_registration','registration_controlled_C3_I3',
            '\nADAPTIVE COMMON-REGISTRATION REPAIR | Source071. Original54 failures remain. Same4/5 and old arrival seeds,not a blind retest.')


def report(data):
    from .confirmation_analysis import ratio
    from .final_report_tables import interval
    from .final_service_decomposition import pair
    from .service_accounting import compare
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    lines=['# LongBench400: 등록 수정 후의 Controlled C3 Bridge','',
        '이 검사는 원래 54개 Bridge 이후에 선택한 Adaptive Follow-up이다.',
        '세 조건과 여섯 Arm을 모두 유지했다. 유리한 조건만 다시 실행하지 않았다.',
        '상태: '+('`MEASURED_ONLINE_CONTROLLED_REPAIR`.' if data['complete'] else '`PARTIAL_OR_INVALID`.'),
        '정합성, 상대 Decode 보호, Absolute SLO와 TTFT 우위는 별도 Gate다.','',
        '## 동결한 계약','',
        '원래 I3 Prompt16개, QPS1.253873과 Arrival Seed2026112551/2552/2553을 유지한다.',
        '성능 GPU Pair는 원래 Bridge와 같은4/5다. 새 Proof는6/7을 사용한다.',
        '조건은 Output16/Cap1, Output64/Cap1, Output64/Cap2다.',
        '각 조건은 NativeStrong, NativeCommon, Layer8, StockToken8, Hybrid4+정확한 L/8, Mode0Token Control을 포함한다.',
        '각 Arm은 독립 Server Start3회다. 전체 성능 Start는54회다.',
        'NativeStrong은 원래 Scheduler를 유지한다. 표의 Cap은 Common-policy Arm에만 적용한다.',
        'Source Credit은 Offload Arm8, Native0이다. 실제 Batch는 강제하지 않는다.',
        'Layer/Token/Hybrid의 Logical P/D Rows는 각각12676608/3621888이다.',
        'Actual Padding과 Batch는 JSON에 별도 기록한다.',
        'Model은 Qwen3-8B, Revision `b968826d9c46dd6066d109eabc6255188de91218`이다.',
        'BI0, Original BF16, Exact Causal Attention, Full36 Decode와 B8192/S1024를 유지한다.',
        'KV는42GiB/Block128이다. Prefix Cache를 사용하지 않는다.',
        'Output은 `ignore_eos=True`다. 이 결과는 Natural EOS 분포가 아니다.',
        'Source SHA256은 `'+plan['source_capsule_SHA256']+'`다.',
        'Dataset SHA256은 `6429dde7a7dd44156636e7f9c8f041d7b7d0b6f91c80bc2cce1697dc51e8c780`이다.','',
        '## 결과','',
        '| Output / Common Cap | Arm | Starts | TTFT p95(ms) | ITL p99(ms) | TPOT p95(ms/token) | SLO-goodput(output tokens/s) |',
        '|---|---|---:|---:|---:|---:|---:|']
    fields=('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')
    for cell in data['cells']:
        for arm,item in cell['arms'].items():
            values=' | '.join(f'{item["means"][f]:.6f}' for f in fields) if item['starts'] else 'N/A | N/A | N/A | N/A'
            lines.append(f'| {cell["controlled_output"]}/{cell["D_cap"]} | {arm} | {item["starts"]} | '+values+' |')
    lines += ['', '## Native 기준과 강한 Token 기준','',
        '| Output / Cap | Arm / Reference | TTFT p95 비율 [95% CI] | ITL p99 비율 [95% CI] | TPOT p95 비율 [95% CI] | Goodput 비율 [95% CI] | Decode 5% | Absolute Decode≥95% |',
        '|---|---|---|---|---|---|---|---|']
    for cell in data['cells']:
        for arm,refs in cell['comparisons'].items():
            for ref,test in refs.items():
                lines.append(f'| {cell["controlled_output"]}/{cell["D_cap"]} | {arm}/{ref} | '+
                    ' | '.join(interval(test['ratios'][f]) for f in fields)+
                    f' | {test["decode_5percent_CI_pass"]} | {test["absolute_decode_pass"]} |')
    effects=[];within=[]
    if data['complete']:
        for cap,output in ((1,16),(1,64),(2,64)):
            blocks=[c for c in plan['cells'] if c['D_cap']==cap and c['output']==output]
            for arm in blocks[0]['jobs']:
                old=[];new=[];boundaries=[]
                for block in blocks:
                    name=block['jobs'][arm]
                    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                    parent=cfg['registration_bridge_original']
                    old.append(json.loads((ROOT/'analysis'/(parent+'_metrics.json')).read_text()))
                    new.append(json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text()))
                    boundaries.append(compare(name,parent))
                effects.append(dict(output=output,cap=cap,arm=arm,
                    new_over_old={f:ratio([r[f] for r in new],[r[f] for r in old]) for f in fields},
                    same_ID_host_boundary_changes=boundaries))
            for arm in ('Layer','Hybrid','TokenMode0Control'):
                within.append(dict(output=output,cap=cap,arm=arm,reference='StockToken',
                    rows=[pair(c['jobs'][arm],c['jobs']['StockToken']) for c in blocks]))
        write(ROOT/'analysis/registration_controlled_bridge_policy_effects_v1.json',
            dict(effects=effects,within_source_comparisons=within,evidence='TRACE_ANALYSIS_ADAPTIVE_COMMON_RUNTIME_REPAIR',
                 scope='Same original4/5,prompt IDs,arrival,cut and output. Old and repaired starts are separate time blocks. '
                       'Use unchanged NativeStrong as a drift witness. Registration also changes subsequent admission and batching. '
                       'Host boundary differences are not GPU-exclusive cost or a decomposition of TTFTp95.'))
        lines += ['', '## 원래 Source와 수정 후 Source','',
            '| Output / Cap | Arm | New/Old TTFT p95 [95% CI] | New/Old ITL p99 [95% CI] | New/Old TPOT p95 [95% CI] | New/Old Goodput [95% CI] |',
            '|---|---|---|---|---|---|']
        for effect in effects:
            lines.append(f'| {effect["output"]}/{effect["cap"]} | {effect["arm"]} | '+
                ' | '.join(interval(effect['new_over_old'][f]) for f in fields)+' |')
    lines += ['', '## 해석 제한','',
        '원래 실패와 새 수정 결과를 모두 보존한다.',
        '같은 Request, Arrival, Cut과 Output을 사용한다. 하지만 두 Source는 서로 다른 Time Block에서 실행한다.',
        '변경하지 않은 NativeStrong도 Drift Control로 표시한다. 이를 완전한 동시 A/B로 표현하지 않는다.',
        '등록 정책은 이후 Admission과 Batch도 바꾼다. Python 비용만의 인과 효과를 분리한 것은 아니다.',
        'Primary Metric은 실제 Coordinator Client-visible Timestamp를 사용한다. HTTP Latency가 아니다.',
        '큰 Client Interval이나 실패한 Gate를 제거하지 않는다.',
        'CI는 Paired Server Start 단위의 Pointwise Empirical 95% CI다. N3의27개 Bootstrap 조합을 열거한다.',
        '다중 비교 보정은 하지 않았다. Request를 독립 Start로 사용하지 않는다.',
        'Natural32/400 Proof와 고정 Output 성능을 구분한다. 같은 Source와 Signature의 Proof만 재사용한다.',
        '이후 Decode Residency는 TTFT에 합산하지 않는다. Host 경계는 GPU-exclusive 시간이 아니다.',
        'Sustainable Capacity, 새로운 Request 일반화와 Online Dynamic 이득은 이 검사로 주장하지 않는다.','',
        '- [동결한 설계](dense_ar_longbench400_2d_20261121/configs/'+PLAN+')',
        '- [전체 Metric와 Gate](dense_ar_longbench400_2d_20261121/analysis/controlled_continuous_I3_bridge_registration_results_v1.json)',
        '- [원래 54개 결과](dense_ar_longbench400_controlled_continuous_bridge.md)','']
    (REPO/'poc_flashvep/reports/dense_ar_longbench400_registration_controlled_bridge.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('build','run','analyze'))
    args=parser.parse_args()
    dict(build=build,run=run,analyze=analyze)[args.action]()
