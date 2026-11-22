"""Freeze strong static choices before new held-out serving measurements."""
import argparse
import json
import time
from .common import ROOT,write,digest
from .recovery_analysis import main as refresh


def policy(cfg):
    return (bool(cfg.get('d_compute_throttle')),bool(cfg.get('defer_D_remote_setup')),
        bool(cfg.get('d_quiescent_admission')),cfg.get('D_running_admission_cap'),
        bool(cfg.get('D_prefill_completion_guard')))


def common_control(qps,key):
    if key[-1]:
        name=f'CompletionAB_Discovery200_q{qps}_Native_Guard1_v1' if key==(True,True,False,None,True) else None
        migrated=ROOT/'state/completion_guard_resource_migration_v2.json'
        if name and migrated.exists():
            name=json.loads(migrated.read_text())['old_to_new'].get(name,name)
        return name
    key=key[:-1]
    if key==(False,False,False,None):
        return f'RemoteSetupAB_Discovery200_q{qps}_NativeStrong_v1'
    if key==(True,False,False,None):
        return f'NativeCompute_Discovery200_q{qps}_v1'
    if key==(True,True,False,None):
        return f'RemoteSetupAB_Discovery200_q{qps}_NativeSetup_v1'
    if key in ((True,True,False,1),(True,True,False,2)):
        return f'DcapTune_Discovery200_q{qps}_Native_Cap{key[-1]}_v1'
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    refresh()
    rows=json.loads((ROOT/'analysis/decode_tail_recovery_discovery_all_v1.json').read_text())['rows']
    selected={}
    all_candidates=[]
    for qps in (1.8,2.8):
        reference=common_control(qps,(False,False,False,None,False))
        base=json.loads((ROOT/'analysis'/(reference+'_metrics.json')).read_text())
        candidates=[]
        for row in rows:
            if row['status']!='MEASURED_ONLINE' or row['QPS']!=qps:
                continue
            name=row['name']
            cfg=json.loads((ROOT/'raw'/name/'configuration.json').read_text())
            axis=cfg.get('axis_kind','Native')
            family='WholeDeflection' if cfg.get('whole_prefill_deflection') else axis
            if family=='Native':
                family='NativeTuned'
            if family not in ('Layer','Token','Hybrid','WholeDeflection','NativeTuned'):
                continue
            other=common_control(qps,policy(cfg))
            m=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
            reasons=[]
            if cfg.get('credit_serialization_control'):
                reasons.append('CREDIT1_SERIALIZATION_CONTROL_NOT_DEFAULT_SERVING')
            if row['task_quality_passed'] is not True:
                reasons.append('TASK_QUALITY_NOT_PASS')
            if other is None or not (ROOT/'analysis'/(other+'_metrics.json')).exists():
                reasons.append('EXACT_COMMON_POLICY_NATIVE_NOT_MEASURED')
                common=None
            else:
                common=json.loads((ROOT/'analysis'/(other+'_metrics.json')).read_text())
            ratios={}
            for label,native in (('NativeStrong',base),('NativeCommon',common)):
                if native is None:
                    continue
                ratios[label]={field:m[field]/native[field] for field in ('ITL_p99_ms','TPOT_p95_ms')}
                if any(v>1.05 for v in ratios[label].values()):
                    reasons.append(label+'_DECODE_POINT_NOT_PASS')
            if m['decode_SLO_attainment']<.95:
                reasons.append('ABSOLUTE_DECODE_NOT_PASS')
            candidate=dict(name=name,family=family,QPS=qps,policy=list(policy(cfg)),
                NativeStrong=reference,NativeCommon=other,eligible=not reasons,reasons=reasons,
                decode_point_ratios=ratios,TTFT_p95_ms=m['TTFT_p95_ms'],
                SLO_goodput=m['SLO_qualified_output_tokens_per_second'])
            candidates.append(candidate)
        chosen={}
        for family in ('Layer','Token','Hybrid','WholeDeflection','NativeTuned'):
            qualified=[r for r in candidates if r['family']==family and r['eligible']]
            chosen[family]=min(qualified,key=lambda r:(r['TTFT_p95_ms'],-r['SLO_goodput'],r['name'])) if qualified else None
        selected[str(qps)]=chosen
        all_candidates.extend(candidates)
    value=dict(version=1,selection_epoch=time.time(),selected=selected,candidates=all_candidates,
        evidence='DISCOVERY_SELECTION_ONLY',
        contract_SHA256=digest(ROOT/'configs/final_continuous_confirmation_contract_v1.json'),
        native_control_extension_SHA256=digest(ROOT/'configs/final_native_tuned_control_extension_v1.json'),
        optional_safety_extension_SHA256=digest(ROOT/'configs/final_optional_baseline_safety_rule_v1.json'),
        token_tail_control_extension_SHA256=digest(ROOT/'configs/final_token_tail_control_extension_v1.json'),
        scope='Select the minimum discovery TTFTp95 within each family after correctness,task quality and both decode point gates. '
            'Use a native control with the same compute/READ/running-cap policy. Keep unmodified NativeStrong. '
            'Quiescent policies without an exact common Native control remain in the ledger but are not eligible. '
            'Source admission and compile geometry remain explicit method differences. '
            'This finite candidate grid is not a global optimum. Point eligibility is not the final independent-start safety CI. '
            'Every selected new-source composition still needs a fresh full400 state/quality proof before final timing.')
    path=ROOT/('configs/final_continuous_static_choices_frozen_v1.json' if args.freeze
        else 'analysis/final_continuous_selection_preview_v1.json')
    if args.freeze:
        assert all(cell[family] is not None for cell in selected.values()
            for family in ('Layer','Token','Hybrid','NativeTuned')),selected
        assert not path.exists(),'Never change a frozen choice after observing held-out timing'
        value['unqualified_optional_families']=[dict(QPS=q,family=family,status='NOT_RUN',
            reason='No measured discovery candidate passes correctness,quality,and both unchanged decode point gates. '
                'Keep all failed candidates in the ledger. Do not force an unsafe family into qualified confirmation.')
            for q,cell in selected.items() for family in ('WholeDeflection',) if cell[family] is None]
    write(path,value)
    print({q:{a:(r['name'] if r else None) for a,r in v.items()} for q,v in selected.items()},flush=True)


if __name__=='__main__':
    main()
