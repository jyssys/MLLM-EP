"""Bind an explicit root review after final reconstruction and manual review."""
import argparse
import json
import time
from .common import ROOT,REPO,write,digest,preserve
from .publication import DOCUMENTS
from .figure_catalog import UNMEASURED_FIGURES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-reviewed',action='store_true',required=True)
    args=parser.parse_args();assert args.root_reviewed
    preserve()
    assert (ROOT/'state/STOP_ANALYSIS_WATCHER.json').exists(),'Freeze the mutable analysis writer before binding'
    assert (ROOT/'state/final_compiler_CPU_postprocess_complete_v1.json').exists()
    report=REPO/'poc_flashvep/reports/dense_ar_longbench400_layer_token_2d_final_summary.md'
    text=report.read_text()
    assert all(f'## {i}.' in text for i in range(1,13))
    assert '작성 상태: 과학적 검증과 보고서 검토를 완료했다.' in text
    questions=[
        (1,'DATASET_PASS_WITH_DECLARED_CELL_SHORTAGE','400 original requests,input100/bin,natural outputs measured. '
         'The long-input natural-output row is52/29/3/16,not25each.', 'data/balanced400_freeze_manifest.json'),
        (2,'C0_CONDITIONAL_AXIS_WIN_NOT_C3_STRONG_GO','Both-safe C0 cells:15Layer,16Token,16tie,17unsafe. '
         'Short input does not determine the winning axis alone.','analysis/axis64_both_axes_decode_safety_audit_v1.json'),
        (3,'CONTROLLED_OUTPUT_EFFECT_CONFIRMED','The same I3 requests,source,pair,arrival and cut show the '
         'declared C0 O16/O64 Hybrid-to-Token sign reversal. Do not generalize to natural C3.',
         'analysis/iso8_hybrid_heldout_confirmation_v1.json'),
        (4,'QPS_ONLY_CROSSOVER_NOT_CONFIRMED','Some C0 output/cut cells reverse. The separate high-QPS '
         'short-input CI includes1. Actual saturation and workload demand remain separate.',
         'analysis/axis_matrix_heldout_confirmation_r2.json'),
        (5,'WORKLOAD_AND_RUNTIME_DEPENDENT','Input,output,cut,queue,credit and query chunks matter. '
         'Discovery freezes C3 best-static choices before test. No universal pure-axis rule.',
         'configs/final_continuous_static_choices_frozen_v1.json'),
        (6,'GENUINE_HYBRID_CORRECTNESS_PASS_C0_SMALL_GAIN','The actual causal DAG conserves36L token-layer work. '
         'One iso8 C0 cell has a small TTFT win. Additional Token-goodput gain is uncertain.',
         'analysis/iso8_hybrid_heldout_confirmation_v1.json'),
        (7,'NO_C3_HYBRID_STRONG_TOKEN_WIN_ON_TESTED_GRID','Strong compiled Token remains. A slower Mode0 '
         'Token diagnostic is never its replacement. Keep all original and repair outcomes.',
         'analysis/final_continuous_strong_confirmation_v1.json'),
        (8,'STATIC_SERVING_GAINS_CONDITION_SPECIFIC','Agg/Native trade-off is measured. Token and deflection '
         'provide serving gains on tested finite traces. Native-relative gain is not capacity multiplication.',
         'analysis/final_continuous_strong_confirmation_v1.json'),
        (9,'DECODE_SAFETY_CONDITION_SPECIFIC','Common registration repair passes L/T/H at both natural400 loads. '
         'Fixed-output bridge has residual failures. The final compiler repair passes Hybrid but not Layer Common-Native ITL. '
         'Keep those same-condition controls and do not substitute outcomes across sources.',
         'analysis/controlled_continuous_I3_bridge_registration_results_v1.json'),
        (10,'HOLD_DECODE_SAFE_WINDOW_NOT_IDENTIFIED','Qualified native ROI confirms P backlog and lower D busy. '
         'Prepared-state executable safe-fit is not measured. No unsafe dynamic insertion.',
         'analysis/recovered_native_shared_window_activity_v1.json'),
        (11,'NO_VALIDATED_MINIMAL_ONLINE_SELECTOR','The exploratory C0 bottleneck rule improves prediction '
         'but has counterexamples and nonzero regret. C3 operator shares and SM/HBM are not measured.',
         'analysis/minimal_calibrated_demand_rules_v1.json'),
        (12,'FAILURES_AND_SCOPE_RETAINED','All strict failures,observer failures,unsafe cells,negative results '
         'and explicit NOT_RUN outcomes remain. The final kernel intervention is the last bounded repair.',
         'analysis/eager_compile_confirmation_v2_results_v1.json')]
    answers=[]
    for number,status,answer,path in questions:
        assert (ROOT/path).exists(),(number,path)
        answers.append(dict(question=number,status=status,answer=answer,evidence_path=path,SHA256=digest(ROOT/path)))
    stops=[
        dict(work='Online dynamic-size/axis/2D insertion and prepared-state safe-fit',status='NOT_RUN',
            reason='Future decode-safe executable windows and a validated online predictor were not established. '
                   'Physical kernel gaps and the bounded offline oracle do not satisfy the safety gate.',
            evidence_path='analysis/recovered_native_shared_window_activity_v1.json'),
        dict(work='Full forced decode batch2/4 Cartesian extension and further credit/cut searches',status='NOT_RUN',
            reason='The primary64 cells and strong continuous/repair comparisons are terminal. Residual native-relative '
                   'safety failures prevent a serving GO. End the bounded campaign instead of repeating parameters until a preferred winner appears.',
            evidence_path='analysis/controlled_continuous_I3_bridge_registration_results_v1.json'),
        dict(work='Qualified C3 exclusive operator shares and SM/HBM counters',status='NOT_MEASURED',
            reason='The qualified operator captures are C0. Their mode-specific hooks do not qualify the C3 compiled path. '
                   'Host enclosure and diagnostic traces do not replace exclusive kernel measurement.',
            evidence_path='analysis/operator_cost_observer_audit_v2.json'),
        dict(work='Partial-only scoped source21 serving starts',status='NOT_RUN',
            reason='Hybrid strict KV/hidden failed. The plan was superseded before its first performance start. '
                   'Every old condition remains explicitly NOT_RUN. Correctness jobs finished normally.',
            evidence_path='state/scoped_compile_confirmation_complete_v1.json'),
        dict(work='Cold-dispatch V1 planned38 GPU starts',status='NOT_RUN',
            reason='The actual immutable dispatch failed the CPU cold-start harness before any GPU launch. '
                   'A separately frozen corrected V2 contains all final proof/performance outcomes.',
            evidence_path='state/eager_v1_cold_dispatch_preflight_stop.json')]
    catalog=json.loads((ROOT/'analysis/figure_catalog_v1.json').read_text())
    reviewed=[dict(path=str((REPO/'poc_flashvep/reports'/n).relative_to(REPO)),
        SHA256=digest(REPO/'poc_flashvep/reports'/n)) for n in DOCUMENTS]
    write(ROOT/'state/scientific_closeout_review_v1.json',dict(time=time.time(),
        reviewer='Root agent, explicit final review. Not an external peer review.',
        all_questions_answered=True,unresolved_mandatory_work=[],answers=answers,
        explicit_not_run_and_stop_reasons=stops,reviewed_reports=reviewed,omitted_figures=list(UNMEASURED_FIGURES),
        figure_catalog_SHA256=digest(ROOT/'analysis/figure_catalog_v1.json'),figure_pairs=len(catalog['figures']),
        scope='Root reviewed the final claims, evidence scopes, failed gates and figures. The current invocation '
              'binds those reviewed artifacts by SHA256. A hash or checkbox does not prove scientific correctness. '
              'Operational resource restoration, archive, commit, push and remote proof remain separate. '
              'Scientific HOLD/NO-GO outcomes are not missing work and are not converted to GO.'))
    preserve();print('SCIENTIFIC_REVIEW_BOUND',len(answers),len(reviewed),len(catalog['figures']),flush=True)


if __name__=='__main__':main()
