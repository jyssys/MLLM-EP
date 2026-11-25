"""Map explicit campaign requirements to final evidence and unresolved gates."""
import argparse
import json
import time

from .common import ROOT,REPORTS,digest,preserve,write
from .delivery_requirement_inventory import IDLE_REPORTS
from .resource_closeout import require_final_closeout
from .token_discovery_analysis import table


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--delivery-version',required=True)
    parser.add_argument('--cpu-log',required=True);args=parser.parse_args();preserve();require_final_closeout()
    target=ROOT/'analysis/final_requirement_audit_v0.json';assert not target.exists()
    summary=json.loads((ROOT/'analysis/final_summary_evidence_v0.json').read_text())
    inventory=json.loads((ROOT/'analysis'/f'delivery_requirement_inventory_{args.delivery_version}.json').read_text())
    assert len(inventory['reports'])==14 and all(r['available'] for r in inventory['reports'])
    assert len(inventory['figures'])==26 and all(r['artifact_available'] for r in inventory['figures'])
    cpu=ROOT/'logs'/args.cpu_log;assert cpu.read_text().rstrip().endswith('OK')
    rows=[]
    def item(key,status,relative,scope):
        path=ROOT/relative;assert path.is_file(),relative
        rows.append(dict(requirement=key,status=status,evidence=relative,SHA256=digest(path),scope=scope))
    item('Prior Dense AR evidence and original checkout/index','VERIFIED_PRESERVED','state/preservation.json',
        'common.preserve verifies the recorded checkout/index and frozen source/report hashes. No checkout or index reset.')
    item('Latest available Notion decisions and scope','AUDITED_WITH_ASOF_LIMIT','../mllm_epd_campaign_contract_and_deviations_v1.md',
        'Main100 amendment applied. Returned Notion as-of is explicit. No claim that all subsequent edits or the full archive were read.')
    item('Primary papers, official code and installed EPD support','AUDITED','../mllm_epd_related_work_and_novelty_audit.md',
        'Stage-level EPD, same-request split, deflection and deferred visual prefill are distinguished. No first-method claim.')
    item('Additional primary paper and withdrawal review','AUDITED','../mllm_epd_related_work_final_supplement_v1.md',
        'Different model/hardware/runtime results are not direct reproductions.')
    item('Actual GPU UUID, topology and ownership','RECORDED','state/environment_initial.json',
        'Definitive comparisons use the same GPU4/5/6. Eight-GPU availability does not require four concurrent experiments.')
    item('Original model/processor/media/prompt provenance and frozen400','VERIFIED_BYTES','analysis/dataset_integrity_audit_v0.json',
        '432 Main400 plus calibration records, original media, pinned model/source files. No video or prompt/page manipulation.')
    item('User-authorized Main100 downsampling','VERIFIED_UNCHANGED_SUBSET','analysis/main100_result_figures_v0.json',
        'Observed natural length informed distribution matching. Tail and discovery/heldout source imbalance are disclosed.')
    item('Natural400 and Main100 measured length statistics','MEASURED_SINGLE_GPU_NOT_ONLINE_GAIN','analysis/dataset_profile.json',
        'Cap/EOS and natural answer lengths are actual. C3 outputs remain separate per run.')
    item('Original online encoder and embedding/KV handoff','EXECUTED_WITH_CONDITIONAL_CORRECTNESS','../mllm_epd_runtime_correctness_report.md',
        'Actual E/P/D process execution, not cached embeddings presented as end-to-end serving. Full quality remains HOLD.')
    item('Agg DP3 and E+colocatedPD DP2 continuous baselines','MEASURED_ONLINE_RUNTIME_QUALITY_HOLD',
        'analysis/main100_baseline_confirmation_C14_v0.json','Same three-GPU budget and trace, compiled continuous scheduling and chunked prefill. N3 at QPS2/4.')
    item('Native EPD motivation QPS sweep and runtime trade-off','MEASURED_ONLINE_RUNTIME_QUALITY_HOLD',
        'analysis/main100_baseline_confirmation_C14_v0.json','Natural and replay separate. Finite cohort is not sustainable capacity or numerical DuetServe reproduction.')
    item('Same actual arrival/order and correct TTFT origin','RECALCULATED',
        'analysis/token_confirmation_C37_audit_v0.json','Scheduled arrival includes admission delay. Relative arrival contracts are matched. Independent starts, not requests, define N3.')
    item('All latency/throughput/SLO and eight failure categories','RECALCULATED',
        'analysis/final_summary_evidence_v0.json','Input+output tokens counted once. Request/token denominators, single-output TPOT, all-ITL AND and drain period are explicit.')
    item('High-load E-not-bottleneck/P-saturated/service-proximity condition','NOT_CONFIRMED',
        '../mllm_epd_prefill_bottleneck_and_decode_idle.md','High-load observer failures stay DIAGNOSTIC. Independent service proximity and simultaneous required bottleneck gates remain unmeasured.')
    item('SM Active/Tensor Core/HBM hardware counters','NOT_MEASURED',
        'analysis/gpu_counter_capability_audit_v0.json','Read-only capability facts only. No privileged counter access or driver mutation. Kernel busy is not these counters.')
    item('Qualified physical E/P/D kernel activity and queues','TRACE_ANALYSIS_LOW_LOAD_ONLY',
        'analysis/low_load_C25_evidence_v0.json','N1 CPC passed. Queue statistics are observation weighted. Not a qualified high-load utilization curve.')
    item('Idle duration CDF and count/duration threshold coverage','TRACE_ANALYSIS_LOW_LOAD_ONLY',
        'analysis/idle_addendum_final_decision_v0.json','Thresholds0.1/1/5/10/20/50/100ms and full/central denominators are preserved. Physical gaps are not safe windows.')
    item('Five causal encoder idle states and idle-ending causes','NOT_FULLY_IDENTIFIED',
        '../mllm_epd_encoder_decoder_idle_final_summary.md','Only kernel/pending-HTTP observations are known. Unknown states were not relabeled actionable.')
    item('Simultaneous E/D idle during P GPU activity','TRACE_ANALYSIS_LOW_LOAD_ONLY',
        'analysis/final_summary_full_vs_central_idle_v0.csv','Full and central denominators differ. P busy is not P queue presence or P saturation.')
    item('Generic exact multimodal prefix/suffix continuation','CORRECTNESS_PASS_ON_TESTED_GEOMETRY',
        '../mllm_epd_generic_token_split_correctness.md','Text/image boundaries, inside image spans, multiple images/pages, full28 KV and MRoPE were inspected. No approximate KV or omitted prompt tokens.')
    item('Frozen10% same-geometry N1/N8 and rejected25%','SCOPED_PASS_AND_PRESERVED_FAILURE',
        '../mllm_epd_encoder_batch_numerical_root_cause_v1.md','25% N8 KV tolerance failure is not rescued by matching answer tokens. No strict cross-batch invariance claim.')
    item('Original Vision batch-only controlled intervention','CONTROLLED_NUMERICAL_DIAGNOSTIC',
        'analysis/vision_BATCH_C38R1_comparison.json','Fixed target pixels/grid and LLM chunks. Different encoder envelope changes embeddings/KV. Not a vision optimization or benign-error proof.')
    item('Full official task quality and2pp non-inferiority','HOLD',
        'analysis/task_quality_C37_audit_v0.json','Zero paid external calls. Official MMLong extraction unavailable without new authority. Mantis main quality interval fails the required margin.')
    item('Strong static/idle-aware token C3 Natural and replay N3','MEASURED_ONLINE_NO_GO',
        'analysis/token_confirmation_C37_audit_v0.json','Discovery-frozen fractions, Main100 and context-heldout50, current-state conservative admission. Not success in an already-verified P-heavy regime.')
    item('NativeStrong and Common-policy controls','MEASURED_ONLINE',
        'analysis/token_confirmation_C37_audit_v0.json','Stock compiled baseline retained. Compiler/transport policy differences are explicit. Same B/S does not imply same GPU batch.')
    item('Frozen absolute and native-relative decode protection','SCOPED_STATIC10_PASS_NOT_SERVING_GAIN',
        'analysis/final_summary_static10_decode_scope_v0.csv','Both relative CI gates and actual per-request all-ITL AND TPOT are checked. Idle policy does not pass every required gate.')
    item('Whole-request deflection optional strong control','MEASURED_ONLINE_NO_GO',
        '../mllm_epd_whole_deflection_online_serving_v1.md','Separate fresh contexts and N3. Not Kairos reproduction or same-request token split.')
    item('Request-level P saving/D admission root cause','TRACE_ANALYSIS_AND_DISTINCT_POLICY_INTERVENTION',
        '../mllm_epd_generic_token_split_online_serving.md','Disjoint host means are not exclusive GPU times or additive p95. Zero-queue oracle is not implemented speedup.')
    item('Native E descriptor one-factor intervention','MEASURED_ONLINE_CONTROLLED_N3',
        'analysis/ec_INLINE_CAUSAL_C44_audit_v0.json','Only descriptor policy toggled. Two periods averaged within each independent start. TTFT improvement fails relative decode gate and does not repair already-inline Token.')
    item('Actual suffix128/256/512/1024/2048 with imported KV','CORRECTNESS_PASS_COST_DIAGNOSTIC',
        'analysis/suffix_QUANTA_C41R3_report_audit_v0.json','15 geometry correctness cases and query conservation. Full observer gate failed. Enclosure is not exclusive kernel or DMA time.')
    item('E-side LLM co-residency and memory','MEASURED_MEMORY_LIMITED_SCOPE',
        '../mllm_epd_dual_side_prefill_headroom_report.md','Vision/full-model/helper KV, device-free and process allocator are separate. Encoder max batch/OOM boundary and matched E/D suffix capacities are not measured.')
    item('Online E/D additional-prefill probes and original SLO','MEASURED_INTERFERENCE_PROBE_PROTECTION_FAILED',
        'analysis/interference_PROBE_C40R4_audit_v0.json','Positive real probe work, N3, no actual request completion contribution. All eight complete gates fail. No-probe failures and fixed order limit unique attribution.')
    item('Actual representative prepared batch/encoder output geometry','COUNTER_CPC_SCOPED_BY_METHOD_GATE',
        'analysis/batch_COUNTER_C46_audit_v0.json','No CUDA events/copies. V2 API failure preserved, four closed methods reused, two fresh repairs. Old clean N3 geometry remains unobserved.')
    item('Future safe-budget prediction and active-decode insertion','NOT_RUN_NO_VALIDATED_SAFETY',
        'analysis/final_summary_evidence_v0.json','No future EOS/arrival oracle in online decisions. No dangerous insertion merely to fill physical kernel gaps.')
    item('Three-way split/pipeline/wavefront/context parallelism','STRUCTURAL_FEASIBILITY_ONLY_NOT_RUN',
        '../mllm_epd_dual_side_parallelism_feasibility_v1.md','Dependencies, communication and memory compared. No measured3GPU speedup or idle-use novelty claim.')
    item('Original source/large tensor/raw evidence integrity','VERIFIED_WITH_HISTORICAL_MISSING_RECORDS_DISCLOSED',
        'analysis/final_summary_evidence_v0.json','Original tensor SHA, gzip round trips and boot ZIP/member hashes checked. Missing historical ZIPs are not recreated from current source.')
    item('CPU regression and observer repair tests','PASS',f'logs/{args.cpu_log}',
        'Actual full suite plus preserved red/green metric-scope, V2 hook and fresh-folder dependency tests. No assertion removed from the serving model.')
    item('Required nine plus five reports and26 figure items','ARTIFACTS_VERIFIED_SEPARATE_FROM_SCIENCE',
        f'analysis/delivery_requirement_inventory_{args.delivery_version}.json','14named reports and26requirements mapped to CSV/PNG/PDF, with reused stems disclosed. Diagnostics and oracle labels remain explicit.')
    item('Required figure execution and CSV/PNG reproduction','VERIFIED_LIMITED_PDF_SCOPE',
        'analysis/final_summary_evidence_v0.json','Fresh guarded outputs, exact CSV after path normalization and PNG pixels. PDF header/EOF only, not full PDF pixel rendering.')
    item('GPU/owned process closeout and all eight burners','VERIFIED_OWNERSHIP_AND_UTILIZATION',
        'state/final_resource_closeout_v0.json','Controller absence, closed leases, approved script hashes, CUDA contexts and measured utilization. Unknown processes remain untouched.')
    table(ROOT/'analysis/final_requirement_audit_v0.csv',rows)
    write(target,dict(time=time.time(),requirements=rows,report_files=list(REPORTS+IDLE_REPORTS),
        report_count=14,figure_requirement_count=26,delivery_checks_complete=True,
        scientific_decisions=summary['decisions'],all_scientific_target_gates_passed=False,
        remote_publication='PENDING_ACTUAL_FINAL_PUSH_AND_REMOTE_SHA_VERIFICATION',
        scope='Requirement-by-requirement evidence map. Failed, unmeasured and safety-gated stages remain explicit. '
            'Delivery verification does not turn scientific HOLD/NO_GO into GO, model quality PASS or validated safe headroom.'))
    preserve();print('FINAL_REQUIREMENTS_EVIDENCE_MAPPED',len(rows),flush=True)


if __name__=='__main__':main()
