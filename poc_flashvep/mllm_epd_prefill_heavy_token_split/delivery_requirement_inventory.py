"""Audit requested artifact availability without treating files as scientific PASS."""
import argparse
import csv
import json
import time

from PIL import Image

from .common import ROOT,REPORTS,digest,preserve,write
from .token_discovery_analysis import table

IDLE_REPORTS=('mllm_epd_encoder_decoder_idle_poc_spec.md','mllm_epd_encoder_decoder_idle_profile.md',
    'mllm_epd_simultaneous_idle_analysis.md','mllm_epd_dual_side_prefill_headroom_report.md',
    'mllm_epd_encoder_decoder_idle_final_summary.md')

# A CSV path can be the original window table rather than a duplicate figure table.
FIGURES=[
    ('main_01','Actual input distribution',['main100_actual_input_distribution_v0'],'MEASURED_PROCESSOR_METADATA',None),
    ('main_02','Natural output distribution',['main100_actual_natural_output_v0'],'MEASURED_SINGLE_GPU_NATURAL',None),
    ('main_03','Input/output scatter',['main100_actual_input_output_v0'],'MEASURED_PROCESSOR_AND_SINGLE_GPU',None),
    ('main_04','TTFT vs QPS',['main100_C14_natural_TTFT_p50_ms_v0','main100_C14_natural_TTFT_p95_ms_v0'],'MEASURED_ONLINE',None),
    ('main_05','TBT/ITL/TPOT vs QPS',['main100_C14_natural_ITL_mean_ms_v0','main100_C14_natural_ITL_p99_ms_v0',
        'main100_C14_natural_TPOT_p95_ms_v0'],'MEASURED_ONLINE',None),
    ('main_06','Input+output throughput',['main100_C14_natural_total_input_output_tokens_per_second_v0'],'MEASURED_ONLINE',None),
    ('main_07','E/P/D busy vs QPS',['idle_scope_busy_vs_QPS_v1'],'QUALIFIED_LOW_LOAD_AND_DIAGNOSTIC_HIGH_LOAD',None),
    ('main_08','E/P/D queue timeline',['idle_scope_C25_aligned_queue_timeline_v1'],'TRACE_ANALYSIS_LOW_LOAD_OBSERVATIONS',None),
    ('main_09','D physical idle CDF',['ec_LOW_CUPTI_C25_CPC_q0.5_v0_E_D_idle_CDF'],'TRACE_ANALYSIS_LOW_LOAD',
        'analysis/ec_LOW_CUPTI_C25_CPC_q0.5_v0_idle_windows.csv'),
    ('main_10','D physical gaps vs imported-KV suffix cost',['suffix_QUANTA_C41R3_physical_gap_fit_oracle_v0'],'OFFLINE_ORACLE_NOT_SAFE_ADMISSION',None),
    ('main_11','Native/Token TTFT CDF',['token_confirmation_C37_Main100_TTFT_CDF_v0'],'MEASURED_ONLINE_QUALITY_PENDING',None),
    ('main_12','Native/Token throughput and goodput',['token_confirmation_C37_Main100_raw_output_tokens_per_second_v0',
        'token_confirmation_C37_Main100_SLO_goodput_v0'],'MEASURED_ONLINE_QUALITY_PENDING',None),
    ('main_13','Joint SLO failure categories',['main100_C37_joint_failure_breakdown_v0'],'TRACE_ANALYSIS_MEASURED_ONLINE',None),
    ('main_14','E/P/D aligned GPU timeline',['idle_scope_C25_aligned_GPU_timeline_v1'],'TRACE_ANALYSIS_LOW_LOAD',None),
    ('main_15','Split fraction vs online performance',['token_discovery_C34_natural_v0','token_discovery_C34_replay_v0'],'MEASURED_ONLINE_DISCOVERY_N1',None),
    ('main_16','Visual ratio vs P latency',['main100_C37_visual_ratio_P_HTTP_v0'],'DIAGNOSTIC_HTTP_NOT_EXCLUSIVE_GPU',None),
    ('idle_01','E/P/D busy vs QPS',['idle_scope_busy_vs_QPS_v1'],'QUALIFIED_LOW_LOAD_AND_DIAGNOSTIC_HIGH_LOAD',None),
    ('idle_02','E/P/D waiting vs QPS',['idle_scope_queue_vs_QPS_v1'],'SCHEDULER_OBSERVATION_MEAN_NOT_TIME_WEIGHTED',None),
    ('idle_03','E/D physical gap CDF',['ec_LOW_CUPTI_C25_CPC_q0.5_v0_E_D_idle_CDF'],'TRACE_ANALYSIS_LOW_LOAD',
        'analysis/ec_LOW_CUPTI_C25_CPC_q0.5_v0_idle_windows.csv'),
    ('idle_04','E/D threshold coverage',['low_load_C25_threshold_coverage_v0'],'TRACE_ANALYSIS_LOW_LOAD',None),
    ('idle_05','Simultaneous idle vs QPS',['idle_scope_simultaneous_vs_QPS_v1'],'PHYSICAL_OVERLAP_NOT_ACTIONABLE_SLACK',None),
    ('idle_06','Aligned GPU activity',['idle_scope_C25_aligned_GPU_timeline_v1'],'TRACE_ANALYSIS_LOW_LOAD',None),
    ('idle_07','Window vs actual quantum costs',['suffix_QUANTA_C41R3_cost_v0',
        'suffix_QUANTA_C41R3_physical_gap_fit_oracle_v0'],'ISOLATED_OR_DIAGNOSTIC_COST_AND_OFFLINE_FIT',None),
    ('idle_08','E memory before/after helper',['interference_PROBE_C40R4_memory_v0'],'ACTUAL_PROCESS_AND_SHARED_DEVICE_SNAPSHOTS',None),
    ('idle_09','Online interference probe latency',['interference_PROBE_C40R4_E_HTTP_p95_ms_v0',
        'interference_PROBE_C40R4_ITL_p99_ms_v0','interference_PROBE_C40R4_TTFT_p95_ms_v0'],
        'INTERFERENCE_PROBE_NOT_TASK_COMPLETION',None),
    ('idle_10','Future parallelism feasibility',['parallelism_feasibility_structural_v0'],'STRUCTURAL_ACCOUNTING_NOT_IMPLEMENTED_SPEEDUP',None),
]


def figure_bundle(root,stem,csv_path=None):
    paths=[root/'figures'/f'{stem}.png',root/'figures'/f'{stem}.pdf',
        root/csv_path if csv_path else root/'figures'/f'{stem}.csv']
    missing=[str(path.relative_to(root)) for path in paths if not path.is_file()]
    result=dict(stem=stem,available=not missing,missing=missing,files=[])
    if missing:return result
    with Image.open(paths[0]) as picture:
        assert picture.format=='PNG' and min(picture.size)>1
        size=picture.size;picture.verify()
    pdf=paths[1].read_bytes()
    assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-1024:]
    with paths[2].open(newline='') as stream:
        reader=csv.DictReader(stream);assert reader.fieldnames
        count=sum(1 for _ in reader);assert count>0
    result.update(PNG_dimensions=list(size),CSV_rows=count,
        checks='PNG decoder verify, PDF header/EOF and nonempty parseable CSV. Not a semantic or full PDF-render audit.')
    result['files']=[dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,SHA256=digest(path)) for path in paths]
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',default='v0');args=parser.parse_args()
    assert args.version.startswith('v') and args.version[1:].isdigit();preserve()
    output=ROOT/'analysis'/f'delivery_requirement_inventory_{args.version}.json';assert not output.exists()
    reports=[]
    for family,names in [('original_main',REPORTS),('idle_addendum',IDLE_REPORTS)]:
        for name in names:
            path=ROOT.parent/name
            reports.append(dict(family=family,file=name,available=path.is_file(),
                SHA256=digest(path) if path.is_file() else None,
                scope='Named artifact availability only. Existing early reports can be superseded by later supplements.'))
    figures=[]
    for key,title,stems,evidence,csv_path in FIGURES:
        bundles=[figure_bundle(ROOT,stem,csv_path) for stem in stems]
        figures.append(dict(requirement=key,title=title,evidence_scope=evidence,
            artifact_available=all(r['available'] for r in bundles),bundles=bundles,
            scientific_PASS_inferred_from_availability=False))
    jobs=[]
    for tag in ['interference_PROBE_C40R4','suffix_QUANTA_C41R3','ec_INLINE_CAUSAL_C44','batch_COUNTER_C46']:
        marker=ROOT/'state'/f'{tag}_controller_complete.json'
        jobs.append(dict(tag=tag,terminal_marker_available=marker.exists(),
            marker_SHA256=digest(marker) if marker.exists() else None,
            scope='File availability only. Live PID and GPU lease checks are required separately.'))
    write(output,dict(time=time.time(),reports=reports,figures=figures,jobs=jobs,
        missing_report_files=[r['file'] for r in reports if not r['available']],
        missing_figure_requirements=[r['requirement'] for r in figures if not r['artifact_available']],
        goal_complete=False,scope='Intermediate delivery inventory. Not full completion proof, correctness, task quality, headroom or serving GO.'))
    table(ROOT/'analysis'/f'delivery_requirement_figures_{args.version}.csv',
        [dict(requirement=r['requirement'],title=r['title'],evidence_scope=r['evidence_scope'],
            artifact_available=r['artifact_available'],stems=';'.join(b['stem'] for b in r['bundles'])) for r in figures])
    preserve();print(json.dumps(dict(reports_available=sum(r['available'] for r in reports),reports_required=len(reports),
        figures_available=sum(r['artifact_available'] for r in figures),figures_required=len(figures),goal_complete=False)),flush=True)


if __name__=='__main__':main()
